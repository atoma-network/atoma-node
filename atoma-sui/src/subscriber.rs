use crate::{
    config::AtomaSuiConfig,
    events::{AtomaEvent, AtomaEventIdentifier, SuiEventParseError},
};
use flume::Sender;
use serde_json::Value;
use std::{path::Path, str::FromStr, time::Duration};
use sui_sdk::{
    rpc_types::{EventFilter, EventPage},
    types::{event::EventID, Identifier},
    SuiClient, SuiClientBuilder,
};
use thiserror::Error;
use tokio::sync::watch::Receiver;
use tracing::{error, info, instrument, trace};

/// The Atoma contract db module name.
const DB_MODULE_NAME: &str = "db";
/// The duration to wait for new events in seconds, if there are no new events.
const DURATION_TO_WAIT_FOR_NEW_EVENTS_IN_MILLIS: u64 = 100;

pub(crate) type Result<T> = std::result::Result<T, SuiEventSubscriberError>;

/// A subscriber for Sui blockchain events.
///
/// This struct provides functionality to subscribe to and process events
/// from the Sui blockchain based on specified filters.
pub struct SuiEventSubscriber {
    /// The configuration values for the subscriber.
    config: AtomaSuiConfig,
    /// The event filter used to specify which events to subscribe to.
    filter: EventFilter,
    /// Node small IDs
    /// These are values used to identify the Atoma's nodes that are under control by
    /// current Sui wallet
    node_small_ids: Option<Vec<u64>>,
    /// Task small IDs
    /// These are values used to identify the Atoma's tasks that are under control by
    /// current Sui wallet
    task_small_ids: Option<Vec<u64>>,
    /// Sender to stream each received event to the `AtomaStateManager` running task.
    state_manager_sender: Sender<AtomaEvent>,
    /// The shutdown signal.
    shutdown_signal: Receiver<bool>,
}

impl SuiEventSubscriber {
    /// Constructor
    pub fn new(
        config: AtomaSuiConfig,
        state_manager_sender: Sender<AtomaEvent>,
        shutdown_signal: Receiver<bool>,
    ) -> Self {
        let node_small_ids = if config.node_small_ids().is_empty() {
            None
        } else {
            Some(config.node_small_ids())
        };
        let task_small_ids = if config.task_small_ids().is_empty() {
            None
        } else {
            Some(config.task_small_ids())
        };
        let filter = EventFilter::MoveModule {
            package: config.atoma_package_id(),
            module: Identifier::new(DB_MODULE_NAME).unwrap(),
        };
        Self {
            config,
            filter,
            node_small_ids,
            task_small_ids,
            state_manager_sender,
            shutdown_signal,
        }
    }

    /// Creates a new `SuiEventSubscriber` instance from a configuration file.
    ///
    /// This method reads the configuration from the specified file path and initializes
    /// a new `SuiEventSubscriber` with the loaded configuration.
    ///
    /// # Arguments
    ///
    /// * `config_path` - A path-like type that represents the location of the configuration file.
    ///
    /// # Returns
    ///
    /// * `Result<Self>` - A Result containing the new `SuiEventSubscriber` instance if successful,
    ///   or an error if the configuration couldn't be read.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// * The configuration file cannot be read or parsed.
    pub fn new_from_config<P: AsRef<Path>>(
        config_path: P,
        state_manager_sender: Sender<AtomaEvent>,
        shutdown_signal: Receiver<bool>,
    ) -> Self {
        let config = AtomaSuiConfig::from_file_path(config_path);
        Self::new(config, state_manager_sender, shutdown_signal)
    }

    /// Builds a SuiClient based on the provided configuration.
    ///
    /// This asynchronous method creates a new SuiClient instance using the settings
    /// specified in the AtomaSuiConfig. It sets up the client with the
    /// configured request timeout and HTTP RPC node address.
    ///
    /// # Arguments
    ///
    /// * `config` - A reference to a AtomaSuiConfig containing the necessary
    ///              configuration parameters.
    ///
    /// # Returns
    ///
    /// * `Result<SuiClient>` - A Result containing the newly created SuiClient if successful,
    ///                         or a SuiEventSubscriberError if the client creation fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// * The SuiClient cannot be built with the provided configuration.
    /// * There's a network issue when connecting to the specified RPC node.
    #[instrument(level = "info", skip_all, fields(
        http_rpc_node_addr = %config.http_rpc_node_addr()
    ))]
    pub async fn build_client(config: &AtomaSuiConfig) -> Result<SuiClient> {
        let mut client_builder = SuiClientBuilder::default();
        if let Some(request_timeout) = config.request_timeout() {
            client_builder = client_builder.request_timeout(request_timeout);
        }
        let client = client_builder.build(config.http_rpc_node_addr()).await?;
        info!("Client built successfully");
        Ok(client)
    }

    /// Runs the event subscriber, continuously processing events from the Sui blockchain.
    ///
    /// This method enters an infinite loop that:
    /// 1. Queries for new events using the configured filter and cursor.
    /// 2. Processes each event concurrently using the specified number of tasks.
    /// 3. Updates the cursor for the next query.
    /// 4. Waits for a short duration if no new events are available.
    ///
    /// # Returns
    ///
    /// * `Result<()>` - A Result indicating success or an error if the subscription process fails.
    ///
    /// # Errors
    ///
    /// This method may return an error if:
    /// * There's a failure in building the Sui client.
    /// * Event querying encounters an error.
    /// * Event processing or handling fails (though these are currently logged and not propagated).
    #[instrument(level = "info", skip_all, fields(package_id))]
    pub async fn run(mut self) -> Result<()> {
        let package_id = self.config.atoma_package_id();
        let limit = self.config.limit();
        let client = Self::build_client(&self.config).await?;

        info!(
            target = "atoma-sui-subscriber",
            event = "subscriber-started",
            "Starting to run events subscriber, for package: {package_id}"
        );

        let mut cursor = read_cursor_from_toml_file(&self.config.cursor_path())?;
        loop {
            tokio::select! {
                    page = client.event_api().query_events(self.filter.clone(), cursor, limit, false) => {
                        let EventPage {
                            data,
                            next_cursor,
                            has_next_page,
                        } = match page {
                            Ok(page) => page,
                            Err(e) => {
                                error!(
                                    target = "atoma-sui-subscriber",
                                    event = "subscriber-read-events-error",
                                    "Failed to read paged events, with error: {e}"
                                );
                                continue;
                            }
                        };
                        cursor = next_cursor;

                        for sui_event in data {
                            let event_name = sui_event.type_.name;
                            trace!(
                                target = "atoma-sui-subscriber",
                                event = "subscriber-received-new-event",
                                event_name = %event_name,
                                "Received new event: {event_name:#?}"
                            );
                            match AtomaEventIdentifier::from_str(event_name.as_str()) {
                                Ok(atoma_event_id) => {
                                    let atoma_event =
                                        parse_event(&atoma_event_id, sui_event.parsed_json).await?;
                                    if filter_event(&atoma_event, self.node_small_ids.as_ref(), self.task_small_ids.as_ref()) {
                                        self.state_manager_sender
                                            .send(atoma_event)
                                            .map_err(Box::new)?;
                                    } else {
                                        continue;
                                    }
                                }
                                Err(e) => {
                                    error!(
                                        target = "atoma-sui-subscriber",
                                        event = "subscriber-event-parse-error",
                                        "Failed to parse event: {e}",
                                    );
                                    // NOTE: `AtomaEvent` didn't match any known event, so we skip it.
                                }
                            }
                        }

                        if !has_next_page {
                            // No new events to read, so let's wait for a while
                            info!(
                                target = "atoma-sui-subscriber",
                                event = "subscriber-no-new-events",
                                wait_duration = DURATION_TO_WAIT_FOR_NEW_EVENTS_IN_MILLIS,
                                "No new events to read, the node is now synced with the Atoma protocol, waiting until the next synchronization..."
                            );
                            tokio::time::sleep(Duration::from_millis(
                                DURATION_TO_WAIT_FOR_NEW_EVENTS_IN_MILLIS,
                                ))
                            .await;
                        }
                    }
                    shutdown_signal_changed = self.shutdown_signal.changed() => {
                        match shutdown_signal_changed {
                            Ok(()) => {
                                if *self.shutdown_signal.borrow() {
                                    info!(
                                    target = "atoma-sui-subscriber",
                                    event = "subscriber-stopped",
                                    "Shutdown signal received, gracefully stopping subscriber..."
                                );
                                // Update the config file with the current cursor
                                write_cursor_to_toml_file(cursor, &self.config.cursor_path())?;
                                break;
                            }
                        }
                        Err(e) => {
                            error!(
                                target = "atoma-sui-subscriber",
                                event = "subscriber-shutdown-signal-error",
                                "Failed to receive shutdown signal: {e}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Reads an event cursor from a TOML file.
///
/// This function attempts to read and parse an event cursor from the specified file path.
/// If the file doesn't exist, it will return `None`. If the file
/// exists, it will attempt to parse its contents as an `EventID`.
///
/// # Arguments
///
/// * `path` - A string slice containing the path to the TOML file
///
/// # Returns
///
/// * `Result<Option<EventID>>` - Returns:
///   * `Ok(Some(EventID))` if the file exists and was successfully parsed
///   * `Ok(None)` if the file doesn't exist (and was created)
///   * `Err(SuiEventSubscriberError)` if:
///     * The file exists but couldn't be read
///     * The file contents couldn't be parsed as TOML
///     * The file couldn't be created when not found
///
/// # Examples
///
/// ```rust,ignore
/// let path = "cursor.toml";
/// match read_cursor_from_toml_file(path) {
///     Ok(Some(cursor)) => println!("Read cursor: {:?}", cursor),
///     Ok(None) => println!("No cursor found, created empty file"),
///     Err(e) => eprintln!("Error reading cursor: {}", e),
/// }
/// ```
fn read_cursor_from_toml_file(path: &str) -> Result<Option<EventID>> {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(SuiEventSubscriberError::CursorFileError(e)),
    };

    Ok(Some(toml::from_str(&content)?))
}

/// Writes an event cursor to a TOML file.
///
/// This function takes an optional event cursor and writes it to the specified file path
/// in TOML format. If the cursor is `None`, no file will be written.
///
/// # Arguments
///
/// * `cursor` - An `Option<EventID>` representing the event cursor to be written
/// * `path` - A string slice containing the path where the TOML file should be written
///
/// # Returns
///
/// * `Result<()>` - Returns `Ok(())` if the write was successful, or an error if:
///   * The cursor serialization to TOML fails
///   * The file write operation fails
///
/// # Examples
///
/// ```rust,ignore
/// use sui_sdk::types::event::EventID;
///
/// let cursor = Some(EventID::default());
/// let path = "cursor.toml";
/// write_cursor_to_toml_file(cursor, path).expect("Failed to write cursor");
/// ```
fn write_cursor_to_toml_file(cursor: Option<EventID>, path: &str) -> Result<()> {
    if let Some(cursor) = cursor {
        let toml_str = toml::to_string(&cursor)?;
        std::fs::write(path, toml_str)?;
    }
    Ok(())
}

/// Handles various Atoma events by delegating to specific handler functions based on the event type.
///
/// This function serves as the main event dispatcher for the Atoma system, routing different event types
/// to their corresponding handler functions. For each event type, it either calls the appropriate handler
/// or returns an unimplemented error for events that are not yet supported.
///
/// # Arguments
///
/// * `event` - A reference to the `AtomaEvent` enum indicating the type of event to handle
/// * `value` - The serialized event data as a `serde_json::Value`
/// * `db` - A reference to the SQLite connection pool for database operations
/// * `node_small_ids` - A slice of node IDs that are relevant for the current context
///
/// # Returns
///
/// Returns a `Result<()>` which is:
/// * `Ok(())` if the event was handled successfully
/// * `Err(_)` if there was an error processing the event or if the event type is not yet implemented
///
/// # Event Types
///
/// Currently implemented events:
/// * `NodeSubscribedToTaskEvent` - Handles node task subscription events
/// * `NodeSubscriptionUpdatedEvent` - Handles updates to node task subscriptions
/// * `NodeUnsubscribedFromTaskEvent` - Handles node task unsubscription events
/// * `TaskRegisteredEvent` - Handles new task registration
/// * `TaskDeprecationEvent` - Handles task deprecation
/// * `StackCreatedEvent` - Handles stack creation
/// * `StackTrySettleEvent` - Handles stack settlement attempts
/// * `NewStackSettlementAttestationEvent` - Handles new stack settlement attestations
/// * `StackSettlementTicketEvent` - Handles stack settlement tickets
/// * `StackSettlementTicketClaimedEvent` - Handles claimed stack settlement tickets
/// * `StackAttestationDisputeEvent` - Handles stack attestation disputes
///
/// Unimplemented events will return an `unimplemented!()` error with a descriptive message.
///
/// # Examples
///
/// ```ignore
/// use atoma_state::SqlitePool;
/// use serde_json::Value;
///
/// async fn example(event: AtomaEvent, value: Value, db: &SqlitePool, node_ids: &[u64]) {
///     match handle_atoma_event(&event, value, db, node_ids).await {
///         Ok(()) => println!("Event handled successfully"),
///         Err(e) => eprintln!("Error handling event: {}", e),
///     }
/// }
/// ```
#[instrument(level = "trace", skip_all)]
async fn parse_event(event: &AtomaEventIdentifier, value: Value) -> Result<AtomaEvent> {
    match event {
        AtomaEventIdentifier::DisputeEvent => {
            Ok(AtomaEvent::DisputeEvent(serde_json::from_value(value)?))
        }
        AtomaEventIdentifier::SettledEvent => {
            Ok(AtomaEvent::SettledEvent(serde_json::from_value(value)?))
        }
        AtomaEventIdentifier::PublishedEvent => {
            Ok(AtomaEvent::PublishedEvent(serde_json::from_value(value)?))
        }
        AtomaEventIdentifier::NewlySampledNodesEvent => Ok(AtomaEvent::NewlySampledNodesEvent(
            serde_json::from_value(value)?,
        )),
        AtomaEventIdentifier::NodeRegisteredEvent => Ok(AtomaEvent::NodeRegisteredEvent(
            serde_json::from_value(value)?,
        )),
        AtomaEventIdentifier::NodeSubscribedToModelEvent => Ok(
            AtomaEvent::NodeSubscribedToModelEvent(serde_json::from_value(value)?),
        ),
        AtomaEventIdentifier::NodeUnsubscribedFromTaskEvent => Ok(
            AtomaEvent::NodeUnsubscribedFromTaskEvent(serde_json::from_value(value)?),
        ),
        AtomaEventIdentifier::NodeSubscribedToTaskEvent => Ok(
            AtomaEvent::NodeSubscribedToTaskEvent(serde_json::from_value(value)?),
        ),
        AtomaEventIdentifier::NodeSubscriptionUpdatedEvent => Ok(
            AtomaEvent::NodeSubscriptionUpdatedEvent(serde_json::from_value(value)?),
        ),
        AtomaEventIdentifier::TaskRegisteredEvent => Ok(AtomaEvent::TaskRegisteredEvent(
            serde_json::from_value(value)?,
        )),
        AtomaEventIdentifier::TaskDeprecationEvent => Ok(AtomaEvent::TaskDeprecationEvent(
            serde_json::from_value(value)?,
        )),
        AtomaEventIdentifier::FirstSubmissionEvent => Ok(AtomaEvent::FirstSubmissionEvent(
            serde_json::from_value(value)?,
        )),
        AtomaEventIdentifier::StackCreatedEvent => Ok(AtomaEvent::StackCreatedEvent(
            serde_json::from_value(value)?,
        )),
        AtomaEventIdentifier::StackTrySettleEvent => Ok(AtomaEvent::StackTrySettleEvent(
            serde_json::from_value(value)?,
        )),
        AtomaEventIdentifier::NewStackSettlementAttestationEvent => Ok(
            AtomaEvent::NewStackSettlementAttestationEvent(serde_json::from_value(value)?),
        ),
        AtomaEventIdentifier::StackSettlementTicketEvent => Ok(
            AtomaEvent::StackSettlementTicketEvent(serde_json::from_value(value)?),
        ),
        AtomaEventIdentifier::StackSettlementTicketClaimedEvent => Ok(
            AtomaEvent::StackSettlementTicketClaimedEvent(serde_json::from_value(value)?),
        ),
        AtomaEventIdentifier::StackAttestationDisputeEvent => Ok(
            AtomaEvent::StackAttestationDisputeEvent(serde_json::from_value(value)?),
        ),
        AtomaEventIdentifier::TaskRemovedEvent => {
            Ok(AtomaEvent::TaskRemovedEvent(serde_json::from_value(value)?))
        }
        AtomaEventIdentifier::RetrySettlementEvent => Ok(AtomaEvent::RetrySettlementEvent(
            serde_json::from_value(value)?,
        )),
        AtomaEventIdentifier::Text2ImagePromptEvent => Ok(AtomaEvent::Text2ImagePromptEvent(
            serde_json::from_value(value)?,
        )),
        AtomaEventIdentifier::Text2TextPromptEvent => Ok(AtomaEvent::Text2TextPromptEvent(
            serde_json::from_value(value)?,
        )),
    }
}

/// Filters events based on a list of small IDs.
///
/// This function checks if the given `AtomaEvent` is associated with any of the small IDs
/// provided in the `node_small_ids` and `task_small_ids` options. It returns `true` if the event
/// is relevant to the specified small IDs, and `false` otherwise.
///
/// # Arguments
///
/// * `event` - A reference to the `AtomaEvent` enum indicating the type of event to filter.
/// * `node_small_ids` - An optional reference to a vector of node IDs that are relevant for the current context.
/// * `task_small_ids` - An optional reference to a vector of task IDs that are relevant for the current context.
///
/// # Returns
///
/// Returns a `bool` indicating whether the event is associated with any of the small IDs:
/// * `true` if the event is relevant to the small IDs,
/// * `false` if it is not.
///
/// # Event Types
///
/// The function specifically checks for the following event types:
/// * `NodeSubscribedToTaskEvent`
/// * `NodeUnsubscribedFromTaskEvent`
/// * `NodeSubscriptionUpdatedEvent`
/// * `StackCreatedEvent`
/// * `StackTrySettleEvent`
/// * `NewStackSettlementAttestationEvent`
/// * `StackSettlementTicketEvent`
/// * `StackSettlementTicketClaimedEvent`
/// * `TaskDeprecationEvent`
/// * `TaskRemovedEvent`
///
/// For all other event types, the function returns `true`, indicating that they are not
/// filtered out by small IDs.
fn filter_event(
    event: &AtomaEvent,
    node_small_ids: Option<&Vec<u64>>,
    task_small_ids: Option<&Vec<u64>>,
) -> bool {
    match (node_small_ids, task_small_ids) {
        (Some(node_small_ids), Some(task_small_ids)) => match event {
            AtomaEvent::NodeSubscribedToTaskEvent(event) => {
                node_small_ids.contains(&event.node_small_id.inner)
                    && task_small_ids.contains(&event.task_small_id.inner)
            }
            AtomaEvent::NodeUnsubscribedFromTaskEvent(event) => {
                node_small_ids.contains(&event.node_small_id.inner)
                    && task_small_ids.contains(&event.task_small_id.inner)
            }
            AtomaEvent::NodeSubscriptionUpdatedEvent(event) => {
                node_small_ids.contains(&event.node_small_id.inner)
                    && task_small_ids.contains(&event.task_small_id.inner)
            }
            AtomaEvent::StackCreatedEvent(event) => {
                node_small_ids.contains(&event.selected_node_id.inner)
                    && task_small_ids.contains(&event.task_small_id.inner)
            }
            AtomaEvent::StackTrySettleEvent(event) => {
                node_small_ids.contains(&event.selected_node_id.inner)
                    || event
                        .requested_attestation_nodes
                        .iter()
                        .any(|id| node_small_ids.contains(&id.inner))
            }
            AtomaEvent::NewStackSettlementAttestationEvent(event) => {
                node_small_ids.contains(&event.attestation_node_id.inner)
            }
            AtomaEvent::StackSettlementTicketEvent(event) => {
                node_small_ids.contains(&event.selected_node_id.inner)
                    || event
                        .requested_attestation_nodes
                        .iter()
                        .any(|id| node_small_ids.contains(&id.inner))
            }
            AtomaEvent::StackSettlementTicketClaimedEvent(event) => {
                node_small_ids.contains(&event.selected_node_id.inner)
                    || event
                        .attestation_nodes
                        .iter()
                        .any(|id| node_small_ids.contains(&id.inner))
            }
            AtomaEvent::TaskDeprecationEvent(event) => {
                task_small_ids.contains(&event.task_small_id.inner)
            }
            AtomaEvent::TaskRemovedEvent(event) => {
                task_small_ids.contains(&event.task_small_id.inner)
            }
            _ => true,
        },
        (Some(node_small_ids), None) => match event {
            AtomaEvent::NodeSubscribedToTaskEvent(event) => {
                node_small_ids.contains(&event.node_small_id.inner)
            }
            AtomaEvent::NodeUnsubscribedFromTaskEvent(event) => {
                node_small_ids.contains(&event.node_small_id.inner)
            }
            AtomaEvent::NodeSubscriptionUpdatedEvent(event) => {
                node_small_ids.contains(&event.node_small_id.inner)
            }
            AtomaEvent::StackCreatedEvent(event) => {
                node_small_ids.contains(&event.selected_node_id.inner)
            }
            AtomaEvent::StackTrySettleEvent(event) => {
                node_small_ids.contains(&event.selected_node_id.inner)
                    || event
                        .requested_attestation_nodes
                        .iter()
                        .any(|id| node_small_ids.contains(&id.inner))
            }
            AtomaEvent::NewStackSettlementAttestationEvent(event) => {
                node_small_ids.contains(&event.attestation_node_id.inner)
            }
            AtomaEvent::StackSettlementTicketEvent(event) => {
                node_small_ids.contains(&event.selected_node_id.inner)
                    || event
                        .requested_attestation_nodes
                        .iter()
                        .any(|id| node_small_ids.contains(&id.inner))
            }
            AtomaEvent::StackSettlementTicketClaimedEvent(event) => {
                node_small_ids.contains(&event.selected_node_id.inner)
                    || event
                        .attestation_nodes
                        .iter()
                        .any(|id| node_small_ids.contains(&id.inner))
            }
            _ => true,
        },
        (None, Some(task_small_ids)) => match event {
            AtomaEvent::TaskDeprecationEvent(event) => {
                task_small_ids.contains(&event.task_small_id.inner)
            }
            AtomaEvent::TaskRemovedEvent(event) => {
                task_small_ids.contains(&event.task_small_id.inner)
            }
            AtomaEvent::StackCreatedEvent(event) => {
                task_small_ids.contains(&event.task_small_id.inner)
            }
            AtomaEvent::NodeSubscribedToTaskEvent(event) => {
                task_small_ids.contains(&event.task_small_id.inner)
            }
            AtomaEvent::NodeUnsubscribedFromTaskEvent(event) => {
                task_small_ids.contains(&event.task_small_id.inner)
            }
            AtomaEvent::NodeSubscriptionUpdatedEvent(event) => {
                task_small_ids.contains(&event.task_small_id.inner)
            }
            _ => true,
        },
        (None, None) => true,
    }
}

#[derive(Debug, Error)]
pub enum SuiEventSubscriberError {
    #[error("Failed to read events: {0}")]
    ReadEventsError(#[from] sui_sdk::error::Error),
    #[error("Failed to parse event: {0}")]
    SuiEventParseError(#[from] SuiEventParseError),
    #[error("Failed to deserialize event: {0}")]
    DeserializeError(#[from] serde_json::Error),
    #[error("Failed to send event to state manager: {0}")]
    SendEventError(#[from] Box<flume::SendError<AtomaEvent>>),
    #[error("Failed to read/write cursor to file: {0}")]
    CursorFileError(#[from] std::io::Error),
    #[error("Failed to serialize cursor: {0}")]
    SerializeCursorError(#[from] toml::ser::Error),
    #[error("Failed to deserialize cursor: {0}")]
    DeserializeCursorError(#[from] toml::de::Error),
}

#[cfg(test)]
mod tests {
    use crate::events::{
        NodeSmallId, NodeSubscribedToTaskEvent, NodeUnsubscribedFromTaskEvent, StackCreatedEvent,
        StackSmallId, TaskMetrics, TaskRegisteredEvent, TaskRole, TaskSmallId,
    };
    use sui_sdk::types::digests::TransactionDigest;
    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn test_filter_event_with_both_ids() {
        let node_small_ids = vec![1, 2, 3];
        let task_small_ids = vec![10, 20, 30];

        let event_subscribed = AtomaEvent::NodeSubscribedToTaskEvent(NodeSubscribedToTaskEvent {
            node_small_id: NodeSmallId { inner: 1 },
            task_small_id: TaskSmallId { inner: 10 },
            price_per_compute_unit: 0,
            max_num_compute_units: 0,
        });

        let event_unsubscribed =
            AtomaEvent::NodeUnsubscribedFromTaskEvent(NodeUnsubscribedFromTaskEvent {
                node_small_id: NodeSmallId { inner: 2 },
                task_small_id: TaskSmallId { inner: 20 },
            });

        let event_not_matched = AtomaEvent::NodeSubscribedToTaskEvent(NodeSubscribedToTaskEvent {
            node_small_id: NodeSmallId { inner: 4 },
            task_small_id: TaskSmallId { inner: 40 },
            price_per_compute_unit: 0,
            max_num_compute_units: 0,
        });

        assert!(filter_event(
            &event_subscribed,
            Some(&node_small_ids),
            Some(&task_small_ids)
        ));
        assert!(filter_event(
            &event_unsubscribed,
            Some(&node_small_ids),
            Some(&task_small_ids)
        ));
        assert!(!filter_event(
            &event_not_matched,
            Some(&node_small_ids),
            Some(&task_small_ids)
        ));
    }

    #[test]
    fn test_filter_event_with_only_node_ids() {
        let node_small_ids = vec![1, 2, 3];

        let event_subscribed = AtomaEvent::NodeSubscribedToTaskEvent(NodeSubscribedToTaskEvent {
            node_small_id: NodeSmallId { inner: 1 },
            task_small_id: TaskSmallId { inner: 10 },
            price_per_compute_unit: 0,
            max_num_compute_units: 0,
        });

        let event_not_matched = AtomaEvent::NodeSubscribedToTaskEvent(NodeSubscribedToTaskEvent {
            node_small_id: NodeSmallId { inner: 4 },
            task_small_id: TaskSmallId { inner: 40 },
            price_per_compute_unit: 0,
            max_num_compute_units: 0,
        });

        assert!(filter_event(&event_subscribed, Some(&node_small_ids), None));
        assert!(!filter_event(
            &event_not_matched,
            Some(&node_small_ids),
            None
        ));
    }

    #[test]
    fn test_filter_event_with_only_task_ids() {
        let task_small_ids = vec![10, 20, 30];

        let event_subscribed = AtomaEvent::NodeSubscribedToTaskEvent(NodeSubscribedToTaskEvent {
            node_small_id: NodeSmallId { inner: 4 },
            task_small_id: TaskSmallId { inner: 10 },
            price_per_compute_unit: 0,
            max_num_compute_units: 0,
        });

        let event_not_matched = AtomaEvent::NodeSubscribedToTaskEvent(NodeSubscribedToTaskEvent {
            node_small_id: NodeSmallId { inner: 5 },
            task_small_id: TaskSmallId { inner: 50 },
            price_per_compute_unit: 0,
            max_num_compute_units: 0,
        });

        assert!(filter_event(&event_subscribed, None, Some(&task_small_ids)));
        assert!(!filter_event(
            &event_not_matched,
            None,
            Some(&task_small_ids)
        ));
    }

    #[test]
    fn test_filter_event_with_no_ids() {
        let event_subscribed = AtomaEvent::NodeSubscribedToTaskEvent(NodeSubscribedToTaskEvent {
            node_small_id: NodeSmallId { inner: 1 },
            task_small_id: TaskSmallId { inner: 10 },
            price_per_compute_unit: 0,
            max_num_compute_units: 0,
        });

        assert!(filter_event(&event_subscribed, None, None));
    }

    #[test]
    fn test_filter_event_with_stack_created_event() {
        let node_small_ids = vec![1, 2, 3];
        let task_small_ids = vec![10, 20, 30];

        let event = AtomaEvent::StackCreatedEvent(StackCreatedEvent {
            selected_node_id: NodeSmallId { inner: 1 },
            task_small_id: TaskSmallId { inner: 10 },
            owner_address: "test".to_string(),
            stack_id: "test".to_string(),
            stack_small_id: StackSmallId { inner: 1 },
            num_compute_units: 0,
            price: 0,
        });

        assert!(filter_event(
            &event,
            Some(&node_small_ids),
            Some(&task_small_ids)
        ));
    }

    #[test]
    fn test_filter_event_with_unrelated_event() {
        let node_small_ids = vec![1, 2, 3];
        let task_small_ids = vec![10, 20, 30];

        let event = AtomaEvent::TaskRegisteredEvent(TaskRegisteredEvent {
            task_small_id: TaskSmallId { inner: 40 },
            role: TaskRole { inner: 0 },
            model_name: Some("test".to_string()),
            is_deprecated: false,
            valid_until_epoch: None,
            deprecated_at_epoch: None,
            optimizations: vec![],
            security_level: 0,
            minimum_reputation_score: Some(155),
            task_metrics: TaskMetrics {
                compute_unit: 0,
                time_unit: Some(0),
                value: Some(0),
            },
            task_id: "test".to_string(),
        });

        assert!(filter_event(
            &event,
            Some(&node_small_ids),
            Some(&task_small_ids)
        ));
    }

    #[test]
    fn test_read_cursor_from_empty_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_str().unwrap();

        let result = read_cursor_from_toml_file(path);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SuiEventSubscriberError::DeserializeCursorError(_)
        ));
    }

    #[test]
    fn test_read_cursor_from_valid_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_str().unwrap();

        // Create a valid EventID and write it to file
        let event_id = EventID {
            tx_digest: TransactionDigest::default(),
            event_seq: 0,
        };
        let toml_str = toml::to_string(&event_id).unwrap();
        std::fs::write(path, toml_str).unwrap();

        let result = read_cursor_from_toml_file(path);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_read_cursor_from_invalid_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_str().unwrap();

        // Write invalid TOML content
        std::fs::write(path, "invalid toml content").unwrap();

        let result = read_cursor_from_toml_file(path);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SuiEventSubscriberError::DeserializeCursorError(_)
        ));
    }

    #[test]
    fn test_write_cursor_none() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_str().unwrap();

        let result = write_cursor_to_toml_file(None, path);
        assert!(result.is_ok());

        // File should be empty
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.is_empty());
    }

    #[test]
    fn test_write_cursor_some() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_str().unwrap();

        let event_id = EventID {
            tx_digest: TransactionDigest::default(),
            event_seq: 0,
        };
        let result = write_cursor_to_toml_file(Some(event_id), path);
        assert!(result.is_ok());

        // Verify written content
        let content = std::fs::read_to_string(path).unwrap();
        let read_event_id: EventID = toml::from_str(&content).unwrap();
        assert_eq!(read_event_id, event_id);
    }

    #[test]
    fn test_write_cursor_to_readonly_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_str().unwrap();

        // Make file read-only
        let mut perms = std::fs::metadata(path).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(path, perms).unwrap();

        let event_id = EventID {
            tx_digest: TransactionDigest::default(),
            event_seq: 0,
        };
        let result = write_cursor_to_toml_file(Some(event_id), path);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SuiEventSubscriberError::CursorFileError(_)
        ));
    }
}
