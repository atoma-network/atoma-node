use crate::{
    config::Config as SuiConfig,
    events::{
        AtomaEvent, AtomaEventIdentifier, StackCreateAndUpdateEvent, StackCreatedEvent,
        SuiEventParseError,
    },
};
use flume::Sender;
use serde_json::Value;
use std::{path::Path, str::FromStr, time::Duration};
use sui_sdk::{
    rpc_types::{EventFilter, EventPage, SuiTransactionBlockResponseOptions},
    types::{base_types::SuiAddress, digests::TransactionDigest, event::EventID, Identifier},
    SuiClient, SuiClientBuilder,
};
use thiserror::Error;
use tokio::sync::{
    mpsc::{self, UnboundedSender},
    oneshot,
    watch::Receiver,
};
use tracing::{error, info, instrument, trace};

/// The Atoma contract db module name.
const DB_MODULE_NAME: &str = "db";

/// The duration to wait for new events in seconds, if there are no new events.
const DURATION_TO_WAIT_FOR_NEW_EVENTS_IN_MILLIS: u64 = 100;

pub(crate) type Result<T> = std::result::Result<T, SuiEventSubscriberError>;

/// Represents the number of compute units available, stored as a 64-bit unsigned integer.
type ComputeUnits = i64;
/// Represents the small identifier for a stack, stored as a 64-bit unsigned integer.
type StackSmallId = i64;

/// Represents the result of a blockchain query for stack information.
type StackQueryResult = (Option<StackSmallId>, Option<ComputeUnits>);

/// Represents a receiver for stack retrieval requests.
pub(crate) type StackRetrieveReceiver = mpsc::UnboundedReceiver<(
    TransactionDigest,
    ComputeUnits,
    StackSmallId,
    oneshot::Sender<StackQueryResult>,
)>;

/// A subscriber for Sui blockchain events.
///
/// This struct provides functionality to subscribe to and process events
/// from the Sui blockchain based on specified filters.
pub struct Subscriber {
    /// The configuration values for the subscriber.
    config: SuiConfig,

    /// The event filter used to specify which events to subscribe to.
    filter: EventFilter,

    /// Sender to stream each received event to the `AtomaStateManager` running task.
    state_manager_sender: Sender<AtomaEvent>,

    /// Sender to stream confidential compute requests to the `AtomaTDX` running task.
    confidential_compute_service_sender: UnboundedSender<AtomaEvent>,

    //// Channel receiver to handle transaction digest queries from the `AtomaService` task.
    /// Receives tuples containing:
    /// - A transaction digest to query
    /// - A oneshot sender to respond with the number of events found (if any)
    stack_retrieve_receiver: StackRetrieveReceiver,

    /// The shutdown signal.
    shutdown_signal: Receiver<bool>,
}

impl Subscriber {
    /// Constructor
    ///
    /// # Panics
    /// - If identifier creation fails for DB module name
    /// - If event filtering setup fails
    #[must_use]
    pub fn new(
        config: SuiConfig,
        state_manager_sender: Sender<AtomaEvent>,
        stack_retrieve_receiver: StackRetrieveReceiver,
        confidential_compute_service_sender: UnboundedSender<AtomaEvent>,
        shutdown_signal: Receiver<bool>,
    ) -> Self {
        let filter = EventFilter::MoveModule {
            package: config.atoma_package_id(),
            module: Identifier::new(DB_MODULE_NAME).unwrap(),
        };
        Self {
            config,
            filter,
            state_manager_sender,
            confidential_compute_service_sender,
            stack_retrieve_receiver,
            shutdown_signal,
        }
    }

    /// Creates a new `Subscriber` instance from a configuration file.
    ///
    /// This method reads the configuration from the specified file path and initializes
    /// a new `Subscriber` with the loaded configuration.
    ///
    /// # Arguments
    ///
    /// * `config_path` - A path-like type that represents the location of the configuration file.
    ///
    /// # Returns
    ///
    /// * `Result<Self>` - A Result containing the new `Subscriber` instance if successful,
    ///   or an error if the configuration couldn't be read.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// * The configuration file cannot be read or parsed.
    pub fn new_from_config<P: AsRef<Path>>(
        config_path: P,
        state_manager_sender: Sender<AtomaEvent>,
        stack_retrieve_receiver: StackRetrieveReceiver,
        confidential_compute_service_sender: UnboundedSender<AtomaEvent>,
        shutdown_signal: Receiver<bool>,
    ) -> Self {
        let config = SuiConfig::from_file_path(config_path);
        Self::new(
            config,
            state_manager_sender,
            stack_retrieve_receiver,
            confidential_compute_service_sender,
            shutdown_signal,
        )
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
    pub async fn build_client(config: &SuiConfig) -> Result<SuiClient> {
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
    /// This method enters an infinite loop that handles three main types of operations:
    ///
    /// 1. Stack Retrieval:
    ///    - Receives transaction digests and responds with compute units information
    ///    - Processes StackCreatedEvents from transactions and forwards them to the state manager
    ///
    /// 2. Event Processing:
    ///    - Queries for new events using the configured filter and cursor
    ///    - Parses and filters events based on node and task IDs
    ///    - Forwards relevant events to the state manager
    ///    - Updates the cursor periodically (every CURSOR_FILE_UPDATE_ITERATIONS)
    ///    - Implements backoff when no new events are available
    ///
    /// 3. Shutdown Handling:
    ///    - Monitors a shutdown signal
    ///    - Performs graceful shutdown by saving the current cursor
    ///
    /// # Returns
    ///
    /// * `Result<()>` - A Result indicating success or an error if the subscription process fails.
    ///
    /// # Errors
    ///
    /// This method may return an error if:
    /// * There's a failure in building the Sui client
    /// * Event querying encounters an error
    /// * Stack retrieval operations fail
    /// * Event processing fails
    /// * Writing the cursor file fails
    /// * Communication with the state manager fails
    #[instrument(level = "trace", skip_all, fields(package_id))]
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
                    Some((tx_digest, estimated_compute_units, selected_stack_small_id, result_sender)) = self.stack_retrieve_receiver.recv() => {
                        let tx_events = client
                            .read_api()
                            .get_transaction_with_options(
                                tx_digest,
                                SuiTransactionBlockResponseOptions {
                                    show_events: true, ..Default::default()
                                }
                            )
                            .await?
                            .events;
                        let mut compute_units = None;
                        let mut stack_small_id = None;
                        if let Some(tx_events) = tx_events {
                            for event in &tx_events.data {
                                let event_identifier = AtomaEventIdentifier::from_str(event.type_.name.as_str())?;
                                if event_identifier == AtomaEventIdentifier::StackCreatedEvent {
                                    // NOTE: In this case, the transaction contains a stack creation event,
                                    // which means that whoever made a request to the service has already paid
                                    // to buy new compute units.
                                    // We need to count the compute units used by the transaction.
                                    let event: StackCreatedEvent = serde_json::from_value(event.parsed_json.clone())?;
                                    if event.stack_small_id.inner != selected_stack_small_id as u64 {
                                        // NOTE: This is a safety check to ensure that the stack small id
                                        // is the same as the one defined in the original transaction
                                        continue;
                                    }
                                    if estimated_compute_units > event.num_compute_units as i64 {
                                        // NOTE: If the estimated compute units are greater than the event compute units,
                                        // this means that whoever made a request to the service has requested more compute units
                                        // than those that it paid for. In this case, we should not process the event, and break
                                        // out of the loop. This will send `None` values to the Atoma service, which will
                                        // trigger an error back to the client.
                                        // SAFETY: It is fine if we do not process the [`StackCreatedEvent`] right away, as it will
                                        // be catched later by the Sui's event subscriber.
                                        error!(
                                            target = "atoma-sui-subscriber",
                                            event = "subscriber-stack-create-event-error",
                                            "Stack create event with id {} has more compute units than the transaction used, this is not possible",
                                            event.stack_small_id.inner
                                        );
                                        break;
                                    }
                                    #[allow(clippy::cast_possible_wrap)]
                                    let event: StackCreateAndUpdateEvent = (event, estimated_compute_units as i64).into();
                                    // NOTE: We also send the event to the state manager, so it can be processed
                                    // right away.
                                    compute_units = Some(event.num_compute_units as i64);
                                    stack_small_id = Some(event.stack_small_id.inner as i64);
                                    self.state_manager_sender
                                        .send(AtomaEvent::StackCreateAndUpdateEvent(event))
                                        .map_err(Box::new)?;
                                    // We found the stack creation event, so we can break out of the loop
                                    break;
                                }
                            }
                        }
                        // Send the compute units to the Atoma service, so it can be used to validate the
                        // request.
                        result_sender
                            .send((stack_small_id, compute_units))
                            .map_err(|_| SuiEventSubscriberError::SendComputeUnitsError)?;
                    }
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
                                    let sender = sui_event.sender;
                                    let atoma_event = match parse_event(&atoma_event_id, sui_event.parsed_json, sender, sui_event.timestamp_ms).await {
                                        Ok(atoma_event) => atoma_event,
                                        Err(e) => {
                                            error!(
                                                target = "atoma-sui-subscriber",
                                                event = "subscriber-event-parse-error",
                                                event_name = %event_name,
                                                "Failed to parse event: {e}",
                                            );
                                            continue;
                                        }
                                    };
                                    if filter_event(
                                        &atoma_event,
                                        self.config.node_small_ids().as_ref(),
                                        self.config.task_small_ids().as_ref(),
                                    ) {
                                        self.handle_atoma_event(atoma_event_id, atoma_event).await?;
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
                            // Update the cursor file with the current cursor
                            write_cursor_to_toml_file(cursor, &self.config.cursor_path())?;
                            // No new events to read, so let's wait for a while
                            trace!(
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

    /// Handles an Atoma event by sending it to the appropriate service.
    ///
    /// This method routes events to either the confidential compute service or the state manager
    /// based on the event type. Specifically:
    /// - `NewKeyRotationEvent` events are sent to the confidential compute service
    /// - All other events are sent to the state manager
    ///
    /// # Arguments
    ///
    /// * `atoma_event_id` - The identifier specifying the type of Atoma event
    /// * `atoma_event` - The actual event data to be processed
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the event was successfully sent to the appropriate service,
    /// or an error if sending failed.
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - Sending to the confidential compute service fails (`SuiEventSubscriberError::SendComputeUnitsError`)
    /// - Sending to the state manager fails (wrapped `flume::SendError`)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use your_crate::{AtomaEventIdentifier, AtomaEvent, SuiEventSubscriber};
    /// # async fn example(subscriber: &SuiEventSubscriber) -> Result<(), Box<dyn std::error::Error>> {
    /// let event_id = AtomaEventIdentifier::TaskRegisteredEvent;
    /// let event = AtomaEvent::TaskRegisteredEvent(/* ... */);
    /// subscriber.handle_atoma_event(event_id, event).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(level = "trace", skip_all, fields(event_id))]
    async fn handle_atoma_event(
        &self,
        atoma_event_id: AtomaEventIdentifier,
        atoma_event: AtomaEvent,
    ) -> Result<()> {
        if atoma_event_id == AtomaEventIdentifier::NewKeyRotationEvent {
            self.confidential_compute_service_sender
                .send(atoma_event)
                .map_err(|e| {
                    error!(
                        target = "atoma-sui-subscriber",
                        event = "subscriber-send-new-key-rotation-event-error",
                        "Failed to send new key rotation event: {e}"
                    );
                    SuiEventSubscriberError::SendComputeUnitsError
                })?;
        } else {
            self.state_manager_sender.send(atoma_event).map_err(|e| {
                error!(
                    target = "atoma-sui-subscriber",
                    event = "subscriber-send-event-error",
                    "Failed to send event: {e}"
                );
                Box::new(e)
            })?;
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
/// * `NewKeyRotationEvent` - Handles new key rotation requests
/// * `NodePublicKeyCommittmentEvent` - Handles node key rotation commitment events
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
async fn parse_event(
    event: &AtomaEventIdentifier,
    value: Value,
    sender: SuiAddress,
    timestamp_ms: Option<u64>,
) -> Result<AtomaEvent> {
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
        AtomaEventIdentifier::NodeRegisteredEvent => Ok(AtomaEvent::NodeRegisteredEvent((
            serde_json::from_value(value)?,
            sender,
        ))),
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
        AtomaEventIdentifier::StackCreatedEvent => Ok(AtomaEvent::StackCreatedEvent((
            serde_json::from_value(value)?,
            timestamp_ms,
        ))),
        AtomaEventIdentifier::StackTrySettleEvent => Ok(AtomaEvent::StackTrySettleEvent((
            serde_json::from_value(value)?,
            timestamp_ms,
        ))),
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
        AtomaEventIdentifier::NewKeyRotationEvent => Ok(AtomaEvent::NewKeyRotationEvent(
            serde_json::from_value(value)?,
        )),
        AtomaEventIdentifier::NodePublicKeyCommittmentEvent => Ok(
            AtomaEvent::NodePublicKeyCommittmentEvent(serde_json::from_value(value)?),
        ),
    }
}

fn filter_event_by_node(event: &AtomaEvent, node_small_ids: &[u64]) -> bool {
    match event {
        AtomaEvent::NodeRegisteredEvent((event, _)) => {
            node_small_ids.contains(&event.node_small_id.inner)
        }
        AtomaEvent::NodeSubscribedToModelEvent(event) => {
            node_small_ids.contains(&event.node_small_id.inner)
        }
        AtomaEvent::NodeSubscribedToTaskEvent(event) => {
            node_small_ids.contains(&event.node_small_id.inner)
        }
        AtomaEvent::NodeUnsubscribedFromTaskEvent(event) => {
            node_small_ids.contains(&event.node_small_id.inner)
        }
        AtomaEvent::NodeSubscriptionUpdatedEvent(event) => {
            node_small_ids.contains(&event.node_small_id.inner)
        }
        _ => true,
    }
}

fn filter_event_by_task(event: &AtomaEvent, task_small_ids: &[u64]) -> bool {
    match event {
        AtomaEvent::TaskDeprecationEvent(event) => {
            task_small_ids.contains(&event.task_small_id.inner)
        }
        AtomaEvent::TaskRemovedEvent(event) => task_small_ids.contains(&event.task_small_id.inner),
        AtomaEvent::StackCreatedEvent((event, _)) => {
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
    }
}

fn filter_event(
    event: &AtomaEvent,
    node_small_ids: Option<&Vec<u64>>,
    task_small_ids: Option<&Vec<u64>>,
) -> bool {
    match (node_small_ids, task_small_ids) {
        (Some(node_ids), Some(task_ids)) => {
            filter_event_by_node(event, node_ids) && filter_event_by_task(event, task_ids)
        }
        (Some(node_ids), None) => filter_event_by_node(event, node_ids),
        (None, Some(task_ids)) => filter_event_by_task(event, task_ids),
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
    #[error("Failed to send compute units to state manager")]
    SendComputeUnitsError,
    #[error("Failed to read/write cursor to file: {0}")]
    CursorFileError(#[from] std::io::Error),
    #[error("Failed to serialize cursor: {0}")]
    SerializeCursorError(#[from] toml::ser::Error),
    #[error("Failed to deserialize cursor: {0}")]
    DeserializeCursorError(#[from] toml::de::Error),
    #[error("Failed to convert stack small id: {0}")]
    ConversionError(#[from] std::num::TryFromIntError),
}

#[cfg(test)]
mod tests {
    use crate::events::{
        NodeSmallId, NodeSubscribedToTaskEvent, NodeUnsubscribedFromTaskEvent, SecurityLevel,
        StackCreatedEvent, StackSmallId, TaskRegisteredEvent, TaskRole, TaskSmallId,
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
            price_per_one_million_compute_units: 0,
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
            price_per_one_million_compute_units: 0,
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
            price_per_one_million_compute_units: 0,
            max_num_compute_units: 0,
        });

        let event_not_matched = AtomaEvent::NodeSubscribedToTaskEvent(NodeSubscribedToTaskEvent {
            node_small_id: NodeSmallId { inner: 4 },
            task_small_id: TaskSmallId { inner: 40 },
            price_per_one_million_compute_units: 0,
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
            price_per_one_million_compute_units: 0,
            max_num_compute_units: 0,
        });

        let event_not_matched = AtomaEvent::NodeSubscribedToTaskEvent(NodeSubscribedToTaskEvent {
            node_small_id: NodeSmallId { inner: 5 },
            task_small_id: TaskSmallId { inner: 50 },
            price_per_one_million_compute_units: 0,
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
            price_per_one_million_compute_units: 0,
            max_num_compute_units: 0,
        });

        assert!(filter_event(&event_subscribed, None, None));
    }

    #[test]
    fn test_filter_event_with_stack_created_event() {
        let node_small_ids = vec![1, 2, 3];
        let task_small_ids = vec![10, 20, 30];

        let event = AtomaEvent::StackCreatedEvent((
            StackCreatedEvent {
                selected_node_id: NodeSmallId { inner: 1 },
                task_small_id: TaskSmallId { inner: 10 },
                owner: "test".to_string(),
                stack_id: "test".to_string(),
                stack_small_id: StackSmallId { inner: 1 },
                num_compute_units: 0,
                price_per_one_million_compute_units: 0,
            },
            None,
        ));

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
            security_level: SecurityLevel { inner: 0 },
            minimum_reputation_score: Some(155),
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
