use crate::{
    config::AtomaSuiConfig,
    events::{AtomaEvent, SuiEventParseError},
    handlers::{handle_atoma_event, handle_event_with_retries},
};
use atoma_state::SqlitePool;
use futures::stream::{self, StreamExt};
use std::{path::Path, str::FromStr, time::Duration};
use sui_sdk::{
    rpc_types::{EventFilter, EventPage},
    types::event::EventID,
    SuiClient, SuiClientBuilder,
};
use thiserror::Error;
use tokio::sync::watch::Receiver;
use tracing::{error, info, instrument, trace};

/// The duration to wait for new events in seconds, if there are no new events.
const DURATION_TO_WAIT_FOR_NEW_EVENTS_IN_MILLIS: u64 = 100;
/// The default number of concurrent event handling tasks to run.
const DEFAULT_NUM_CONCURRENT_TASKS: usize = 32;

pub(crate) type Result<T> = std::result::Result<T, SuiEventSubscriberError>;

/// A subscriber for Sui blockchain events.
///
/// This struct provides functionality to subscribe to and process events
/// from the Sui blockchain based on specified filters.
pub struct SuiEventSubscriber {
    /// The configuration values for the subscriber.
    config: AtomaSuiConfig,
    /// The database url.
    database_url: String,
    /// The event filter used to specify which events to subscribe to.
    filter: EventFilter,
    /// The ID of the last processed event, used for pagination.
    cursor: Option<EventID>,
    /// The shutdown signal.
    shutdown_signal: Receiver<bool>,
}

impl SuiEventSubscriber {
    /// Constructor
    pub fn new(
        config: AtomaSuiConfig,
        database_url: String,
        shutdown_signal: Receiver<bool>,
    ) -> Self {
        let filter = EventFilter::Package(config.atoma_package_id());
        Self {
            config,
            database_url,
            filter,
            cursor: None,
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
        database_url: String,
        shutdown_signal: Receiver<bool>,
    ) -> Self {
        let config = AtomaSuiConfig::from_file_path(config_path);
        Self::new(config, database_url, shutdown_signal)
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
        let num_concurrent_tasks = self
            .config
            .num_concurrent_tasks()
            .unwrap_or(DEFAULT_NUM_CONCURRENT_TASKS);

        let client = Self::build_client(&self.config).await?;
        let db = SqlitePool::connect(&self.database_url)
            .await
            .map_err(|e| SuiEventSubscriberError::StateManagerError(e.into()))?;
        let node_small_ids = self.config.small_ids();

        info!("Starting to run events subscriber, for package: {package_id}");

        loop {
            if *self.shutdown_signal.borrow() {
                info!("Shutdown signal received, gracefully stopping subscriber...");
                break;
            }

            let event_filter = self.filter.clone();
            let EventPage {
                data,
                next_cursor,
                has_next_page,
            } = match client
                .event_api()
                .query_events(event_filter, self.cursor, limit, false)
                .await
            {
                Ok(page) => page,
                Err(e) => {
                    error!("Failed to read events, with error: {e}");
                    continue;
                }
            };
            self.cursor = next_cursor;

            let db = db.clone();
            stream::iter(data)
                .for_each_concurrent(num_concurrent_tasks, |sui_event| {
                    let db = db.clone();
                    let node_small_ids = node_small_ids.clone();
                    async move {
                        let event_name = sui_event.type_.name;
                        trace!("Received new event: {event_name:#?}");
                        match AtomaEvent::from_str(event_name.as_str()) {
                            Ok(atoma_event) => {
                                match handle_atoma_event(
                                    &atoma_event,
                                    sui_event.parsed_json.clone(),
                                    &db,
                                    &node_small_ids,
                                )
                                .await
                                {
                                    Ok(_) => {
                                        trace!(
                                            "Event with name: {event_name} handled successfully"
                                        );
                                    }
                                    Err(e) => {
                                        error!("Failed to handle event: {e}");
                                        handle_event_with_retries(
                                            &atoma_event,
                                            sui_event.parsed_json,
                                            &db,
                                            &node_small_ids,
                                        )
                                        .await
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Failed to parse event: {e}");
                                // NOTE: `AtomaEvent` didn't match any known event, so we skip it.
                            }
                        }
                    }
                })
                .await;

            if !has_next_page {
                // No new events to read, so let's wait for a while
                info!("No new events to read, so let's wait for {DURATION_TO_WAIT_FOR_NEW_EVENTS_IN_MILLIS} millis");
                tokio::time::sleep(Duration::from_millis(
                    DURATION_TO_WAIT_FOR_NEW_EVENTS_IN_MILLIS,
                ))
                .await;
            }
        }

        Ok(())
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
    #[error("State manager error: {0}")]
    StateManagerError(#[from] atoma_state::StateManagerError),
}
