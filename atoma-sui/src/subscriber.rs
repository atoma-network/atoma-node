use crate::{
    config::SuiEventSubscriberConfig,
    events::{AtomaEvent, SuiEventParseError}, handlers::handle_atoma_event,
};
use std::{path::Path, str::FromStr, time::Duration};
use sui_sdk::{
    rpc_types::{EventFilter, EventPage},
    types::event::EventID,
    SuiClient, SuiClientBuilder,
};
use thiserror::Error;
use tracing::{error, info, info_span, instrument, trace, Span};

/// The duration to wait for new events in seconds, if there are no new events.
const DURATION_TO_WAIT_FOR_NEW_EVENTS_IN_MILLIS: u64 = 100;

type Result<T> = std::result::Result<T, SuiEventSubscriberError>;

/// A subscriber for Sui blockchain events.
///
/// This struct provides functionality to subscribe to and process events
/// from the Sui blockchain based on specified filters.
pub struct SuiEventSubscriber {
    /// The configuration values for the subscriber.
    config: SuiEventSubscriberConfig,
    /// The event filter used to specify which events to subscribe to.
    filter: EventFilter,
    /// The ID of the last processed event, used for pagination.
    cursor: Option<EventID>,
    /// The span used to trace the events subscriber.
    span: Span,
}

impl SuiEventSubscriber {
    /// Constructor
    pub fn new(config: SuiEventSubscriberConfig) -> Self {
        let filter = EventFilter::Package(config.package_id());
        Self {
            config,
            filter,
            cursor: None,
            span: info_span!("events-subscriber"),
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
    pub fn new_from_config<P: AsRef<Path>>(config_path: P) -> Result<Self> {
        let config = SuiEventSubscriberConfig::from_file_path(config_path);
        Ok(Self::new(config))
    }

    /// Builds a SuiClient based on the provided configuration.
    ///
    /// This asynchronous method creates a new SuiClient instance using the settings
    /// specified in the SuiEventSubscriberConfig. It sets up the client with the
    /// configured request timeout and HTTP RPC node address.
    ///
    /// # Arguments
    ///
    /// * `config` - A reference to a SuiEventSubscriberConfig containing the necessary
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
    #[instrument(skip_all, fields(
        http_rpc_node_addr = %config.http_rpc_node_addr()
    ))]
    pub async fn build_client(config: &SuiEventSubscriberConfig) -> Result<SuiClient> {
        let client = SuiClientBuilder::default()
            .request_timeout(config.request_timeout())
            .build(config.http_rpc_node_addr())
            .await?;
        info!("Client built successfully");
        Ok(client)
    }

    #[instrument(skip_all)]
    pub async fn run(mut self) -> Result<()> {
        let package_id = self.config.package_id();
        let limit = self.config.limit();

        let client = Self::build_client(&self.config).await?;

        info!("Starting to run events subscriber, for package: {package_id}");

        loop {
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
            for sui_event in data.iter() {
                trace!("Received new event: {sui_event:#?}");
                let atoma_event = AtomaEvent::from_str(&sui_event.type_.name.as_str())?;
                handle_atoma_event(atoma_event)?;
            }

            if !has_next_page {
                // No new events to read, so let's wait for a while
                info!("No new events to read, so let's wait for {DURATION_TO_WAIT_FOR_NEW_EVENTS_IN_MILLIS} millis");
                tokio::time::sleep(Duration::from_millis(
                    DURATION_TO_WAIT_FOR_NEW_EVENTS_IN_MILLIS,
                ))
                .await;
                continue;
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum SuiEventSubscriberError {
    #[error("Failed to read events: {0}")]
    ReadEventsError(#[from] sui_sdk::error::Error),
    #[error("Failed to parse event: {0}")]
    SuiEventParseError(#[from] SuiEventParseError),
}
