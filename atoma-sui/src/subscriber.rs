use sui_sdk::{rpc_types::EventFilter, types::event::EventID, SuiClient};
use crate::config::SuiEventSubscriberConfig;

type Result<T> = std::result::Result<T, SuiEventSubscriberError>;

/// A subscriber for Sui blockchain events.
///
/// This struct provides functionality to subscribe to and process events
/// from the Sui blockchain based on specified filters.
pub struct SuiEventSubscriber {
    /// The Sui client used for interacting with the blockchain.
    client: SuiClient,
    /// The event filter used to specify which events to subscribe to.
    filter: EventFilter,
    /// The HTTP address of the RPC node.
    http_rpc_node_addr: String,
    /// The ID of the last processed event, used for pagination.
    last_event_id: Option<EventID>,
}

impl SuiEventSubscriber {
    /// Constructor
    pub fn new(client: SuiClient, filter: EventFilter, http_rpc_node_addr: String) -> Self {
        Self {
            client,
            filter,
            http_rpc_node_addr,
            last_event_id: None,
        }
    }

    pub fn new_from_config<P: AsRef<Path>>(config_path: P) -> Result<Self> {
        let config = SuiEventSubscriberConfig::from_file_path(config_path);
        let filter = EventFilter::new(config.package_id());
        Ok(Self::new(client, filter, config.http_rpc_node_addr()))
    }

    pub fn run(&self) -> Result<(), Error> {
        let events = self.client.read_events(
            self.http_rpc_node_addr,
            self.filter,
            self.last_event_id,
        )?;
    }
}

#[derive(Debug, Error)]
pub enum SuiEventSubscriberError {
    #[error("Failed to read events: {0}")]
    ReadEventsError(String),
}
