use sui_sdk::{rpc_types::EventFilter, types::event::EventID, SuiClient};

pub struct SuiEventSubscriber {
    client: SuiClient,
    filter: EventFilter,
    http_rpc_node_addr: String,
    last_event_id: Option<EventID>,
}

impl SuiEventSubscriber {
    pub fn new(client: SuiClient, filter: EventFilter, http_rpc_node_addr: String) -> Self {
        Self {
            client,
            filter,
            http_rpc_node_addr,
            last_event_id: None,
        }
    }
}
