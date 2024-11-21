use prost::Message;

#[derive(Clone, Message)]
pub struct UpdatePublicAddress {
    #[prost(uint64, tag = "1")]
    node_small_id: u64,
    #[prost(string, tag = "2")]
    address: String,
}

impl UpdatePublicAddress {
    pub fn new(node_small_id: u64, address: &str) -> Self {
        Self {
            node_small_id,
            address: address.to_string(),
        }
    }
    pub fn node_small_id(&self) -> u64 {
        self.node_small_id
    }

    pub fn address(&self) -> &str {
        &self.address
    }
}
