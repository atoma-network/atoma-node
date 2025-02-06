use serde::{Deserialize, Serialize};

pub enum AtomaP2pEvent {
    /// An event emitted when a node joins the Atoma network and registers its public URL,
    /// among its peers.
    NodePublicUrlRegistrationEvent {
        /// The public URL of the node.
        public_url: String,

        /// The small ID of the node.
        node_small_id: u64,

        /// The timestamp of the event.
        timestamp: u64,

        /// The country of the node.
        country: String,
    },

    /// An event emitted when a node's small ID ownership needs to be verified.
    VerifyNodeSmallIdOwnership {
        /// The small ID of the node.
        node_small_id: u64,

        /// The Sui address of the node.
        sui_address: String,
    },
}

/// Enum representing different types of messages that can be gossiped across the Atoma network.
#[derive(Debug, Serialize, Deserialize)]
pub enum GossipMessage {
    /// Message containing usage metrics of a node
    UsageMetrics(UsageMetrics),
}

/// A message containing usage metrics for a node.
///
/// This struct represents a signed message that includes the node's small ID,
/// public URL, and a cryptographic signature to ensure the authenticity of the metrics.
#[derive(Debug, Serialize, Deserialize)]
pub struct UsageMetrics {
    /// The node small ID of the node
    pub node_small_id: u64,

    /// The public URL of the node
    pub node_public_url: String,

    /// The country where the node is located
    pub country: String,

    /// A cryptographic signature of the usage metrics
    pub signature: Vec<u8>,

    /// The timestamp of the usage metrics
    pub timestamp: u64,
}
