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
    },

    /// An event emitted when a node's small ID ownership needs to be verified.
    VerifyNodeSmallIdOwnership {
        /// The small ID of the node.
        node_small_id: u64,

        /// The Sui address of the node.
        sui_address: String,
    },
}

/// A message requesting the public address of the available nodes on the network.  
/// This message is used to request the public address of the sender from the network.  
///
/// This message is not signed and is not used to authenticate the sender.
#[derive(Debug, Serialize, Deserialize)]
pub struct AddressRequest {
    /// The timestamp of the request
    pub timestamp: u64,
}

/// A message containing a public address and its authentication details.
///
/// This struct represents a signed message that associates a public address with
/// a public key and includes a timestamp to ensure freshness. The signature
/// proves that the sender controls the private key corresponding to the public key.
#[derive(Debug, Serialize, Deserialize)]
pub struct AddressResponse {
    /// The public address string to be associated with the sender
    pub address: String,

    /// A cryptographic signature proving ownership of the private key
    pub signature: Vec<u8>,

    /// Unix timestamp indicating when the message was created
    pub timestamp: u64,

    /// The node's small id (assigned by the Atoma smart contract)
    pub node_small_id: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum GossipMessage {
    AddressRequest,
    AddressResponse(AddressResponse),
}
