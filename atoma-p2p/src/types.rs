use serde::{Deserialize, Serialize};

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
