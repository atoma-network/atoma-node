use serde::{Deserialize, Serialize};
use sui_sdk::types::crypto::{Signature, ToFromBytes};

use crate::{metrics::NodeMetrics, service::AtomaP2pNodeError};

/// The length of Sui's SDK `Signature` type in bytes
/// see <https://github.com/MystenLabs/sui/blob/main/crates/sui-types/src/crypto.rs#L809>
const SIGNATURE_LENGTH: usize = 97;

type Result<T, E = AtomaP2pNodeError> = std::result::Result<T, E>;

/// An enum representing different types of events that can be emitted by the Atoma P2P node.
pub enum AtomaP2pEvent {
    /// An event emitted when a node joins the Atoma network and registers its public URL,
    /// among its peers.
    NodeMetricsRegistrationEvent {
        /// The public URL of the node.
        public_url: String,

        /// The small ID of the node.
        node_small_id: u64,

        /// The timestamp of the event.
        timestamp: u64,

        /// The country of the node.
        country: String,

        /// The metrics of the node.
        node_metrics: NodeMetrics,
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
    /// Message containing a signed node message
    SignedNodeMessage(SignedNodeMessage),
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

/// Represents usage metrics collected from a node in the network
///
/// This struct contains essential information about a node's status and location
/// that is periodically published to the network for monitoring and analytics purposes.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeP2pMetadata {
    /// The public URL where the node can be accessed
    /// This is typically the endpoint that other nodes and clients use to connect
    pub node_public_url: String,

    /// A unique identifier for the node (smaller representation of the full ID)
    /// Used for quick identification and reference in the network
    pub node_small_id: u64,

    /// The country where the node is located (ISO country code)
    /// Used for geographical distribution analysis and latency optimization
    pub country: String,

    /// Unix timestamp indicating when the metrics were collected
    /// Helps track the freshness of the metrics and synchronize data across nodes
    pub timestamp: u64,
}

/// A struct containing a serialized message and its hash
pub struct SerializedMessage {
    /// The serialized message
    pub message: Vec<u8>,

    /// The hash of the serialized message
    pub hash: blake3::Hash,
}

/// A trait for serializing a message (with ciborium) and returning the hash of the serialized message
pub trait SerializeWithHash {
    /// Serialize the message and return the hash of the serialized message
    ///
    /// # Errors
    ///
    /// Returns an error if the message cannot be serialized
    /// or if the hash cannot be computed
    fn serialize_with_hash(&self) -> Result<SerializedMessage>;
}

/// A message containing usage metrics for a node.
///
/// This struct represents a signed message that includes the node's small ID,
/// public URL, and a cryptographic signature to ensure the authenticity of the metrics.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeMessage {
    /// The node metadata
    pub node_metadata: NodeP2pMetadata,

    /// The node metrics
    pub node_metrics: NodeMetrics,
}

impl SerializeWithHash for NodeMessage {
    fn serialize_with_hash(&self) -> Result<SerializedMessage> {
        let mut bytes = Vec::new();
        ciborium::into_writer(self, &mut bytes)
            .map_err(AtomaP2pNodeError::UsageMetricsSerializeError)?;
        Ok(SerializedMessage {
            hash: blake3::hash(&bytes),
            message: bytes,
        })
    }
}

/// A signed message containing a node message.
///
/// This struct represents a signed message that includes a node message and a cryptographic signature
/// to ensure the authenticity of the message.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct SignedNodeMessage {
    /// The node message
    pub node_message: NodeMessage,

    /// The signature of the node message
    #[serde(skip)]
    pub signature: Vec<u8>,
}

/// A trait for serializing a message (with ciborium)
///
/// This trait is used to serialize a message and return the serialized message
/// as a vector of bytes.
pub trait SerializeWithSignature {
    /// Serialize the message and return the serialized message
    /// as a vector of bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the message cannot be serialized
    fn serialize_with_signature(&self) -> Result<Vec<u8>>;

    /// Deserialize the message from a vector of bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the message cannot be deserialized
    fn deserialize_with_signature(data: &[u8]) -> Result<Self>
    where
        Self: Sized;
}

impl SerializeWithSignature for SignedNodeMessage {
    fn serialize_with_signature(&self) -> Result<Vec<u8>, AtomaP2pNodeError> {
        let mut serialized = self.signature.clone();
        let mut serialized_node_message = Vec::new();
        ciborium::into_writer(&self.node_message, &mut serialized_node_message)
            .map_err(AtomaP2pNodeError::UsageMetricsSerializeError)?;
        serialized.extend(serialized_node_message);
        Ok(serialized)
    }

    fn deserialize_with_signature(data: &[u8]) -> Result<Self, AtomaP2pNodeError> {
        let signature = Signature::from_bytes(&data[..SIGNATURE_LENGTH])
            .map_err(|e| AtomaP2pNodeError::SignatureParseError(e.to_string()))?;

        let sig_bytes = signature.as_ref();
        let remainder = &data[SIGNATURE_LENGTH..];

        let node_message = ciborium::from_reader(remainder)?;

        Ok(Self {
            node_message,
            signature: sig_bytes.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use sui_keys::keystore::{AccountKeystore, InMemKeystore};

    use super::*;

    #[test]
    fn test_serialization_and_deserialization() {
        let node_message = NodeMessage {
            node_metadata: NodeP2pMetadata {
                node_public_url: "https://example.com".to_string(),
                node_small_id: 1,
                country: "US".to_string(),
                timestamp: 1_718_275_200,
            },
            node_metrics: NodeMetrics::default(),
        };
        let keystore = InMemKeystore::new_insecure_for_tests(1);
        let active_address = keystore.addresses()[0];
        let serialized_node_message = node_message.serialize_with_hash().unwrap();
        let signature = keystore
            .sign_hashed(&active_address, serialized_node_message.hash.as_bytes())
            .expect("Failed to sign message");
        let should_be_signed_node_message = SignedNodeMessage {
            node_message,
            signature: signature.as_ref().to_vec(),
        };
        let serialized = should_be_signed_node_message
            .serialize_with_signature()
            .unwrap();
        let deserialized = SignedNodeMessage::deserialize_with_signature(&serialized).unwrap();
        assert_eq!(deserialized, should_be_signed_node_message);
    }
}
