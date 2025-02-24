use crate::{broadcast_metrics::NodeMetrics, errors::AtomaP2pNodeError};
use serde::{Deserialize, Serialize};
use sui_sdk::types::crypto::{
    Ed25519SuiSignature, Secp256k1SuiSignature, Secp256r1SuiSignature, SuiSignatureInner,
};

/// The length of Sui's SDK ed25519 `Signature` type in bytes
/// see <https://github.com/MystenLabs/sui/blob/main/crates/sui-types/src/crypto.rs#L809>
pub const ED25519_SIGNATURE_LENGTH: usize = 97;

/// The length of Sui's SDK secp256k1 `Signature` type in bytes
/// see <https://github.com/MystenLabs/sui/blob/main/crates/sui-types/src/crypto.rs#L843>
pub const SECP256K1_SIGNATURE_LENGTH: usize = 98;

/// The length of Sui's SDK secp256r1 `Signature` type in bytes
/// see <https://github.com/MystenLabs/sui/blob/main/crates/sui-types/src/crypto.rs#L891>
pub const SECP256R1_SIGNATURE_LENGTH: usize = 98;

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
        let signature_len = data
            .first()
            .map(|&flag| match flag {
                f if f == Ed25519SuiSignature::SCHEME.flag() => Ok(ED25519_SIGNATURE_LENGTH),
                f if f == Secp256k1SuiSignature::SCHEME.flag() => Ok(SECP256K1_SIGNATURE_LENGTH),
                f if f == Secp256r1SuiSignature::SCHEME.flag() => Ok(SECP256R1_SIGNATURE_LENGTH),
                f => Err(AtomaP2pNodeError::SignatureParseError(format!(
                    "Invalid signature scheme, expected 0x00, 0x01 or 0x02, received {f:#04x}",
                ))),
            })
            .ok_or_else(|| {
                AtomaP2pNodeError::SignatureParseError(
                    "Invalid signature scheme: the data is empty".to_string(),
                )
            })??;
        let node_message = ciborium::from_reader(&data[signature_len..])?;
        Ok(Self {
            node_message,
            signature: data[..signature_len].to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use fastcrypto::{
        hash::{Digest, HashFunction},
        secp256k1::Secp256k1KeyPair,
        secp256r1::Secp256r1KeyPair,
        traits::KeyPair,
    };
    use sui_keys::keystore::{AccountKeystore, InMemKeystore};
    use sui_sdk::types::crypto::{Signature, ToFromBytes};

    use super::*;

    #[test]
    fn test_serialization_and_deserialization_with_ed25519_signature() {
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

    #[derive(Default)]
    struct Blake3Hash {
        hasher: blake3::Hasher,
    }

    impl HashFunction<32> for Blake3Hash {
        fn new() -> Self {
            Self {
                hasher: blake3::Hasher::new(),
            }
        }

        fn update<Data: AsRef<[u8]>>(&mut self, data: Data) {
            self.hasher.update(data.as_ref());
        }

        fn finalize(self) -> Digest<32> {
            let hash = self.hasher.finalize();
            Digest {
                digest: *hash.as_bytes(),
            }
        }

        /// Compute the digest of the given data and consume the hash function.
        fn digest<Data: AsRef<[u8]>>(data: Data) -> Digest<32> {
            let mut hasher = Self::new();
            hasher.update(data);
            hasher.finalize()
        }
    }

    #[test]
    fn test_serialization_and_deserialization_with_secp256k1_signature() {
        let node_message = NodeMessage {
            node_metadata: NodeP2pMetadata {
                node_public_url: "https://example.com".to_string(),
                node_small_id: 1,
                country: "US".to_string(),
                timestamp: 1_718_275_200,
            },
            node_metrics: NodeMetrics::default(),
        };
        let serialized_node_message = node_message.serialize_with_hash().unwrap();
        let keypair = Secp256k1KeyPair::generate(&mut rand::thread_rng());
        let public_key = keypair.public();
        let signature =
            keypair.sign_with_hash::<Blake3Hash>(serialized_node_message.hash.as_bytes());
        let mut signature_bytes = Vec::with_capacity(SECP256K1_SIGNATURE_LENGTH);
        signature_bytes.push(Secp256k1SuiSignature::SCHEME.flag());
        signature_bytes.extend_from_slice(signature.as_ref());
        signature_bytes.extend_from_slice(public_key.as_ref());
        let signature = Secp256k1SuiSignature::from_bytes(&signature_bytes).unwrap();
        let signature = Signature::Secp256k1SuiSignature(signature);
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

    #[test]
    fn test_serialization_and_deserialization_with_secp256r1_signature() {
        let node_message = NodeMessage {
            node_metadata: NodeP2pMetadata {
                node_public_url: "https://example.com".to_string(),
                node_small_id: 1,
                country: "US".to_string(),
                timestamp: 1_718_275_200,
            },
            node_metrics: NodeMetrics::default(),
        };
        let serialized_node_message = node_message.serialize_with_hash().unwrap();
        let keypair = Secp256r1KeyPair::generate(&mut rand::thread_rng());
        let public_key = keypair.public();
        let signature =
            keypair.sign_with_hash::<Blake3Hash>(serialized_node_message.hash.as_bytes());
        let mut signature_bytes = Vec::with_capacity(SECP256K1_SIGNATURE_LENGTH);
        signature_bytes.push(Secp256r1SuiSignature::SCHEME.flag());
        signature_bytes.extend_from_slice(signature.as_ref());
        signature_bytes.extend_from_slice(public_key.as_ref());
        let signature = Secp256r1SuiSignature::from_bytes(&signature_bytes).unwrap();
        let signature = Signature::Secp256r1SuiSignature(signature);
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
