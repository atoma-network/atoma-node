use config::ConfigError;
use libp2p::{
    gossipsub::{ConfigBuilderError, PublishError, SubscriptionError},
    swarm::DialError,
    TransportError,
};
use thiserror::Error;

use crate::service::StateManagerEvent;

#[derive(Debug, Error)]
pub enum AtomaP2pNodeError {
    #[error("Failed to build gossipsub config: {0}")]
    GossipsubConfigError(#[from] ConfigBuilderError),
    #[error("Failed to build behaviour: {0}")]
    BehaviourBuildError(String),
    #[error("Failed to subscribe to topic: {0}")]
    GossipsubSubscriptionError(#[from] SubscriptionError),
    #[error("Failed to listen on address: {0}")]
    SwarmListenOnError(#[from] TransportError<std::io::Error>),
    #[error("Failed to dial bootstrap node: {0}")]
    BootstrapNodeDialError(#[from] DialError),
    #[error("Failed to parse signature: {0}")]
    SignatureParseError(String),
    #[error("Failed to verify signature: {0}")]
    SignatureVerificationError(String),
    #[error("Invalid public address: {0}")]
    InvalidPublicAddressError(String),
    #[error("Failed to parse listen address: {0}")]
    ListenAddressParseError(#[from] libp2p::multiaddr::Error),
    #[error("Failed to build TCP config: {0}")]
    TcpConfigBuildError(#[from] ConfigError),
    #[error("Failed to initialize noise encryption: {0}")]
    NoiseError(#[from] libp2p::noise::Error),
    #[error("Failed to send event to state manager: {0}")]
    StateManagerError(#[from] flume::SendError<StateManagerEvent>),
    #[error("Failed to sign hashed message, with error: {0}")]
    SignatureError(String),
    #[error("Failed to publish gossipsub message: {0}")]
    GossipsubMessagePublishError(#[from] PublishError),
    #[error("Failed to verify node small ID ownership: {0}")]
    NodeSmallIdOwnershipVerificationError(String),
    #[error("Failed to send usage metrics")]
    UsageMetricsSendError,
    #[error("Failed to serialize usage metrics: `{0}`")]
    UsageMetricsSerializeError(#[from] ciborium::ser::Error<std::io::Error>),
    #[error("Failed to deserialize usage metrics: `{0}`")]
    UsageMetricsDeserializeError(#[from] ciborium::de::Error<std::io::Error>),
    #[error("Failed to parse URL: `{0}`")]
    UrlParseError(#[from] url::ParseError),
    #[error("Country code is invalid: `{0}`")]
    InvalidCountryCodeError(String),
    #[error("Validation error: `{0}`")]
    ValidationError(#[from] validator::ValidationError),
    #[error("Failed to compute usage metrics: `{0}`")]
    UsageMetricsComputeError(#[from] crate::metrics::NodeMetricsError),
    #[error("Invalid config: `{0}`")]
    InvalidConfig(String),
    #[error("Invalid message length")]
    InvalidMessageLengthError,
}
