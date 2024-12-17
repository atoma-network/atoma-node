use crate::{
    config::P2pAtomaNodeConfig,
    types::{AddressResponse, AtomaP2pEvent, GossipMessage},
};
use flume::Sender;
use futures::StreamExt;
use libp2p::{
    gossipsub::{self, ConfigBuilderError},
    noise,
    swarm::{DialError, NetworkBehaviour, SwarmEvent},
    tcp, yamux, Multiaddr, PeerId, Swarm, SwarmBuilder, TransportError,
};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use sui_keys::keystore::{AccountKeystore, FileBasedKeystore};
use thiserror::Error;
use tokio::sync::{oneshot, watch};
use tracing::{debug, error, instrument, trace};

/// The topic that the P2P network will use to gossip messages
const GOSPUBSUB_TOPIC: &str = "atoma-p2p";

type StateManagerEvent = (AtomaP2pEvent, Option<oneshot::Sender<bool>>);

/// Network behavior configuration for the P2P Atoma node, combining multiple libp2p protocols.
///
/// This struct implements the `NetworkBehaviour` trait and coordinates three main networking components
/// for peer discovery, message broadcasting, and distributed routing.
#[derive(NetworkBehaviour)]
struct MyBehaviour {
    /// Handles publish-subscribe messaging across the P2P network.
    /// Used for broadcasting node addresses and other network messages using a gossip protocol
    /// that ensures efficient message propagation.
    gossipsub: gossipsub::Behaviour,
}

/// A P2P node implementation for the Atoma network that handles peer discovery,
/// message broadcasting, and network communication.
pub struct P2pAtomaNode {
    /// The cryptographic keystore containing the node's keys for signing messages
    /// and managing identities. Wrapped in an Arc for thread-safe shared access.
    keystore: Arc<FileBasedKeystore>,

    /// The libp2p swarm that manages all network behaviors and connections.
    /// Handles peer discovery, message routing, and protocol negotiations using
    /// the configured MyBehaviour protocols (gossipsub).
    swarm: Swarm<MyBehaviour>,

    /// The publicly accessible URL of this node that other peers can use to connect.
    /// This URL is shared with other nodes during peer discovery and address exchange.
    public_url: String,

    /// A compact numerical identifier for the node, used in network messages and logging.
    /// Provides a shorter alternative to the full node ID for quick reference and debugging.
    node_small_id: u64,

    /// Sender channel to the state manager
    /// Used to send events to the state manager for storing authenticated
    /// gossipsub messages (containing public URLs of registered nodes)
    state_manager_sender: Sender<StateManagerEvent>,

    /// Whether this peer is a client or a node in the Atoma network.
    /// In the case of a client, it will store the public URLs of the participating nodes
    /// in the protocol. In the case, it is a node, it does not store the public URLs of the
    /// participating nodes, neither any public URL of clients. That said, a node must
    /// always share its own public URL with the peers in the network.
    is_client: bool,
}

impl P2pAtomaNode {
    /// Initializes and configures a new P2P Atoma node with networking capabilities.
    ///
    /// This method sets up a complete libp2p node with the following features:
    /// - TCP and QUIC transport layers with noise encryption and yamux multiplexing
    /// - Gossipsub for peer-to-peer message broadcasting
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters for the P2P node including:
    ///   - `heartbeat_interval` - Interval for gossipsub protocol heartbeats
    ///   - `idle_connection_timeout` - Duration after which inactive connections are closed
    ///   - `listen_addr` - Network address the node will listen on
    ///   - `seed_nodes` - List of bootstrap nodes to connect to initially
    ///   - `public_url` - Publicly accessible URL for this node
    ///   - `node_small_id` - Compact numerical identifier for the node
    ///
    /// * `keystore` - Thread-safe reference to a file-based keystore containing the node's
    ///   cryptographic keys for signing messages and establishing secure connections
    ///
    /// * `state_manager_sender` - Sender channel to the state manager
    ///
    /// * `is_client` - Whether this peer is a client or a node in the Atoma network.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing either:
    /// - `Ok(P2pAtomaNode)` - A fully configured P2P node ready to start
    /// - `Err(P2pAtomaNodeError)` - An error indicating what went wrong during setup
    ///
    /// # Errors
    ///
    /// This function can return several error types:
    /// - `SwarmBuildError` - Failed to create the libp2p swarm
    /// - `BehaviourBuildError` - Failed to configure networking behaviors
    /// - `GossipsubSubscriptionError` - Failed to subscribe to the gossip topic
    /// - `SwarmListenOnError` - Failed to bind to the specified network address
    /// - `ListenAddressParseError` - Invalid listen address format
    /// - `SeedNodeDialError` - Failed to connect to a seed node
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_p2p::{P2pAtomaNode, P2pAtomaNodeConfig};
    /// use std::sync::Arc;
    /// use sui_keys::keystore::FileBasedKeystore;
    ///
    /// async fn setup_node() -> Result<P2pAtomaNode, P2pAtomaNodeError> {
    ///     let config = P2pAtomaNodeConfig {
    ///         heartbeat_interval: std::time::Duration::from_secs(1),
    ///         idle_connection_timeout: std::time::Duration::from_secs(30),
    ///         listen_addr: "/ip4/127.0.0.1/tcp/8080".to_string(),
    ///         seed_nodes: vec![],
    ///         public_url: "node1.example.com".to_string(),
    ///         node_small_id: 1,
    ///     };
    ///     let keystore = Arc::new(FileBasedKeystore::new(&std::path::PathBuf::from("keys"))?);
    ///     
    ///     P2pAtomaNode::start(config, keystore, state_manager_sender, false)
    /// }
    /// ```
    ///
    /// # Network Architecture
    ///
    /// The node uses a layered networking approach:
    /// 1. Transport Layer: TCP/QUIC with noise encryption
    /// 2. Protocol Layer:
    ///    - Gossipsub for message broadcasting
    /// 3. Application Layer: Custom message handling for node addresses and requests
    ///
    /// # Security Considerations
    ///
    /// - All network connections are encrypted using the noise protocol
    /// - Messages are signed using the node's private key
    /// - Peer connections are authenticated
    /// - The node validates all incoming messages
    #[instrument(level = "info", skip_all)]
    pub fn start(
        config: P2pAtomaNodeConfig,
        keystore: Arc<FileBasedKeystore>,
        state_manager_sender: Sender<StateManagerEvent>,
        is_client: bool,
    ) -> Result<Self, P2pAtomaNodeError> {
        let mut swarm = SwarmBuilder::with_new_identity()
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )
            .map_err(|e| {
                error!(
                    target = "atoma-p2p",
                    event = "build_swarm",
                    error = %e,
                    "Failed to build swarm"
                );
                P2pAtomaNodeError::SwarmBuildError(e.to_string())
            })?
            .with_quic()
            .with_behaviour(|key| {
                // To content-address message, we can take the hash of message and use it as an ID.
                let message_id_fn = |message: &gossipsub::Message| {
                    let mut s = DefaultHasher::new();
                    message.data.hash(&mut s);
                    gossipsub::MessageId::from(s.finish().to_string())
                };

                let gossipsub_config = gossipsub::ConfigBuilder::default()
                    .heartbeat_interval(config.heartbeat_interval)
                    .validation_mode(gossipsub::ValidationMode::Strict)
                    .message_id_fn(message_id_fn)
                    .build()?;

                let gossipsub = gossipsub::Behaviour::new(
                    gossipsub::MessageAuthenticity::Signed(key.clone()),
                    gossipsub_config,
                )?;

                Ok(MyBehaviour { gossipsub })
            })
            .map_err(|e| {
                error!(
                    target = "atoma-p2p",
                    event = "build_behaviour",
                    error = %e,
                    "Failed to build behaviour"
                );
                P2pAtomaNodeError::BehaviourBuildError(e.to_string())
            })?
            .with_swarm_config(|c| c.with_idle_connection_timeout(config.idle_connection_timeout))
            .build();

        let topic = gossipsub::IdentTopic::new(GOSPUBSUB_TOPIC);
        swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&topic)
            .map_err(|e| {
                error!(
                    target = "atoma-p2p",
                    event = "subscribe_to_topic",
                    error = %e,
                    "Failed to subscribe to topic"
                );
                P2pAtomaNodeError::GossipsubSubscriptionError(e)
            })?;
        swarm.listen_on(config.listen_addr.parse()?).map_err(|e| {
            error!(
                target = "atoma-p2p",
                event = "listen_on_error",
                listen_addr = config.listen_addr,
                error = %e,
                "Failed to listen on address"
            );
            P2pAtomaNodeError::SwarmListenOnError(e)
        })?;

        for seed_addr in config.seed_nodes {
            match seed_addr.parse::<Multiaddr>() {
                Ok(addr) => {
                    swarm.dial(addr).map_err(|e| {
                        error!(
                            target = "atoma-p2p",
                            event = "dial_seed_node",
                            seed_node = seed_addr,
                            error = %e,
                            "Failed to dial seed node"
                        );
                        P2pAtomaNodeError::SeedNodeDialError(e)
                    })?;
                    debug!(
                        target = "atoma-p2p",
                        event = "dialed_seed_node",
                        seed_node = seed_addr,
                        "Dialed seed node"
                    );
                }
                Err(e) => {
                    error!(
                        target = "atoma-p2p",
                        event = "seed_node_parse_error",
                        seed_node = seed_addr,
                        error = %e,
                        "Failed to parse seed node address"
                    );
                }
            }
        }

        utils::publish_start_message(
            &mut swarm,
            &keystore,
            &config.public_url,
            config.node_small_id,
            is_client,
        )?;

        Ok(Self {
            swarm,
            public_url: config.public_url,
            node_small_id: config.node_small_id,
            keystore,
            state_manager_sender,
            is_client,
        })
    }

    /// Starts the P2P node's main event loop, handling network events and shutdown signals.
    ///
    /// This method runs an infinite loop that processes:
    /// - Network events from the libp2p swarm (gossipsub messages, peer discovery, routing updates)
    /// - Shutdown signals for graceful termination
    ///
    /// # Event Handling
    ///
    /// ## Gossipsub Messages
    /// Processes incoming messages on the gossip network using `handle_gossipsub_message`.
    ///
    /// ## MDNS Events
    /// - Discovered: Adds newly discovered peers to the gossipsub mesh
    /// - Expired: Removes expired peers from the gossipsub mesh
    ///
    /// ## Kademlia Events
    /// - RoutingUpdated: Updates the gossipsub peer list when Kademlia routing changes
    ///
    /// # Arguments
    ///
    /// * `self` - Takes ownership of the P2P node instance
    /// * `shutdown_signal` - A watch channel receiver for shutdown coordination
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` when the node shuts down gracefully, or a `P2pAtomaNodeError` if an error occurs.
    ///
    /// # Errors
    ///
    /// Can return errors from:
    /// - Gossipsub message handling
    /// - Network event processing
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use tokio::sync::watch;
    /// use atoma_p2p::{P2pAtomaNode, P2pAtomaNodeConfig};
    ///
    /// async fn run_node(node: P2pAtomaNode) {
    ///     let (shutdown_tx, shutdown_rx) = watch::channel(false);
    ///     
    ///     // Run the node in the background
    ///     let node_handle = tokio::spawn(node.run(shutdown_rx));
    ///     
    ///     // Signal shutdown when needed
    ///     shutdown_tx.send(true).expect("Failed to send shutdown signal");
    ///     
    ///     // Wait for the node to shut down
    ///     node_handle.await.expect("Node failed to shut down cleanly");
    /// }
    /// ```
    ///
    /// # Shutdown Behavior
    ///
    /// The node will shut down when either:
    /// - The shutdown signal is set to `true`
    /// - The shutdown channel is closed
    ///
    /// In both cases, the node will break from its event loop and terminate gracefully.
    #[instrument(level = "info", skip(self))]
    pub async fn run(
        mut self,
        mut shutdown_signal: watch::Receiver<bool>,
    ) -> Result<(), P2pAtomaNodeError> {
        loop {
            tokio::select! {
                event = self.swarm.select_next_some() => {
                    match event {
                        SwarmEvent::Behaviour(MyBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                            message_id,
                            message,
                            propagation_source,
                        })) => {
                            match self.handle_gossipsub_message(&message.data, &message_id, &propagation_source).await {
                                Ok(_) => {}
                                Err(e) => {
                                    error!(
                                        target = "atoma-p2p",
                                        event = "gossipsub_message_error",
                                        error = %e,
                                        "Failed to handle gossipsub message"
                                    );
                                }
                            }
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::Gossipsub(gossipsub::Event::Subscribed {
                            peer_id,
                            topic,
                        })) => {
                            debug!(
                                target = "atoma-p2p",
                                event = "gossipsub_subscribed",
                                peer_id = %peer_id,
                                topic = %topic,
                                "Peer subscribed to topic"
                            );
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::Gossipsub(gossipsub::Event::Unsubscribed {
                            peer_id,
                            topic,
                        })) => {
                            debug!(
                                target = "atoma-p2p",
                                event = "gossipsub_unsubscribed",
                                peer_id = %peer_id,
                                topic = %topic,
                                "Peer unsubscribed from topic"
                            );
                            // TODO: should we remove the locally stored public url from an unsubscribed peer?
                        }
                        _ => {}
                    }
                }
                shutdown_signal_changed = shutdown_signal.changed() => {
                    match shutdown_signal_changed {
                        Ok(()) => {
                            if *shutdown_signal.borrow() {
                                tracing::trace!(
                                    target = "atoma-p2p",
                                    event = "shutdown_signal",
                                    "Shutdown signal received, shutting down"
                                );
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                target = "atoma-p2p",
                                event = "shutdown_signal_error",
                                error = %e,
                                "Shutdown signal channel closed"
                            );
                            // NOTE: We want to break here as well, since no one can signal shutdown anymore
                            break;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Handles incoming gossipsub messages in the P2P network.
    ///
    /// This method processes two types of messages:
    /// 1. Address Response messages - Contains node address information with cryptographic proof
    /// 2. Address Request messages - Requests for node address information
    ///
    /// # Message Flow
    ///
    /// ## Address Response
    /// When receiving an address response:
    /// 1. Deserializes the CBOR message data
    /// 2. Extracts address, signature, timestamp, and node ID
    /// 3. Verifies the cryptographic signature by:
    ///    - Creating a message hash from the address, node ID, and timestamp
    ///    - Validating the signature against this hash
    /// 4. If valid, rebroadcasts the message to other peers
    /// 5. If invalid, drops the message and returns an error
    ///
    /// ## Address Request
    /// When receiving an address request:
    /// 1. Generates current timestamp
    /// 2. Creates a message hash from the node's public URL, ID, and timestamp
    /// 3. Signs the hash using the node's keystore
    /// 4. Constructs and serializes an AddressResponse using CBOR
    /// 5. Publishes the response to the network
    ///
    /// # Note
    ///
    /// We do not re-publish the node's own message, just return `Ok(())`
    ///
    /// # Arguments
    ///
    /// * `message_data` - Raw bytes of the received gossipsub message (CBOR encoded)
    /// * `message_id` - Unique identifier for the gossipsub message
    /// * `propagation_source` - The peer that forwarded us this message
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if message processing succeeds, or an error variant if any step fails.
    ///
    /// # Errors
    ///
    /// This function can return several error types:
    /// * `GossipsubMessageDataParseError` - If CBOR message deserialization fails
    /// * `SignatureVerificationError` - If signature verification fails for an address response
    /// * `GossipsubMessageRebroadcastError` - If rebroadcasting the message fails
    /// * `SignHashedMessageError` - If signing the response message fails
    ///
    /// # Security Considerations
    ///
    /// - Messages are cryptographically signed to prevent tampering
    /// - Timestamps are included to prevent replay attacks
    /// - Invalid signatures cause messages to be dropped without rebroadcast
    /// - Uses blake3 for secure message hashing
    /// - Uses CBOR for efficient binary serialization
    ///
    /// # Message Format
    ///
    /// The messages are serialized using CBOR, but here's a JSON representation for readability:
    ///
    /// ```json
    /// // Address Response
    /// {
    ///     "address": "node1.example.com:8080",
    ///     "signature": [bytes],
    ///     "timestamp": 1234567890,
    ///     "node_small_id": 42
    /// }
    /// ```
    ///
    /// The actual wire format uses CBOR encoding for more efficient binary representation
    /// and better handling of binary data like signatures.
    #[instrument(level = "debug", skip_all)]
    pub async fn handle_gossipsub_message(
        &mut self,
        message_data: &[u8],
        message_id: &gossipsub::MessageId,
        propagation_source: &PeerId,
    ) -> Result<(), P2pAtomaNodeError> {
        debug!(
            target = "atoma-p2p",
            event = "gossipsub_message",
            message_id = %message_id,
            propagation_source = %propagation_source,
            "Received gossipsub message"
        );
        if propagation_source == self.swarm.local_peer_id() {
            trace!(
                target = "atoma-p2p",
                event = "gossipsub_message_from_self",
                "Gossipsub message from self"
            );
            // Do not re-publish the node's own message, just return `Ok(())
            return Ok(());
        }
        let gossip_message: GossipMessage = serde_cbor::from_slice(message_data)?;
        match gossip_message {
            GossipMessage::AddressResponse(response) => {
                debug!(
                    target = "atoma-p2p",
                    event = "gossipsub_message_data",
                    "Received gossipsub message data"
                );
                // Validate the message
                utils::validate_public_address_message(&response)?;
                // Check if the signature is valid
                let AddressResponse {
                    address,
                    signature,
                    timestamp,
                    node_small_id,
                } = response;
                let message_hash = blake3::hash(
                    &[
                        address.as_bytes(),
                        &node_small_id.to_le_bytes(),
                        &timestamp.to_le_bytes(),
                    ]
                    .concat(),
                );
                // Verify the signature of the message
                utils::verify_signature(signature.as_slice(), message_hash.as_bytes())?;
                // Verify the node small ID ownership
                utils::verify_node_small_id_ownership(
                    node_small_id,
                    signature.as_slice(),
                    self.state_manager_sender.clone(),
                )
                .await?;
                // Rebroadcast the message to the network peers
                let topic = gossipsub::IdentTopic::new(GOSPUBSUB_TOPIC);
                if let Err(e) = self
                    .swarm
                    .behaviour_mut()
                    .gossipsub
                    .publish(topic, message_data)
                {
                    error!(
                        target = "atoma-p2p",
                        event = "gossipsub_message_rebroadcast_error",
                        error = %e,
                        "Failed to rebroadcast gossipsub message"
                    );
                    return Err(P2pAtomaNodeError::GossipsubMessageRebroadcastError(
                        e.to_string(),
                    ));
                }
                // If the current peer is a client, we need to store the public URL in the state manager
                if self.is_client {
                    let event = AtomaP2pEvent::NodePublicUrlRegistrationEvent {
                        public_url: address,
                        node_small_id,
                        timestamp,
                    };
                    self.state_manager_sender.send((event, None)).map_err(|e| {
                        error!(
                            target = "atoma-p2p",
                            event = "gossipsub_message_state_manager_error",
                            error = %e,
                            "Failed to send event to state manager"
                        );
                        P2pAtomaNodeError::StateManagerError(e)
                    })?;
                }
            }
            GossipMessage::AddressRequest => {
                debug!(
                    target = "atoma-p2p",
                    event = "gossipsub_message_data",
                    "Received gossipsub message data"
                );
                let timestamp = std::time::Instant::now().elapsed().as_secs();
                let message_hash = blake3::hash(
                    &[
                        self.public_url.as_bytes(),
                        &self.node_small_id.to_le_bytes(),
                        &timestamp.to_le_bytes(),
                    ]
                    .concat(),
                );
                let address = self.keystore.addresses()[0];
                let signature = self
                    .keystore
                    .sign_hashed(&address, message_hash.as_bytes())
                    .map_err(|e| {
                        error!(
                            target = "atoma-p2p",
                            event = "gossipsub_message_sign_hashed_error",
                            error = %e,
                            "Failed to sign hashed message"
                        );
                        P2pAtomaNodeError::SignHashedMessageError(e.to_string())
                    })?;
                let response = GossipMessage::AddressResponse(AddressResponse {
                    address: self.public_url.clone(),
                    signature: signature.as_ref().to_vec(),
                    timestamp,
                    node_small_id: self.node_small_id,
                });
                let serialized_response = serde_cbor::to_vec(&response)?;
                let topic = gossipsub::IdentTopic::new(GOSPUBSUB_TOPIC);
                if let Err(e) = self
                    .swarm
                    .behaviour_mut()
                    .gossipsub
                    .publish(topic, serialized_response)
                {
                    error!(
                        target = "atoma-p2p",
                        event = "gossipsub_message_rebroadcast_error",
                        error = %e,
                        "Failed to rebroadcast gossipsub message"
                    );
                    return Err(P2pAtomaNodeError::GossipsubMessageRebroadcastError(
                        e.to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum P2pAtomaNodeError {
    #[error("Failed to build gossipsub config: {0}")]
    GossipsubConfigError(#[from] ConfigBuilderError),
    #[error("Failed to build gossipsub: {0}")]
    GossipsubBuildError(String),
    #[error("Failed to build swarm: {0}")]
    SwarmBuildError(String),
    #[error("Failed to build behaviour: {0}")]
    BehaviourBuildError(String),
    #[error("Failed to subscribe to topic: {0}")]
    GossipsubSubscriptionError(#[from] gossipsub::SubscriptionError),
    #[error("Failed to listen on address: {0}")]
    SwarmListenOnError(#[from] TransportError<std::io::Error>),
    #[error("Failed to parse listen address: {0}")]
    ListenAddressParseError(#[from] libp2p::multiaddr::Error),
    #[error("Failed to dial seed node: {0}")]
    SeedNodeDialError(#[from] DialError),
    #[error("Failed to parse gossipsub message data: {0}")]
    GossipsubMessageDataParseError(#[from] serde_cbor::Error),
    #[error("Failed to sign hashed message: {0}")]
    SignHashedMessageError(String),
    #[error("Failed to parse signature: {0}")]
    SignatureParseError(String),
    #[error("Failed to verify signature: {0}")]
    SignatureVerificationError(String),
    #[error("Failed to rebroadcast gossipsub message: {0}")]
    GossipsubMessageRebroadcastError(String),
    #[error("Invalid public address: {0}")]
    InvalidPublicAddressError(String),
    #[error("Failed to send event to state manager: {0}")]
    StateManagerError(#[from] flume::SendError<StateManagerEvent>),
    #[error("Failed to sign hashed message, with error: {0}")]
    SignatureError(String),
    #[error("Failed to publish gossipsub message: {0}")]
    GossipsubMessagePublishError(#[from] gossipsub::PublishError),
    #[error("Failed to verify node small ID ownership: {0}")]
    NodeSmallIdOwnershipVerificationError(String),
}

mod utils {
    use fastcrypto::{
        ed25519::{Ed25519PublicKey, Ed25519Signature},
        secp256k1::{Secp256k1PublicKey, Secp256k1Signature},
        secp256r1::{Secp256r1PublicKey, Secp256r1Signature},
        traits::{ToFromBytes as FastCryptoToFromBytes, VerifyingKey},
    };
    use sui_sdk::types::{
        base_types::SuiAddress,
        crypto::{PublicKey, Signature, SignatureScheme, SuiSignature, ToFromBytes},
    };

    use super::*;

    /// The threshold for considering a timestamp as expired
    const EXPIRED_TIMESTAMP_THRESHOLD: u64 = 10 * 60; // 10 minutes

    /// Validates an address response message by checking URL format and timestamp freshness.
    ///
    /// This function performs two key validations:
    /// 1. Ensures the address URL starts with either "http://" or "https://"
    /// 2. Verifies the timestamp is within an acceptable time window relative to the current time
    ///
    /// # Arguments
    ///
    /// * `response` - The `AddressResponse` containing the address and timestamp to validate
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If all validation checks pass
    /// * `Err(P2pAtomaNodeError)` - If any validation fails
    ///
    /// # Errors
    ///
    /// Returns `P2pAtomaNodeError::InvalidPublicAddressError` when:
    /// * The URL doesn't start with "http://" or "https://"
    /// * The timestamp is older than `EXPIRED_TIMESTAMP_THRESHOLD` (10 minutes)
    /// * The timestamp is in the future
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let response = AddressResponse {
    ///     address: "https://example.com".to_string(),
    ///     timestamp: std::time::Instant::now().elapsed().as_secs(),
    ///     signature: vec![],
    ///     node_small_id: 1,
    /// };
    ///
    /// match validate_public_address_message(response) {
    ///     Ok(()) => println!("Address response is valid"),
    ///     Err(e) => println!("Invalid address response: {}", e),
    /// }
    /// ```
    ///
    /// # Security Considerations
    ///
    /// This validation helps prevent:
    /// * Invalid or malicious URLs from being propagated
    /// * Replay attacks by enforcing timestamp freshness
    /// * Future timestamp manipulation attempts
    #[instrument(level = "debug", skip_all)]
    pub(crate) fn validate_public_address_message(
        response: &AddressResponse,
    ) -> Result<(), P2pAtomaNodeError> {
        let now = std::time::Instant::now().elapsed().as_secs();

        // Check if the URL is valid
        // TODO: Better validation using URL crate with `parse` method ?
        if !response.address.starts_with("http://") && !response.address.starts_with("https://") {
            error!(
                target = "atoma-p2p",
                event = "invalid_url_format",
                "Invalid URL format"
            );
            return Err(P2pAtomaNodeError::InvalidPublicAddressError(
                "Invalid URL format".to_string(),
            ));
        }

        // Check if the timestamp is within a reasonable time frame
        if now - response.timestamp > EXPIRED_TIMESTAMP_THRESHOLD || response.timestamp > now {
            error!(
                target = "atoma-p2p",
                event = "invalid_timestamp",
                "Timestamp is invalid, timestamp: {}, now: {}",
                response.timestamp,
                now
            );
            return Err(P2pAtomaNodeError::InvalidPublicAddressError(
                "Timestamp is too old".to_string(),
            ));
        }

        Ok(())
    }

    /// Verifies a cryptographic signature against a message hash using the signature scheme embedded in the signature.
    ///
    /// This function supports multiple signature schemes:
    /// - ED25519
    /// - Secp256k1
    /// - Secp256r1
    ///
    /// # Arguments
    ///
    /// * `signature_bytes` - Raw bytes of the signature. This should include both the signature data and metadata
    ///   that allows extracting the public key and signature scheme.
    /// * `body_hash` - A 32-byte array containing the hash of the message that was signed. This is typically
    ///   produced using blake3 or another cryptographic hash function.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the signature is valid for the given message hash
    /// * `Err(P2pAtomaNodeError)` - If any step of the verification process fails
    ///
    /// # Errors
    ///
    /// This function will return an error in the following situations:
    /// * `SignatureParseError` - If the signature bytes cannot be parsed into a valid signature structure
    ///   or if the public key cannot be extracted from the signature
    /// * `SignatureVerificationError` - If the signature verification fails (indicating the signature is invalid
    ///   for the given message)
    /// * `SignatureParseError` - If the signature uses an unsupported signature scheme
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use your_crate::utils::verify_signature;
    ///
    /// let message = b"Hello, world!";
    /// let message_hash = blake3::hash(message).as_bytes();
    /// let signature_bytes = // ... obtained from somewhere ...
    ///
    /// match verify_signature(&signature_bytes, &message_hash) {
    ///     Ok(()) => println!("Signature is valid!"),
    ///     Err(e) => println!("Signature verification failed: {}", e),
    /// }
    /// ```
    #[instrument(level = "trace", skip_all)]
    pub(crate) fn verify_signature(
        signature_bytes: &[u8],
        body_hash: &[u8; 32],
    ) -> Result<(), P2pAtomaNodeError> {
        let signature = Signature::from_bytes(signature_bytes).map_err(|e| {
            error!("Failed to parse signature");
            P2pAtomaNodeError::SignatureParseError(e.to_string())
        })?;
        let public_key_bytes = signature.public_key_bytes();
        let signature_scheme = signature.scheme();
        let public_key =
            PublicKey::try_from_bytes(signature_scheme, public_key_bytes).map_err(|e| {
                error!("Failed to extract public key from bytes, with error: {e}");
                P2pAtomaNodeError::SignatureParseError(e.to_string())
            })?;

        match signature_scheme {
            SignatureScheme::ED25519 => {
                let public_key = Ed25519PublicKey::from_bytes(public_key.as_ref()).unwrap();
                let signature = Ed25519Signature::from_bytes(signature_bytes).unwrap();
                public_key.verify(body_hash, &signature).map_err(|e| {
                    error!("Failed to verify signature");
                    P2pAtomaNodeError::SignatureVerificationError(e.to_string())
                })?;
            }
            SignatureScheme::Secp256k1 => {
                let public_key = Secp256k1PublicKey::from_bytes(public_key.as_ref()).unwrap();
                let signature = Secp256k1Signature::from_bytes(signature_bytes).unwrap();
                public_key.verify(body_hash, &signature).map_err(|e| {
                    error!("Failed to verify signature");
                    P2pAtomaNodeError::SignatureVerificationError(e.to_string())
                })?;
            }
            SignatureScheme::Secp256r1 => {
                let public_key = Secp256r1PublicKey::from_bytes(public_key.as_ref()).unwrap();
                let signature = Secp256r1Signature::from_bytes(signature_bytes).unwrap();
                public_key.verify(body_hash, &signature).map_err(|e| {
                    error!("Failed to verify signature");
                    P2pAtomaNodeError::SignatureVerificationError(e.to_string())
                })?;
            }
            e => {
                error!("Currently unsupported signature scheme, error: {e}");
                return Err(P2pAtomaNodeError::SignatureParseError(e.to_string()));
            }
        }
        Ok(())
    }

    #[instrument(level = "trace", skip_all)]
    pub(crate) async fn verify_node_small_id_ownership(
        node_small_id: u64,
        signature: &[u8],
        state_manager_sender: Sender<StateManagerEvent>,
    ) -> Result<(), P2pAtomaNodeError> {
        let signature = Signature::from_bytes(signature).map_err(|e| {
            error!("Failed to parse signature");
            P2pAtomaNodeError::SignatureParseError(e.to_string())
        })?;
        let public_key_bytes = signature.public_key_bytes();
        let public_key =
            PublicKey::try_from_bytes(signature.scheme(), public_key_bytes).map_err(|e| {
                error!("Failed to extract public key from bytes, with error: {e}");
                P2pAtomaNodeError::SignatureParseError(e.to_string())
            })?;
        let sui_address = SuiAddress::from(&public_key);
        let (sender, receiver) = oneshot::channel();
        if let Err(e) = state_manager_sender.send((
            AtomaP2pEvent::VerifyNodeSmallIdOwnership {
                node_small_id,
                sui_address: sui_address.to_string(),
            },
            Some(sender),
        )) {
            error!(
                target = "atoma-p2p",
                event = "failed_to_send_event_to_state_manager",
                error = %e,
                "Failed to send event to state manager"
            );
            return Err(P2pAtomaNodeError::StateManagerError(e));
        }
        match receiver.await {
            Ok(result) => {
                if result {
                    Ok(())
                } else {
                    Err(P2pAtomaNodeError::NodeSmallIdOwnershipVerificationError(
                        "Node small ID ownership verification failed".to_string(),
                    ))
                }
            }
            Err(e) => {
                error!("Failed to receive result from state manager, with error: {e}");
                Err(P2pAtomaNodeError::NodeSmallIdOwnershipVerificationError(
                    e.to_string(),
                ))
            }
        }
    }

    /// Publishes an initial message to the P2P network when a node starts up.
    ///
    /// This function handles two different scenarios based on whether the node is a client or a full node:
    ///
    /// 1. For clients:
    ///    - Creates and publishes an address request message
    ///    - Used to discover active nodes in the network
    ///
    /// 2. For full nodes:
    ///    - Creates a signed address response containing the node's public URL
    ///    - Signs the message with the node's private key
    ///    - Includes a timestamp to prevent replay attacks
    ///
    /// # Arguments
    ///
    /// * `swarm` - Mutable reference to the libp2p swarm managing network behaviors
    /// * `keystore` - Reference to the file-based keystore containing node credentials
    /// * `public_url` - The node's publicly accessible URL
    /// * `node_small_id` - Compact numerical identifier for the node
    /// * `is_client` - Boolean flag indicating whether this is a client or full node
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the message is published successfully, or a `P2pAtomaNodeError` if any step fails.
    ///
    /// # Errors
    ///
    /// This function can return several error types:
    /// * `SignatureError` - If signing the message hash fails
    /// * `GossipsubMessageDataParseError` - If CBOR serialization fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use libp2p::Swarm;
    /// use sui_keys::keystore::FileBasedKeystore;
    ///
    /// async fn start_node(
    ///     swarm: &mut Swarm<MyBehaviour>,
    ///     keystore: &FileBasedKeystore,
    /// ) -> Result<(), P2pAtomaNodeError> {
    ///     publish_start_message(
    ///         swarm,
    ///         keystore,
    ///         "https://node1.example.com",
    ///         1,
    ///         false,
    ///     )?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Message Format
    ///
    /// For full nodes, the signed message contains:
    /// ```json
    /// {
    ///     "address": "https://node.example.com",
    ///     "node_small_id": 42,
    ///     "timestamp": 1234567890,
    ///     "signature": [bytes]
    /// }
    /// ```
    ///
    /// For clients, a simple address request message is sent.
    ///
    /// # Security Considerations
    ///
    /// - Messages from full nodes are cryptographically signed
    /// - Includes timestamps to prevent replay attacks
    /// - Uses blake3 for secure message hashing
    /// - Signatures can be verified by other nodes
    /// - Messages are broadcast on a specific gossipsub topic
    ///
    /// # Network Behavior
    ///
    /// - Messages are published to the "atoma-p2p" gossipsub topic
    /// - All connected peers subscribed to this topic will receive the message
    /// - Messages are propagated through the P2P network using gossipsub protocol
    #[instrument(level = "debug", skip_all)]
    pub(crate) fn publish_start_message(
        swarm: &mut Swarm<MyBehaviour>,
        keystore: &FileBasedKeystore,
        public_url: &str,
        node_small_id: u64,
        is_client: bool,
    ) -> Result<(), P2pAtomaNodeError> {
        if is_client {
            let message = GossipMessage::AddressRequest;
            let serialized_message = serde_cbor::to_vec(&message)?;
            let topic = gossipsub::IdentTopic::new(GOSPUBSUB_TOPIC);
            swarm
                .behaviour_mut()
                .gossipsub
                .publish(topic, serialized_message)
                .map_err(|e| {
                    error!("Failed to publish address request message, with error: {e}");
                    P2pAtomaNodeError::GossipsubMessagePublishError(e)
                })?;
        } else {
            let active_address = keystore.addresses()[0];
            let timestamp = std::time::Instant::now().elapsed().as_secs();
            let blake3_hash = blake3::hash(
                &[
                    public_url.as_bytes(),
                    &node_small_id.to_le_bytes(),
                    &timestamp.to_le_bytes(),
                ]
                .concat(),
            );
            let signature = keystore
                .sign_hashed(&active_address, blake3_hash.as_bytes())
                .map_err(|e| {
                    error!(
                        target = "atoma-p2p",
                        event = "sign_hashed_error",
                        error = %e,
                        "Failed to sign hashed message"
                    );
                    P2pAtomaNodeError::SignatureError(e.to_string())
                })?;
            let signature_bytes = signature.signature_bytes().to_vec();
            let message = GossipMessage::AddressResponse(AddressResponse {
                address: public_url.to_string(),
                node_small_id,
                timestamp,
                signature: signature_bytes,
            });
            let serialized_message = serde_cbor::to_vec(&message)?;
            let topic = gossipsub::IdentTopic::new(GOSPUBSUB_TOPIC);
            swarm
                .behaviour_mut()
                .gossipsub
                .publish(topic, serialized_message)
                .map_err(|e| {
                    error!("Failed to publish address response message, with error: {e}");
                    P2pAtomaNodeError::GossipsubMessagePublishError(e)
                })?;
        }
        Ok(())
    }
}
