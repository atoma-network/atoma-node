use crate::{
    config::P2pAtomaNodeConfig,
    types::{AddressResponse, GossipMessage},
};
use futures::StreamExt;
use libp2p::{
    gossipsub::{self, ConfigBuilderError},
    kad::{self, store::MemoryStore},
    mdns, noise,
    swarm::{DialError, NetworkBehaviour, SwarmEvent},
    tcp, yamux, Multiaddr, PeerId, Swarm, SwarmBuilder, TransportError,
};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use sui_keys::keystore::{AccountKeystore, FileBasedKeystore};
use thiserror::Error;
use tokio::sync::watch;
use tracing::{debug, error, instrument};

/// The topic that the P2P network will use to gossip messages
const GOSPUBSUB_TOPIC: &str = "atoma-p2p";

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

    /// Enables automatic peer discovery on local networks using multicast DNS.
    /// Particularly useful for development and local testing environments where nodes
    /// need to find each other without explicit configuration.
    mdns: mdns::tokio::Behaviour,

    /// Provides distributed hash table (DHT) functionality for peer discovery and routing.
    /// Helps maintain network connectivity in larger, distributed deployments by implementing
    /// the Kademlia protocol with a memory-based storage backend.
    kademlia: kad::Behaviour<MemoryStore>,
}

/// A P2P node implementation for the Atoma network that handles peer discovery,
/// message broadcasting, and network communication.
pub struct P2pAtomaNode {
    /// The cryptographic keystore containing the node's keys for signing messages
    /// and managing identities. Wrapped in an Arc for thread-safe shared access.
    keystore: Arc<FileBasedKeystore>,

    /// The libp2p swarm that manages all network behaviors and connections.
    /// Handles peer discovery, message routing, and protocol negotiations using
    /// the configured MyBehaviour protocols (gossipsub, mdns, kademlia).
    swarm: Swarm<MyBehaviour>,

    /// The publicly accessible URL of this node that other peers can use to connect.
    /// This URL is shared with other nodes during peer discovery and address exchange.
    public_url: String,

    /// A compact numerical identifier for the node, used in network messages and logging.
    /// Provides a shorter alternative to the full node ID for quick reference and debugging.
    node_small_id: u64,
}

impl P2pAtomaNode {
    /// Initializes and configures a new P2P Atoma node with networking capabilities.
    ///
    /// This method sets up a complete libp2p node with the following features:
    /// - TCP and QUIC transport layers with noise encryption and yamux multiplexing
    /// - Gossipsub for peer-to-peer message broadcasting
    /// - MDNS for local peer discovery
    /// - Kademlia DHT for distributed peer routing
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
    ///     P2pAtomaNode::start(config, keystore)
    /// }
    /// ```
    ///
    /// # Network Architecture
    ///
    /// The node uses a layered networking approach:
    /// 1. Transport Layer: TCP/QUIC with noise encryption
    /// 2. Protocol Layer:
    ///    - Gossipsub for message broadcasting
    ///    - MDNS for local peer discovery
    ///    - Kademlia for distributed routing
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

                let mdns = mdns::tokio::Behaviour::new(
                    mdns::Config::default(),
                    key.public().to_peer_id(),
                )?;

                let store = kad::store::MemoryStore::new(key.public().to_peer_id());
                let kademlia = kad::Behaviour::new(key.public().to_peer_id(), store);

                Ok(MyBehaviour {
                    gossipsub,
                    mdns,
                    kademlia,
                })
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

        Ok(Self {
            swarm,
            public_url: config.public_url,
            node_small_id: config.node_small_id,
            keystore,
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
                            ..
                        })) => {
                            self.handle_gossipsub_message(&message.data, &message_id)?;
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::Mdns(mdns::Event::Discovered(peers))) => {
                            self.handle_mdns_discovered_peers_event(peers);
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::Mdns(mdns::Event::Expired(peers))) => {
                            self.handle_mdns_expired_peers_event(peers);
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::Kademlia(kad::Event::RoutingUpdated {
                            peer,
                            is_new_peer,
                            old_peer,
                            ..
                        })) => {
                            self.handle_kademlia_routing_updated_event(peer, is_new_peer, old_peer);
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
    /// 1. Deserializes the message data
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
    /// 4. Constructs and serializes an AddressResponse
    /// 5. Publishes the response to the network
    ///
    /// # Arguments
    ///
    /// * `message_data` - Raw bytes of the received gossipsub message
    /// * `message_id` - Unique identifier for the gossipsub message
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if message processing succeeds, or an error variant if any step fails.
    ///
    /// # Errors
    ///
    /// This function can return several error types:
    /// * `GossipsubMessageDataParseError` - If message deserialization fails
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
    ///
    /// # Example Message Format
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
    #[instrument(level = "debug", skip_all)]
    pub fn handle_gossipsub_message(
        &mut self,
        message_data: &[u8],
        message_id: &gossipsub::MessageId,
    ) -> Result<(), P2pAtomaNodeError> {
        debug!(
            target = "atoma-p2p",
            event = "gossipsub_message",
            message_id = %message_id,
            "Received gossipsub message"
        );
        let gossip_message: GossipMessage = serde_json::from_slice(&message_data)?;
        match gossip_message {
            GossipMessage::AddressResponse(response) => {
                debug!(
                    target = "atoma-p2p",
                    event = "gossipsub_message_data",
                    "Received gossipsub message data"
                );
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
                if let Err(e) =
                    utils::verify_signature(signature.as_slice(), message_hash.as_bytes())
                {
                    // if signature is invalid, we don't want to rebroadcast the message to the network
                    error!(
                        target = "atoma-p2p",
                        event = "gossipsub_message_signature_verification_error",
                        error = %e,
                        "Failed to verify signature"
                    );
                    return Err(P2pAtomaNodeError::SignatureVerificationError(e.to_string()));
                }
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
                let serialized_response = serde_json::to_vec(&response)?;
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

    /// Handles the discovery of new peers through multicast DNS (mDNS) on the local network.
    ///
    /// When new peers are discovered on the local network, this method:
    /// - Adds them to the gossipsub peer mesh
    /// - Enables direct message exchange with newly discovered peers
    /// - Expands the network topology to include local participants
    ///
    /// # Arguments
    ///
    /// * `peers` - A vector of `(PeerId, Multiaddr)` tuples where:
    ///   - `PeerId` is the unique identifier of the discovered peer
    ///   - `Multiaddr` is the peer's network address for establishing connections
    ///
    /// # Behavior
    ///
    /// For each discovered peer:
    /// 1. Logs the discovery event with the peer's ID
    /// 2. Adds the peer to gossipsub's explicit peer list
    /// 3. Enables the node to directly exchange messages with the new peer
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // When new peers are discovered on the local network
    /// let discovered_peers = vec![
    ///     (PeerId::random(), "/ip4/192.168.1.1/tcp/1234".parse().unwrap()),
    ///     (PeerId::random(), "/ip4/192.168.1.2/tcp/1234".parse().unwrap()),
    /// ];
    /// node.handle_mdns_discovered_peers_event(discovered_peers);
    /// ```
    ///
    /// # Note
    ///
    /// This method is crucial for local network peer discovery and mesh formation:
    /// - Enables zero-configuration networking in local environments
    /// - Automatically establishes connections with nearby peers
    /// - Particularly useful for development and testing environments
    /// - Complements other peer discovery mechanisms like Kademlia DHT
    ///
    /// The mDNS discovery process is automatic and continuous, allowing the network
    /// to dynamically adapt as peers join the local network.
    #[instrument(level = "debug", skip_all)]
    fn handle_mdns_discovered_peers_event(&mut self, peers: Vec<(PeerId, Multiaddr)>) {
        for (peer_id, _) in peers {
            debug!(
                target = "atoma-p2p",
                event = "mdns_discovered",
                peer_id = %peer_id,
                "MDNS discovered"
            );
            self.swarm
                .behaviour_mut()
                .gossipsub
                .add_explicit_peer(&peer_id);
        }
    }

    /// Handles the expiration of peers discovered through multicast DNS (mDNS).
    ///
    /// When peers become unavailable on the local network, this method:
    /// - Removes them from the gossipsub peer mesh
    /// - Updates the network topology accordingly
    /// - Maintains a clean peer list by removing stale connections
    ///
    /// # Arguments
    ///
    /// * `peers` - A vector of `(PeerId, Multiaddr)` tuples where:
    ///   - `PeerId` is the unique identifier of the expired peer
    ///   - `Multiaddr` is the peer's last known network address (preserved for logging/debugging)
    ///
    /// # Behavior
    ///
    /// For each expired peer:
    /// 1. Logs the expiration event with the peer's ID
    /// 2. Removes the peer from gossipsub's explicit peer list
    /// 3. Allows the network to adjust its topology without the expired peer
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // When peers become unavailable
    /// let expired_peers = vec![
    ///     (PeerId::random(), "/ip4/192.168.1.1/tcp/1234".parse().unwrap()),
    ///     (PeerId::random(), "/ip4/192.168.1.2/tcp/1234".parse().unwrap()),
    /// ];
    /// node.handle_mdns_expired_peers_event(expired_peers);
    /// ```
    ///
    /// # Note
    ///
    /// This method is part of the local peer discovery system and helps maintain
    /// network health by ensuring that only currently available peers are kept
    /// in the gossipsub mesh. This is particularly important in dynamic network
    /// environments where peers may frequently join and leave the network.
    #[instrument(level = "debug", skip_all)]
    fn handle_mdns_expired_peers_event(&mut self, peers: Vec<(PeerId, Multiaddr)>) {
        for (peer_id, _) in peers {
            debug!(
                target = "atoma-p2p",
                event = "mdns_expired",
                peer_id = %peer_id,
                "MDNS expired"
            );
            self.swarm
                .behaviour_mut()
                .gossipsub
                .remove_explicit_peer(&peer_id);
        }
    }

    /// Handles updates to the Kademlia routing table by synchronizing peer connections with gossipsub.
    ///
    /// This method maintains consistency between Kademlia's DHT routing and gossipsub's peer mesh by:
    /// - Adding newly discovered peers to the gossipsub network
    /// - Removing peers that are no longer part of the Kademlia routing table
    ///
    /// # Arguments
    ///
    /// * `peer` - The `PeerId` of the peer whose routing status has changed
    /// * `is_new_peer` - Boolean indicating if this is a newly discovered peer
    /// * `old_peer` - Optional `PeerId` of a peer that was replaced in the routing table
    ///
    /// # Behavior
    ///
    /// - When `is_new_peer` is true:
    ///   - Adds the new peer to gossipsub's explicit peer list
    ///   - This ensures that important DHT peers are also part of the gossip network
    ///
    /// - When `old_peer` contains a value:
    ///   - Removes the old peer from gossipsub's explicit peer list
    ///   - This prevents maintaining unnecessary connections to peers no longer in the DHT
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // When a new peer replaces an old one
    /// node.handle_kademlia_routing_updated_event(
    ///     new_peer_id,
    ///     true,
    ///     Some(old_peer_id)
    /// );
    ///
    /// // When just discovering a new peer
    /// node.handle_kademlia_routing_updated_event(
    ///     new_peer_id,
    ///     true,
    ///     None
    /// );
    /// ```
    ///
    /// # Note
    ///
    /// This synchronization helps maintain an efficient network topology by ensuring
    /// that peers important for DHT routing are also available for message propagation
    /// through gossipsub.
    #[instrument(level = "debug", skip_all)]
    fn handle_kademlia_routing_updated_event(
        &mut self,
        peer: PeerId,
        is_new_peer: bool,
        old_peer: Option<PeerId>,
    ) {
        debug!(
            target = "atoma-p2p",
            event = "kademlia_routing_updated",
            "Kademlia routing updated"
        );
        if is_new_peer {
            self.swarm
                .behaviour_mut()
                .gossipsub
                .add_explicit_peer(&peer);
        }
        if let Some(old_peer) = old_peer {
            self.swarm
                .behaviour_mut()
                .gossipsub
                .remove_explicit_peer(&old_peer);
        }
    }
}

mod utils {
    use fastcrypto::{
        ed25519::{Ed25519PublicKey, Ed25519Signature},
        secp256k1::{Secp256k1PublicKey, Secp256k1Signature},
        secp256r1::{Secp256r1PublicKey, Secp256r1Signature},
        traits::{ToFromBytes as FastCryptoToFromBytes, VerifyingKey},
    };
    use sui_sdk::types::crypto::{
        PublicKey, Signature, SignatureScheme, SuiSignature, ToFromBytes,
    };

    use super::*;

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
    pub fn verify_signature(
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
}

#[derive(Debug, Error)]
pub enum P2pAtomaNodeError {
    #[error("Failed to build gossipsub config: {0}")]
    GossipsubConfigError(#[from] ConfigBuilderError),
    #[error("Failed to build gossipsub: {0}")]
    GossipsubBuildError(String),
    #[error("Failed to build swarm: {0}")]
    SwarmBuildError(String),
    #[error("Failed to build mdns: {0}")]
    MdnsBuildError(String),
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
    GossipsubMessageDataParseError(#[from] serde_json::Error),
    #[error("Failed to sign hashed message: {0}")]
    SignHashedMessageError(String),
    #[error("Failed to parse signature: {0}")]
    SignatureParseError(String),
    #[error("Failed to verify signature: {0}")]
    SignatureVerificationError(String),
    #[error("Failed to rebroadcast gossipsub message: {0}")]
    GossipsubMessageRebroadcastError(String),
}
