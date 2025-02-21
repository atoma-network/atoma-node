use crate::{
    config::AtomaP2pNodeConfig,
    timer::usage_metrics_timer_task,
    types::{AtomaP2pEvent, NodeMessage, SerializeWithSignature, SignedNodeMessage},
};
use config::ConfigError;
use flume::Sender;
use futures::StreamExt;
use libp2p::{
    gossipsub::{self, ConfigBuilderError},
    identify, kad, mdns, noise,
    swarm::{DialError, NetworkBehaviour, SwarmEvent},
    tcp, yamux, PeerId, Swarm, SwarmBuilder, TransportError,
};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use sui_keys::keystore::{AccountKeystore, FileBasedKeystore};
use thiserror::Error;
use tokio::{
    sync::{mpsc::UnboundedReceiver, oneshot, watch},
    task::JoinHandle,
};
use tracing::{debug, error, instrument};

/// The topic that the P2P network will use to gossip messages
const METRICS_GOSPUBSUB_TOPIC: &str = "atoma-p2p-usage-metrics";

type StateManagerEvent = (AtomaP2pEvent, Option<oneshot::Sender<bool>>);

/// Network behavior configuration for the P2P Atoma node, combining multiple libp2p protocols.
///
/// This struct implements the `NetworkBehaviour` trait and coordinates three main networking components
/// for peer discovery, message broadcasting, and distributed routing.
#[derive(NetworkBehaviour)]
struct AtomaP2pBehaviour {
    /// Handles publish-subscribe messaging across the P2P network.
    /// Used for broadcasting node addresses and other network messages using a gossip protocol
    /// that ensures efficient message propagation.
    gossipsub: gossipsub::Behaviour,

    /// Provides a way to identify the node and its capabilities.
    /// Used to discover nodes in the network and to share information about the node,
    /// useful for kademlia and mdns routing.
    identify: identify::Behaviour,

    /// Provides distributed hash table (DHT) functionality for peer discovery and routing.
    /// Helps maintain network connectivity in larger, distributed deployments by implementing
    /// the Kademlia protocol with a memory-based storage backend.
    kademlia: kad::Behaviour<kad::store::MemoryStore>,

    /// Enables automatic peer discovery on local networks using multicast DNS.
    /// Particularly useful for development and local testing environments where nodes
    /// need to find each other without explicit configuration.
    mdns: mdns::tokio::Behaviour,
}

/// A P2P node implementation for the Atoma network that handles peer discovery,
/// message broadcasting, and network communication.
pub struct AtomaP2pNode {
    /// The cryptographic keystore containing the node's keys for signing messages
    /// and managing identities. Wrapped in an Arc for thread-safe shared access.
    keystore: Arc<FileBasedKeystore>,

    /// The libp2p swarm that manages all network behaviors and connections.
    /// Handles peer discovery, message routing, and protocol negotiations using
    /// the configured `AtomaP2pBehaviour` protocols (gossipsub).
    swarm: Swarm<AtomaP2pBehaviour>,

    /// Join handle for the timer task, which is associated with a task that
    /// periodically advertises the node to publish usage metrics to the gossipsub topic.
    timer_join_handle: JoinHandle<Result<(), AtomaP2pNodeError>>,

    /// Sender channel to the state manager
    /// Used to send events to the state manager for storing authenticated
    /// gossipsub messages (containing public URLs of registered nodes)
    state_manager_sender: Sender<StateManagerEvent>,

    /// Receiver channel to receive usage metrics from the timer task
    usage_metrics_rx: UnboundedReceiver<NodeMessage>,

    /// Whether this peer is a client or a node in the Atoma network.
    /// In the case of a client, it will store the public URLs of the participating nodes
    /// in the protocol. In the case, it is a node, it does not store the public URLs of the
    /// participating nodes, neither any public URL of clients. That said, a node must
    /// always share its own public URL with the peers in the network.
    is_client: bool,
}

impl AtomaP2pNode {
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
    /// - `Ok(AtomaP2pNode)` - A fully configured P2P node ready to start
    /// - `Err(AtomaP2pNodeError)` - An error indicating what went wrong during setup
    ///
    /// # Errors
    ///
    /// This function can return several error types:
    /// - `BehaviourBuildError` - Failed to configure networking behaviors
    /// - `GossipsubSubscriptionError` - Failed to subscribe to the gossip topic
    /// - `SwarmListenOnError` - Failed to bind to the specified network address
    /// - `SeedNodeDialError` - Failed to connect to a seed node
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_p2p::{AtomaP2pNode, AtomaP2pNodeConfig};
    /// use std::sync::Arc;
    /// use sui_keys::keystore::FileBasedKeystore;
    ///
    /// async fn setup_node() -> Result<AtomaP2pNode, AtomaP2pNodeError> {
    ///     let config = AtomaP2pNodeConfig {
    ///         heartbeat_interval: std::time::Duration::from_secs(1),
    ///         idle_connection_timeout: std::time::Duration::from_secs(30),
    ///         listen_addr: "/ip4/127.0.0.1/udp/8080/quic-v1/p2p/QmHash...".to_string(),
    ///         seed_nodes: vec![],
    ///         public_url: "node1.example.com".to_string(),
    ///         node_small_id: 1,
    ///     };
    ///     let keystore = Arc::new(FileBasedKeystore::new(&std::path::PathBuf::from("keys"))?);
    ///
    ///     AtomaP2pNode::start(config, keystore, state_manager_sender, false)
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
        config: AtomaP2pNodeConfig,
        keystore: Arc<FileBasedKeystore>,
        state_manager_sender: Sender<StateManagerEvent>,
        is_client: bool,
    ) -> Result<Self, AtomaP2pNodeError> {
        if !is_client
            && (config.public_url.is_none()
                || config.node_small_id.is_none()
                || config.country.is_none())
        {
            error!(
                target = "atoma-p2p",
                event = "invalid_config",
                "Invalid config, either public_url, node_small_id or country is not set, this should never happen"
            );
            return Err(AtomaP2pNodeError::InvalidConfig(
                "Invalid config, either public_url, node_small_id or country is not set, this should never happen".to_string(),
            ));
        }
        let mut swarm = SwarmBuilder::with_new_identity()
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )?
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
                    .validate_messages()
                    .validation_mode(gossipsub::ValidationMode::Strict)
                    .message_id_fn(message_id_fn)
                    .build()
                    .map_err(|e| {
                        error!(
                            target = "atoma-p2p",
                            event = "build_gossipsub_config",
                            error = %e,
                            "Failed to build gossipsub config"
                        );
                        AtomaP2pNodeError::GossipsubConfigError(e)
                    })?;

                let gossipsub = gossipsub::Behaviour::new(
                    gossipsub::MessageAuthenticity::Signed(key.clone()),
                    gossipsub_config,
                )?;

                let store = kad::store::MemoryStore::new(key.public().to_peer_id());
                let kademlia = kad::Behaviour::new(key.public().to_peer_id(), store);

                let mdns = mdns::tokio::Behaviour::new(
                    mdns::Config::default(),
                    key.public().to_peer_id(),
                )?;

                let identify = identify::Behaviour::new(identify::Config::new(
                    "atoma-p2p/0.1.0".to_string(),
                    key.public(),
                ));

                Ok(AtomaP2pBehaviour {
                    gossipsub,
                    identify,
                    kademlia,
                    mdns,
                })
            })
            .map_err(|e| {
                error!(
                    target = "atoma-p2p",
                    event = "build_behaviour",
                    error = %e,
                    "Failed to build behaviour"
                );
                AtomaP2pNodeError::BehaviourBuildError(e.to_string())
            })?
            .with_swarm_config(|c| c.with_idle_connection_timeout(config.idle_connection_timeout))
            .build();

        let topic = gossipsub::IdentTopic::new(METRICS_GOSPUBSUB_TOPIC);
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
                AtomaP2pNodeError::GossipsubSubscriptionError(e)
            })?;
        let listen_addr = config.listen_addr.parse().map_err(|e| {
            error!(
                target = "atoma-p2p",
                event = "address_parse_error",
                listen_addr = config.listen_addr,
                error = %e,
                "Failed to parse listen address"
            );
            AtomaP2pNodeError::ListenAddressParseError(e)
        })?;

        if let Err(e) = swarm.listen_on(listen_addr) {
            error!(
                target = "atoma-p2p",
                event = "listen_on_error",
                listen_addr = config.listen_addr,
                error = %e,
                "Failed to listen on address"
            );
            return Err(AtomaP2pNodeError::SwarmListenOnError(e));
        }

        for peer_id in config.bootstrap_nodes {
            match peer_id.parse::<PeerId>() {
                Ok(peer_id) => {
                    swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, "/dnsaddr/bootstrap.libp2p.io".parse()?);
                    debug!(
                        target = "atoma-p2p",
                        event = "dialed_bootstrap_node",
                        peer_id = %peer_id,
                        "Dialed bootstrap node"
                    );
                }
                Err(e) => {
                    error!(
                        target = "atoma-p2p",
                        event = "bootstrap_node_parse_error",
                        peer_id = %peer_id,
                        error = %e,
                        "Failed to parse bootstrap node address"
                    );
                }
            }
        }

        let (usage_metrics_tx, usage_metrics_rx) = tokio::sync::mpsc::unbounded_channel();

        let timer_join_handle = usage_metrics_timer_task(
            config.country,
            config.metrics_endpoints.clone(),
            is_client,
            config.public_url,
            config.node_small_id,
            usage_metrics_tx,
        );

        Ok(Self {
            keystore,
            swarm,
            timer_join_handle,
            state_manager_sender,
            usage_metrics_rx,
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
    /// - Message: Processes incoming messages on the gossip network using `handle_gossipsub_message`
    /// - Subscribed: Logs when peers subscribe to topics
    /// - Unsubscribed: Logs when peers unsubscribe from topics (TODO: consider removing stored public URLs)
    ///
    /// ## Usage Metrics
    /// Handles periodic usage metrics updates from the timer task
    ///
    /// ## Shutdown Signals
    /// Monitors the shutdown channel for termination requests
    ///
    /// # Arguments
    ///
    /// * `self` - Takes ownership of the P2P node instance
    /// * `shutdown_signal` - A watch channel receiver for shutdown coordination
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` when the node shuts down gracefully, or a `AtomaP2pNodeError` if an error occurs.
    ///
    /// # Errors
    ///
    /// Can return errors from:
    /// - Gossipsub message handling
    /// - Usage metrics processing
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use tokio::sync::watch;
    /// use atoma_p2p::{AtomaP2pNode, AtomaP2pNodeConfig};
    ///
    /// async fn run_node(node: AtomaP2pNode) {
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
    /// In both cases, the node will:
    /// 1. Abort the timer task
    /// 2. Break from the event loop
    /// 3. Return `Ok(())`
    ///
    /// # Concurrency
    ///
    /// Uses `tokio::select!` to handle multiple asynchronous events concurrently:
    /// 1. Swarm events (network activity)
    /// 2. Usage metrics updates
    /// 3. Shutdown signals
    ///
    /// # Logging
    ///
    /// The method logs:
    /// - Gossipsub message handling errors
    /// - Peer subscription/unsubscription events
    /// - Usage metrics processing errors
    /// - Shutdown events
    #[instrument(level = "info", skip_all)]
    pub async fn run(
        mut self,
        mut shutdown_signal: watch::Receiver<bool>,
    ) -> Result<(), AtomaP2pNodeError> {
        loop {
            tokio::select! {
                event = self.swarm.select_next_some() => {
                    match event {
                        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                            message_id,
                            message,
                            propagation_source,
                        })) => {
                            match self.handle_gossipsub_message(&message.data, &message_id, &propagation_source).await {
                                Ok(()) => {}
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
                        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Gossipsub(gossipsub::Event::Subscribed {
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
                        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Gossipsub(gossipsub::Event::Unsubscribed {
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
                        }
                        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Mdns(mdns::Event::Discovered(discovered_peers))) => {
                            debug!(
                                target = "atoma-p2p",
                                event = "mdns_discovered",
                                num_discovered_peers = %discovered_peers.len(),
                                "Mdns discovered"
                            );
                            for (peer_id, multiaddr) in discovered_peers {
                                debug!(
                                    target = "atoma-p2p",
                                    event = "mdns_discovered_peer",
                                    peer_id = %peer_id,
                                    multiaddr = %multiaddr,
                                    "Mdns discovered peer"
                                );
                                self.swarm.behaviour_mut().kademlia.add_address(&peer_id, multiaddr);
                            }
                        }
                        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Mdns(mdns::Event::Expired(expired_peers))) => {
                            debug!(
                                target = "atoma-p2p",
                                event = "mdns_expired",
                                num_expired_peers = %expired_peers.len(),
                                "Mdns expired"
                            );
                            for (peer_id, multiaddr) in expired_peers {
                                debug!(
                                    target = "atoma-p2p",
                                    event = "mdns_expired_peer",
                                    peer_id = %peer_id,
                                    multiaddr = %multiaddr,
                                    "Mdns expired peer"
                                );
                                self.swarm.behaviour_mut().kademlia.remove_address(&peer_id, &multiaddr);
                            }
                        }
                        SwarmEvent::ConnectionEstablished {
                            peer_id,
                            num_established,
                            established_in,
                            connection_id,
                            ..
                        } => {
                            debug!(
                                target = "atoma-p2p",
                                event = "peer_connection_established",
                                peer_id = %peer_id,
                                num_established = %num_established,
                                established_in = ?established_in,
                                connection_id = %connection_id,
                                "Peer connection established"
                            );
                        }
                        SwarmEvent::ConnectionClosed {
                            peer_id,
                            connection_id,
                            ..
                        } => {
                            debug!(
                                target = "atoma-p2p",
                                event = "peer_connection_closed",
                                peer_id = %peer_id,
                                connection_id = %connection_id,
                                "Peer connection closed"
                            );
                        }
                        SwarmEvent::NewListenAddr {
                            listener_id,
                            address,
                            ..
                        } => {
                            debug!(
                                target = "atoma-p2p",
                                event = "new_listen_addr",
                                listener_id = %listener_id,
                                address = %address,
                                "New listen address"
                            );
                        }
                        SwarmEvent::ExpiredListenAddr {
                            listener_id,
                            address,
                        } => {
                            debug!(
                                target = "atoma-p2p",
                                event = "expired_listen_addr",
                                listener_id = %listener_id,
                                address = %address,
                                "Expired listen address"
                            );
                        }
                        SwarmEvent::Dialing {
                            peer_id,
                            connection_id,
                            ..
                        } => {
                            debug!(
                                target = "atoma-p2p",
                                event = "dialing",
                                peer_id = ?peer_id,
                                connection_id = %connection_id,
                                "Dialing peer"
                            );
                        }
                        _ => {}
                    }
                }
                Some(usage_metrics) = self.usage_metrics_rx.recv() => {
                    if let Err(e) = self.handle_new_usage_metrics_event(usage_metrics) {
                        error!(
                            target = "atoma-p2p",
                            event = "usage_metrics_error",
                            error = %e,
                            "Failed to handle new usage metrics"
                        );
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
                                // Abort the timer task
                                self.timer_join_handle.abort();
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
    /// This method processes UsageMetrics messages by:
    /// 1. Validating the message's signature and timestamp
    /// 2. Verifying the node's ownership of its small ID
    /// 3. Reporting the validation result to the gossipsub protocol
    /// 4. Storing the node's public URL in the state manager (if this peer is a client)
    ///
    /// # Message Flow
    ///
    /// 1. Receives a message from the gossipsub network
    /// 2. Skips processing if the message is from self
    /// 3. Deserializes the message into a GossipMessage
    /// 4. For UsageMetrics messages:
    ///    - Validates the message using `validate_usage_metrics_message`
    ///    - Reports validation result to gossipsub protocol
    ///    - If this peer is a client, stores the node's public URL in state manager
    ///
    /// # Arguments
    ///
    /// * `message_data` - Raw bytes of the received gossipsub message (CBOR encoded)
    /// * `message_id` - Unique identifier for the gossipsub message
    /// * `propagation_source` - The peer that forwarded this message
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if message processing succeeds, or an error if:
    /// - Message deserialization fails
    /// - Message validation fails
    /// - Reporting validation result fails
    /// - Storing public URL fails (for clients)
    ///
    /// # Errors
    ///
    /// This function can return the following errors:
    /// * `UsageMetricsSerializeError` - If CBOR deserialization fails
    /// * `StateManagerError` - If storing public URL fails (for clients)
    ///
    /// # Security Considerations
    ///
    /// - Messages from self are ignored to prevent message loops
    /// - Messages are validated before processing
    /// - Only clients store public URLs to prevent unnecessary data storage
    /// - Uses CBOR for efficient binary serialization
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let message_data = // ... received from gossipsub ...;
    /// let message_id = // ... message identifier ...;
    /// let propagation_source = // ... peer ID ...;
    ///
    /// match node.handle_gossipsub_message(&message_data, &message_id, &propagation_source).await {
    ///     Ok(()) => println!("Message processed successfully"),
    ///     Err(e) => println!("Failed to process message: {}", e),
    /// }
    /// ```
    ///
    /// # Message Validation
    ///
    /// Messages are validated by:
    /// 1. Checking the URL format and timestamp freshness
    /// 2. Verifying the cryptographic signature
    /// 3. Confirming the node's ownership of its small ID
    ///
    /// Invalid messages are rejected and not propagated further in the network.
    #[instrument(level = "debug", skip_all)]
    pub async fn handle_gossipsub_message(
        &mut self,
        message_data: &[u8],
        message_id: &gossipsub::MessageId,
        propagation_source: &PeerId,
    ) -> Result<(), AtomaP2pNodeError> {
        debug!(
            target = "atoma-p2p",
            event = "gossipsub_message",
            message_id = %message_id,
            propagation_source = %propagation_source,
            "Received gossipsub message"
        );
        if propagation_source == self.swarm.local_peer_id() {
            debug!(
                target = "atoma-p2p",
                event = "gossipsub_message_from_self",
                "Gossipsub message from self"
            );
            // Do not re-publish the node's own message, just return `Ok(())
            return Ok(());
        }
        // Directly deserialize SignedNodeMessage using new method
        let signed_node_message = SignedNodeMessage::deserialize_with_signature(message_data)?;
        let signature_len = signed_node_message.signature.len();
        debug!(
            target = "atoma-p2p",
            event = "gossipsub_message_data",
            "Received gossipsub message data"
        );
        let node_message = &signed_node_message.node_message;
        let node_message_hash = blake3::hash(&message_data[signature_len..]);
        let message_acceptance = match utils::validate_signed_node_message(
            node_message,
            node_message_hash.as_bytes(),
            &signed_node_message.signature,
            &self.state_manager_sender,
        )
        .await
        {
            Ok(()) => gossipsub::MessageAcceptance::Accept,
            Err(e) => {
                error!(
                    target = "atoma-p2p",
                    event = "gossipsub_message_validation_error",
                    error = %e,
                    "Failed to validate gossipsub message"
                );
                // NOTE: We should reject the message if it fails to validate
                // as it means the node is not being following the current protocol
                if let AtomaP2pNodeError::UrlParseError(_) = e {
                    // We remove the peer from the gossipsub topic, because it is not a valid URL and therefore cannot be reached
                    // by clients for processing OpenAI api compatible AI requests, so these peers are not useful for the network
                    self.swarm
                        .behaviour_mut()
                        .gossipsub
                        .remove_explicit_peer(propagation_source);
                }
                gossipsub::MessageAcceptance::Reject
            }
        };
        // Report the message validation result to the gossipsub protocol
        let is_in_mempool = self
            .swarm
            .behaviour_mut()
            .gossipsub
            .report_message_validation_result(message_id, propagation_source, message_acceptance);
        if is_in_mempool {
            debug!(
                target = "atoma-p2p",
                event = "gossipsub_message_in_mempool",
                message_id = %message_id,
                propagation_source = %propagation_source,
                "Gossipsub message already in the mempool, no need to take further actions"
            );
            return Ok(());
        }
        // If the current peer is a client, we need to store the public URL in the state manager
        if self.is_client {
            let node_message = signed_node_message.node_message;
            let event = AtomaP2pEvent::NodeMetricsRegistrationEvent {
                public_url: node_message.node_metadata.node_public_url,
                node_small_id: node_message.node_metadata.node_small_id,
                timestamp: node_message.node_metadata.timestamp,
                country: node_message.node_metadata.country,
                node_metrics: node_message.node_metrics,
            };
            self.state_manager_sender.send((event, None)).map_err(|e| {
                error!(
                    target = "atoma-p2p",
                    event = "gossipsub_message_state_manager_error",
                    error = %e,
                    "Failed to send event to state manager"
                );
                AtomaP2pNodeError::StateManagerError(e)
            })?;
        }

        Ok(())
    }

    /// Handles the publishing of new usage metrics to the P2P network.
    ///
    /// This method performs the following operations:
    /// 1. Creates a hash of the usage metrics data (node_public_url, node_small_id, timestamp)
    /// 2. Signs the hash using the node's keystore
    /// 3. Constructs a UsageMetrics struct with the signed data
    /// 4. Serializes the metrics to CBOR format
    /// 5. Publishes the serialized message to the metrics gossipsub topic
    ///
    /// # Arguments
    ///
    /// * `usage_metrics` - The NodeUsageMetrics containing:
    ///   - node_public_url: Public URL of the node
    ///   - node_small_id: Compact numerical identifier for the node
    ///   - timestamp: Unix timestamp of when metrics were collected
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the metrics were successfully published, or an error if:
    /// - Signing the metrics hash fails
    /// - Serialization of metrics fails
    /// - Publishing to the gossipsub topic fails
    ///
    /// # Errors
    ///
    /// This function can return the following errors:
    /// * `SignatureError` - If signing the metrics hash fails
    /// * `UsageMetricsSerializeError` - If CBOR serialization of the metrics fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let metrics = NodeUsageMetrics {
    ///     node_public_url: "node1.example.com".to_string(),
    ///     node_small_id: 1,
    ///     timestamp: 1234567890,
    /// };
    ///
    /// match node.handle_new_usage_metrics_event(metrics) {
    ///     Ok(()) => println!("Metrics published successfully"),
    ///     Err(e) => println!("Failed to publish metrics: {}", e),
    /// }
    /// ```
    ///
    /// # Security Considerations
    ///
    /// - Metrics are signed to ensure authenticity and prevent tampering
    /// - Uses CBOR for efficient binary serialization
    /// - Published through gossipsub's secure message propagation
    /// - The hash includes all critical metrics data to ensure integrity
    #[instrument(
        level = "info",
        fields(
            node_small_id = %node_message.node_metadata.node_small_id,
            node_public_url = %node_message.node_metadata.node_public_url,
        ),
        skip_all
    )]
    fn handle_new_usage_metrics_event(
        &mut self,
        node_message: NodeMessage,
    ) -> Result<(), AtomaP2pNodeError> {
        let mut bytes = Vec::new();
        ciborium::into_writer(&node_message, &mut bytes).unwrap();
        let hash = blake3::hash(&bytes);

        let signature = self
            .keystore
            .sign_hashed(&self.keystore.addresses()[0], hash.as_bytes())
            .map_err(|e| {
                error!(
                    target = "atoma-p2p",
                    event = "sign_node_message_hash_error",
                    error = %e,
                    "Failed to sign node message hash"
                );
                AtomaP2pNodeError::SignatureError(e.to_string())
            })?;
        let signed_node_message = SignedNodeMessage {
            node_message,
            signature: signature.as_ref().to_vec(),
        };
        let serialized_signed_node_message = signed_node_message.serialize_with_signature()?;
        let topic = gossipsub::IdentTopic::new(METRICS_GOSPUBSUB_TOPIC);
        self.swarm
            .behaviour_mut()
            .gossipsub
            .publish(topic, serialized_signed_node_message)?;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum AtomaP2pNodeError {
    #[error("Failed to build gossipsub config: {0}")]
    GossipsubConfigError(#[from] ConfigBuilderError),
    #[error("Failed to build behaviour: {0}")]
    BehaviourBuildError(String),
    #[error("Failed to subscribe to topic: {0}")]
    GossipsubSubscriptionError(#[from] gossipsub::SubscriptionError),
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
    GossipsubMessagePublishError(#[from] gossipsub::PublishError),
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

mod utils {
    use fastcrypto::{
        ed25519::{Ed25519PublicKey, Ed25519Signature},
        secp256k1::{Secp256k1PublicKey, Secp256k1Signature},
        secp256r1::{Secp256r1PublicKey, Secp256r1Signature},
        traits::{ToFromBytes as FastCryptoToFromBytes, VerifyingKey},
    };
    use flume::Sender;
    use sui_sdk::types::{
        base_types::SuiAddress,
        crypto::{PublicKey, Signature, SignatureScheme, SuiSignature, ToFromBytes},
    };
    use tokio::sync::oneshot;
    use tracing::{error, instrument};
    use url::Url;

    use crate::{types::NodeMessage, AtomaP2pEvent};

    use super::{AtomaP2pNodeError, StateManagerEvent};

    /// The threshold for considering a timestamp as expired
    const EXPIRED_TIMESTAMP_THRESHOLD: u64 = 10 * 60; // 10 minutes

    /// Validates a UsageMetrics message by checking the URL format and timestamp freshness.
    ///
    /// This function performs two key validations:
    /// 1. Ensures the node_public_url is a valid URL format
    /// 2. Verifies the timestamp is within an acceptable time window relative to the current time
    ///
    /// # Arguments
    ///
    /// * `response` - The `UsageMetrics` containing the node_public_url and timestamp to validate
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If all validation checks pass
    /// * `Err(AtomaP2pNodeError)` - If any validation fails
    ///
    /// # Errors
    ///
    /// Returns `AtomaP2pNodeError::InvalidPublicAddressError` when:
    /// * The URL is invalid or malformed
    /// * The timestamp is older than `EXPIRED_TIMESTAMP_THRESHOLD` (10 minutes)
    /// * The timestamp is in the future
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let metrics = UsageMetrics {
    ///     node_public_url: "https://example.com".to_string(),
    ///     timestamp: std::time::Instant::now().elapsed().as_secs(),
    ///     signature: vec![],
    ///     node_small_id: 1,
    /// };
    ///
    /// match validate_public_address_message(&metrics) {
    ///     Ok(()) => println!("UsageMetrics is valid"),
    ///     Err(e) => println!("Invalid UsageMetrics: {}", e),
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
    pub fn validate_node_message_country_url_timestamp(
        node_message: &NodeMessage,
    ) -> Result<(), AtomaP2pNodeError> {
        let now = std::time::Instant::now().elapsed().as_secs();

        let country = node_message.node_metadata.country.as_str();
        validate_country_code(country)?;

        // Check if the URL is valid
        let _usage_metrics_url =
            Url::parse(&node_message.node_metadata.node_public_url).map_err(|e| {
                error!(
                    target = "atoma-p2p",
                    event = "invalid_url_format",
                    error = %e,
                    "Invalid URL format, received address: {}",
                    node_message.node_metadata.node_public_url
                );
                AtomaP2pNodeError::UrlParseError(e)
            })?;

        // Check if the timestamp is within a reasonable time frame
        if now < node_message.node_metadata.timestamp
            || now > node_message.node_metadata.timestamp + EXPIRED_TIMESTAMP_THRESHOLD
        {
            error!(
                target = "atoma-p2p",
                event = "invalid_timestamp",
                "Timestamp is invalid, timestamp: {}, now: {}",
                node_message.node_metadata.timestamp,
                now
            );
            return Err(AtomaP2pNodeError::InvalidPublicAddressError(
                "Timestamp is too far in the past".to_string(),
            ));
        }

        Ok(())
    }

    /// Custom validation function for ISO 3166-1 alpha-2 country codes
    fn validate_country_code(code: &str) -> Result<(), AtomaP2pNodeError> {
        isocountry::CountryCode::for_alpha2(code).map_err(|_| {
            AtomaP2pNodeError::InvalidCountryCodeError("Country code is invalid.".to_string())
        })?;
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
    /// * `Err(AtomaP2pNodeError)` - If any step of the verification process fails
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
    ) -> Result<(), AtomaP2pNodeError> {
        let signature = Signature::from_bytes(signature_bytes).map_err(|e| {
            error!("Failed to parse signature");
            AtomaP2pNodeError::SignatureParseError(e.to_string())
        })?;
        let public_key_bytes = signature.public_key_bytes();
        let signature_scheme = signature.scheme();
        let public_key =
            PublicKey::try_from_bytes(signature_scheme, public_key_bytes).map_err(|e| {
                error!("Failed to extract public key from bytes, with error: {e}");
                AtomaP2pNodeError::SignatureParseError(e.to_string())
            })?;
        let signature_bytes = signature.signature_bytes();

        match signature_scheme {
            SignatureScheme::ED25519 => {
                let public_key = Ed25519PublicKey::from_bytes(public_key.as_ref()).unwrap();
                let signature = Ed25519Signature::from_bytes(signature_bytes).unwrap();
                public_key.verify(body_hash, &signature).map_err(|e| {
                    error!("Failed to verify signature");
                    AtomaP2pNodeError::SignatureVerificationError(e.to_string())
                })?;
            }
            SignatureScheme::Secp256k1 => {
                let public_key = Secp256k1PublicKey::from_bytes(public_key.as_ref()).unwrap();
                let signature = Secp256k1Signature::from_bytes(signature_bytes).unwrap();
                public_key.verify(body_hash, &signature).map_err(|e| {
                    error!("Failed to verify signature");
                    AtomaP2pNodeError::SignatureVerificationError(e.to_string())
                })?;
            }
            SignatureScheme::Secp256r1 => {
                let public_key = Secp256r1PublicKey::from_bytes(public_key.as_ref()).unwrap();
                let signature = Secp256r1Signature::from_bytes(signature_bytes).unwrap();
                public_key.verify(body_hash, &signature).map_err(|e| {
                    error!("Failed to verify signature");
                    AtomaP2pNodeError::SignatureVerificationError(e.to_string())
                })?;
            }
            e => {
                error!("Currently unsupported signature scheme, error: {e}");
                return Err(AtomaP2pNodeError::SignatureParseError(e.to_string()));
            }
        }
        Ok(())
    }

    /// Verifies that a node owns its claimed small ID by checking the signature and querying the state manager.
    ///
    /// This function performs the following steps:
    /// 1. Parses the signature to extract the public key
    /// 2. Derives the Sui address from the public key
    /// 3. Sends a verification request to the state manager
    /// 4. Waits for the state manager's response
    ///
    /// # Arguments
    ///
    /// * `node_small_id` - The small ID claimed by the node
    /// * `signature` - The signature to verify, containing the public key
    /// * `state_manager_sender` - Channel to send verification requests to the state manager
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the node owns the small ID
    /// * `Err(AtomaP2pNodeError)` - If verification fails at any step
    ///
    /// # Errors
    ///
    /// This function can return the following errors:
    /// * `SignatureParseError` - If the signature cannot be parsed
    /// * `StateManagerError` - If the verification request cannot be sent to the state manager
    /// * `NodeSmallIdOwnershipVerificationError` - If the state manager reports the node does not own the ID
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let signature = // ... obtained from node ...;
    /// let result = verify_node_small_id_ownership(
    ///     42,
    ///     &signature,
    ///     state_manager_sender
    /// ).await;
    ///
    /// match result {
    ///     Ok(()) => println!("Verification succeeded"),
    ///     Err(e) => println!("Verification failed: {}", e),
    /// }
    /// ```
    ///
    /// # Security Considerations
    ///
    /// - Uses cryptographic signatures to prevent impersonation
    /// - Relies on the state manager's authoritative record of node ownership
    /// - Protects against node ID spoofing attacks
    #[instrument(level = "debug", skip_all)]
    pub async fn verify_node_small_id_ownership(
        node_small_id: u64,
        signature: &[u8],
        state_manager_sender: &Sender<StateManagerEvent>,
    ) -> Result<(), AtomaP2pNodeError> {
        let signature = Signature::from_bytes(signature).map_err(|e| {
            error!("Failed to parse signature");
            AtomaP2pNodeError::SignatureParseError(e.to_string())
        })?;
        let public_key_bytes = signature.public_key_bytes();
        let public_key =
            PublicKey::try_from_bytes(signature.scheme(), public_key_bytes).map_err(|e| {
                error!("Failed to extract public key from bytes, with error: {e}");
                AtomaP2pNodeError::SignatureParseError(e.to_string())
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
            return Err(AtomaP2pNodeError::StateManagerError(e));
        }
        match receiver.await {
            Ok(result) => {
                if result {
                    Ok(())
                } else {
                    Err(AtomaP2pNodeError::NodeSmallIdOwnershipVerificationError(
                        "Node small ID ownership verification failed".to_string(),
                    ))
                }
            }
            Err(e) => {
                error!("Failed to receive result from state manager, with error: {e}");
                Err(AtomaP2pNodeError::NodeSmallIdOwnershipVerificationError(
                    e.to_string(),
                ))
            }
        }
    }

    /// Validates the messages received from the P2P network
    ///
    /// This function validates the sharing node public addresses messages received from the P2P network by checking the signature and the node small ID ownership.
    ///
    /// # Arguments
    ///
    /// * `response` - The message received from the P2P network
    /// * `state_manager_sender` - The sender of the state manager
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the message is valid, or a `AtomaP2pNodeError` if any step fails.
    #[instrument(level = "debug", skip_all)]
    pub async fn validate_signed_node_message(
        node_message: &NodeMessage,
        node_message_hash: &[u8; 32],
        signature: &[u8],
        state_manager_sender: &Sender<StateManagerEvent>,
    ) -> Result<(), AtomaP2pNodeError> {
        // Validate the message's node public URL and timestamp
        validate_node_message_country_url_timestamp(node_message)?;
        // Verify the signature of the message
        verify_signature(signature, node_message_hash)?;
        // Verify the node small ID ownership
        verify_node_small_id_ownership(
            node_message.node_metadata.node_small_id,
            signature,
            state_manager_sender,
        )
        .await?;
        Ok(())
    }
}

#[cfg(test)]
pub mod tests {
    use crate::{metrics::NodeMetrics, types::NodeP2pMetadata};

    use super::*;
    use flume::unbounded;
    use sui_keys::keystore::InMemKeystore;

    /// Creates a test keystore
    ///
    /// # Returns
    ///
    /// Returns an `InMemKeystore` struct
    #[must_use]
    pub fn create_test_keystore() -> InMemKeystore {
        // Create a new in-memory keystore
        InMemKeystore::new_insecure_for_tests(1)
    }

    /// Creates a test usage metrics message
    ///
    /// # Arguments
    ///
    /// * `keystore` - The keystore to use for signing the message
    /// * `timestamp_offset` - The offset to add to the current timestamp
    /// * `node_small_id` - The small ID of the node
    ///
    /// # Returns
    ///
    /// Returns a `SignedNodeMessage` struct
    ///
    /// # Panics
    ///
    /// Panics if the usage metrics cannot be serialized
    #[must_use]
    pub fn create_test_signed_node_message(
        keystore: &InMemKeystore,
        timestamp_offset: i64,
        node_small_id: u64,
    ) -> SignedNodeMessage {
        let now = std::time::Instant::now().elapsed().as_secs();
        #[allow(clippy::cast_sign_loss)]
        #[allow(clippy::cast_possible_wrap)]
        let timestamp = (now as i64 + timestamp_offset) as u64;

        let node_public_url = "https://test.example.com".to_string();
        let country = "US".to_string();

        let node_message = NodeMessage {
            node_metadata: NodeP2pMetadata {
                node_public_url,
                node_small_id,
                country,
                timestamp,
            },
            node_metrics: NodeMetrics::default(),
        };

        let mut node_message_bytes = Vec::new();
        ciborium::into_writer(&node_message, &mut node_message_bytes)
            .map_err(|e| {
                error!(
                    target = "atoma-p2p",
                    event = "serialize_usage_metrics_error",
                    error = %e,
                    "Failed to serialize usage metrics"
                );
                AtomaP2pNodeError::UsageMetricsSerializeError(e)
            })
            .expect("Failed to serialize usage metrics");
        let message_hash = blake3::hash(&node_message_bytes);

        let active_address = keystore.addresses()[0];
        let signature = keystore
            .sign_hashed(&active_address, message_hash.as_bytes())
            .expect("Failed to sign message");

        SignedNodeMessage {
            node_message,
            signature: signature.as_ref().to_vec(),
        }
    }

    #[tokio::test]
    async fn test_validate_usage_metrics_message_success() {
        let keystore = create_test_keystore();
        let (tx, rx) = unbounded();

        // Create valid usage metrics
        let signed_node_message = create_test_signed_node_message(&keystore, 0, 1);

        // Mock successful node small ID ownership verification
        tokio::spawn(async move {
            let (event, optional_response_sender): (AtomaP2pEvent, Option<oneshot::Sender<bool>>) =
                rx.recv_async().await.unwrap();
            if let AtomaP2pEvent::VerifyNodeSmallIdOwnership {
                node_small_id,
                sui_address: _,
            } = event
            {
                assert_eq!(node_small_id, 1);
                let response_sender = optional_response_sender.unwrap();
                response_sender.send(true).unwrap();
            }
        });

        let mut node_meessage_bytes = Vec::new();
        ciborium::into_writer(&signed_node_message.node_message, &mut node_meessage_bytes).unwrap();
        let node_message_hash = blake3::hash(&node_meessage_bytes);
        // Validation should succeed
        let result = utils::validate_signed_node_message(
            &signed_node_message.node_message,
            node_message_hash.as_bytes(),
            &signed_node_message.signature,
            &tx,
        )
        .await;
        result.unwrap();
        // assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_usage_metrics_message_invalid_url() {
        let keystore = create_test_keystore();
        let (tx, _rx) = unbounded();

        // Create base metrics then modify before serialization
        let node_message = NodeMessage {
            node_metadata: NodeP2pMetadata {
                node_public_url: "invalid_url".to_string(), // Direct invalid URL
                node_small_id: 1,
                country: "US".to_string(),
                timestamp: std::time::Instant::now().elapsed().as_secs(),
            },
            node_metrics: NodeMetrics::default(),
        };

        let mut bytes = Vec::new();
        ciborium::into_writer(&node_message, &mut bytes).unwrap();
        let hash = blake3::hash(&bytes);

        // Serialize modified message
        let signature = keystore
            .sign_hashed(&keystore.addresses()[0], hash.as_bytes())
            .unwrap();

        let bad_metrics = SignedNodeMessage {
            node_message,
            signature: signature.as_ref().to_vec(),
        };

        let result = utils::validate_signed_node_message(
            &bad_metrics.node_message,
            hash.as_bytes(),
            &bad_metrics.signature,
            &tx,
        )
        .await;
        assert!(matches!(result, Err(AtomaP2pNodeError::UrlParseError(_))));
    }

    #[tokio::test]
    async fn test_validate_usage_metrics_message_expired_timestamp() {
        let keystore = create_test_keystore();
        let (tx, _rx) = unbounded();

        // Create metrics with expired timestamp (11 minutes ago)
        let signed_node_message = create_test_signed_node_message(&keystore, -(11 * 60), 1);
        let node_message = &signed_node_message.node_message;
        let mut node_message_bytes = Vec::new();
        ciborium::into_writer(node_message, &mut node_message_bytes).unwrap();
        let node_message_hash = blake3::hash(&node_message_bytes);

        let result = utils::validate_signed_node_message(
            node_message,
            node_message_hash.as_bytes(),
            &signed_node_message.signature,
            &tx,
        )
        .await;
        assert!(matches!(
            result,
            Err(AtomaP2pNodeError::InvalidPublicAddressError(_))
        ));
    }

    #[tokio::test]
    async fn test_validate_usage_metrics_message_future_timestamp() {
        let keystore = create_test_keystore();
        let (tx, _rx) = unbounded();

        // Create metrics with future timestamp
        let signed_node_message = create_test_signed_node_message(&keystore, 60, 1);
        let node_message = &signed_node_message.node_message;
        let mut node_message_bytes = Vec::new();
        ciborium::into_writer(node_message, &mut node_message_bytes).unwrap();
        let node_message_hash = blake3::hash(&node_message_bytes);

        let result = utils::validate_signed_node_message(
            node_message,
            node_message_hash.as_bytes(),
            &signed_node_message.signature,
            &tx,
        )
        .await;
        assert!(matches!(
            result,
            Err(AtomaP2pNodeError::InvalidPublicAddressError(_))
        ));
    }

    #[tokio::test]
    async fn test_validate_usage_metrics_message_invalid_signature() {
        let keystore = create_test_keystore();
        let (tx, _rx) = unbounded();

        let signed_node_message = create_test_signed_node_message(&keystore, 0, 1);
        let node_message = &signed_node_message.node_message;

        // Corrupt the signature part after scheme byte
        let scheme_length = 1; // Ed25519 scheme byte
        let sig_start = scheme_length;
        let mut signature = signed_node_message.signature.clone();
        for byte in &mut signature[sig_start..sig_start + 64] {
            *byte = 0xff;
        }

        let mut node_message_bytes = Vec::new();
        ciborium::into_writer(&node_message, &mut node_message_bytes).unwrap();
        let node_message_hash = blake3::hash(&node_message_bytes);

        let result = utils::validate_signed_node_message(
            &signed_node_message.node_message,
            node_message_hash.as_bytes(),
            &signature,
            &tx,
        )
        .await;
        assert!(matches!(
            result,
            Err(AtomaP2pNodeError::SignatureVerificationError(_))
        ));
    }

    #[tokio::test]
    async fn test_validate_usage_metrics_message_invalid_node_ownership() {
        let keystore = create_test_keystore();
        let (tx, rx) = unbounded();

        // Create valid metrics
        let signed_node_message = create_test_signed_node_message(&keystore, 0, 1);
        let node_message = &signed_node_message.node_message;

        // Mock failed node small ID ownership verification
        tokio::spawn(async move {
            let (event, _response_sender) = rx.recv_async().await.unwrap();
            if let AtomaP2pEvent::VerifyNodeSmallIdOwnership {
                node_small_id,
                sui_address: _,
            } = event
            {
                assert_eq!(node_small_id, 1);
            }
        });

        let mut node_message_bytes = Vec::new();
        ciborium::into_writer(node_message, &mut node_message_bytes).unwrap();
        let node_message_hash = blake3::hash(&node_message_bytes);

        let result = utils::validate_signed_node_message(
            node_message,
            node_message_hash.as_bytes(),
            &signed_node_message.signature,
            &tx,
        )
        .await;

        assert!(matches!(
            result,
            Err(AtomaP2pNodeError::NodeSmallIdOwnershipVerificationError(_))
        ));
    }

    #[tokio::test]
    async fn test_validate_usage_metrics_message_state_manager_error() {
        let keystore = create_test_keystore();
        let (tx, rx) = unbounded();

        // Create valid metrics
        let signed_node_message = create_test_signed_node_message(&keystore, 0, 1);
        let node_message = &signed_node_message.node_message;

        // Mock state manager channel error by dropping the receiver
        drop(rx);

        let mut node_message_bytes = Vec::new();
        ciborium::into_writer(node_message, &mut node_message_bytes).unwrap();
        let node_message_hash = blake3::hash(&node_message_bytes);

        let result = utils::validate_signed_node_message(
            node_message,
            node_message_hash.as_bytes(),
            &signed_node_message.signature,
            &tx,
        )
        .await;
        assert!(matches!(
            result,
            Err(AtomaP2pNodeError::StateManagerError(_))
        ));
    }

    #[tokio::test]
    async fn test_validate_usage_metrics_message_response_channel_error() {
        let keystore = create_test_keystore();
        let (tx, rx) = unbounded();

        // Create valid metrics
        let signed_node_message = create_test_signed_node_message(&keystore, 0, 1);
        let node_message = &signed_node_message.node_message;

        // Mock response channel error
        tokio::spawn(async move {
            let (event, response_sender) = rx.recv_async().await.unwrap();
            if let AtomaP2pEvent::VerifyNodeSmallIdOwnership {
                node_small_id,
                sui_address: _,
            } = event
            {
                assert_eq!(node_small_id, 1);
                // Drop the sender without sending a response
                drop(response_sender);
            }
        });

        let mut node_message_bytes = Vec::new();
        ciborium::into_writer(node_message, &mut node_message_bytes).unwrap();
        let node_message_hash = blake3::hash(&node_message_bytes);

        let result = utils::validate_signed_node_message(
            node_message,
            node_message_hash.as_bytes(),
            &signed_node_message.signature,
            &tx,
        )
        .await;
        assert!(matches!(
            result,
            Err(AtomaP2pNodeError::NodeSmallIdOwnershipVerificationError(_))
        ));
    }
}
