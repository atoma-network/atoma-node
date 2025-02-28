use crate::{
    config::AtomaP2pNodeConfig,
    errors::AtomaP2pNodeError,
    metrics::{
        NetworkMetrics, TOTAL_CONNECTIONS, TOTAL_DIALS_ATTEMPTED, TOTAL_DIALS_FAILED,
        TOTAL_INVALID_GOSSIPSUB_MESSAGES_RECEIVED, TOTAL_MDNS_DISCOVERIES,
    },
    metrics::{
        KAD_ROUTING_TABLE_SIZE, PEERS_CONNECTED, TOTAL_FAILED_GOSSIPSUB_PUBLISHES,
        TOTAL_GOSSIPSUB_MESSAGES_FORWARDED, TOTAL_GOSSIPSUB_PUBLISHES,
        TOTAL_GOSSIPSUB_SUBSCRIPTIONS, TOTAL_INCOMING_CONNECTIONS, TOTAL_OUTGOING_CONNECTIONS,
    },
    timer::usage_metrics_timer_task,
    types::{AtomaP2pEvent, NodeMessage, SerializeWithSignature, SignedNodeMessage},
    utils::{extract_gossipsub_metrics, validate_signed_node_message},
};
use bytes::{BufMut, Bytes, BytesMut};
use flume::Sender;
use futures::StreamExt;
use libp2p::{
    gossipsub::{self},
    identify, identity, kad, mdns, noise,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, PeerId, StreamProtocol, Swarm, SwarmBuilder,
};
use libp2p::{
    metrics::{Metrics, Recorder, Registry},
    Multiaddr,
};
use opentelemetry::KeyValue;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;
use sui_keys::keystore::{AccountKeystore, FileBasedKeystore};
use tokio::{
    sync::{mpsc::UnboundedReceiver, oneshot, watch},
    task::JoinHandle,
};
use tracing::{debug, error, instrument};

/// The topic that the P2P network will use to gossip messages
const METRICS_GOSPUBSUB_TOPIC: &str = "atoma-p2p-usage-metrics";

/// The interval at which the metrics are updated
const METRICS_UPDATE_INTERVAL: Duration = Duration::from_secs(15);

/// The protocol name for the Kademlia DHT
const IPFS_PROTO_NAME: StreamProtocol = StreamProtocol::new("/ipfs/kad/1.0.0");

// Well connected nodes to bootstrap the network (see https://docs.ipfs.tech/concepts/public-utilities/#amino-dht-bootstrappers)
const BOOTSTRAP_NODES: [&str; 4] = [
    "QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
    "QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
    "QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb",
    "QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt",
];

pub type StateManagerEvent = (AtomaP2pEvent, Option<oneshot::Sender<bool>>);

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

    /// Add network metrics
    network_metrics: NetworkMetrics,

    /// Add registry field
    metrics_registry: Registry,
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
    #[instrument(level = "debug", skip_all)]
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
        let mut metrics_registry = libp2p::metrics::Registry::default();

        let local_key = identity::Keypair::generate_ed25519();

        let mut swarm = SwarmBuilder::with_existing_identity(local_key)
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_quic()
            .with_dns()?
            .with_bandwidth_metrics(&mut metrics_registry)
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

                let gossipsub = gossipsub::Behaviour::new_with_metrics(
                    gossipsub::MessageAuthenticity::Signed(key.clone()),
                    gossipsub_config,
                    &mut metrics_registry,
                    gossipsub::MetricsConfig::default(),
                )
                .map_err(|e| AtomaP2pNodeError::InvalidConfig(e.to_string()))?;

                let mut cfg = kad::Config::new(IPFS_PROTO_NAME);
                cfg.set_query_timeout(Duration::from_secs(5 * 60));
                let store = kad::store::MemoryStore::new(key.public().to_peer_id());
                let kademlia = kad::Behaviour::with_config(key.public().to_peer_id(), store, cfg);

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

        // Parse TCP address from the first element in listen_addrs
        let tcp_addr: Multiaddr = config.listen_addrs[0].parse().map_err(|e| {
            error!(
                target = "atoma-p2p",
                event = "address_parse_error",
                listen_addr = %config.listen_addrs[0],
                error = %e,
                "Failed to parse TCP listen address"
            );
            AtomaP2pNodeError::ListenAddressParseError(e)
        })?;

        // Parse QUIC address from the second element in listen_addrs
        let quic_addr: Multiaddr = config.listen_addrs[1].parse().map_err(|e| {
            error!(
                target = "atoma-p2p",
                event = "address_parse_error",
                listen_addr = %config.listen_addrs[1],
                error = %e,
                "Failed to parse QUIC listen address"
            );
            AtomaP2pNodeError::ListenAddressParseError(e)
        })?;

        if let Err(e) = swarm.listen_on(quic_addr.clone()) {
            error!(
                target = "atoma-p2p",
                event = "listen_on_error",
                listen_addr = quic_addr.to_string(),
                error = %e,
                "Failed to listen on QUIC address"
            );
            return Err(AtomaP2pNodeError::SwarmListenOnError(e));
        }

        if let Err(e) = swarm.listen_on(tcp_addr.clone()) {
            error!(
                target = "atoma-p2p",
                event = "listen_on_error",
                listen_addr = tcp_addr.to_string(),
                error = %e,
                "Failed to listen on TCP address"
            );
            return Err(AtomaP2pNodeError::SwarmListenOnError(e));
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

        let network_metrics = NetworkMetrics::default();

        debug!(
            target = "atoma-p2p",
            event = "node_started",
            peer_id = %swarm.local_peer_id(),
            "Libp2p node started"
        );

        debug!(
            target = "atoma-p2p",
            event = "listening_addresses",
            listen_addrs = ?swarm.listeners().map(ToString::to_string).collect::<Vec<_>>(),
            "Listening on addresses"
        );

        // Initialize Kademlia's bootstrap process
        for peer_id in BOOTSTRAP_NODES {
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

        Ok(Self {
            keystore,
            swarm,
            timer_join_handle,
            state_manager_sender,
            usage_metrics_rx,
            is_client,
            network_metrics,
            metrics_registry,
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
    #[instrument(level = "debug", skip_all)]
    pub async fn run(
        mut self,
        mut shutdown_signal: watch::Receiver<bool>,
    ) -> Result<(), AtomaP2pNodeError> {
        // Create a metrics update interval
        let mut metrics_interval = tokio::time::interval(METRICS_UPDATE_INTERVAL);
        let metrics = Metrics::new(&mut self.metrics_registry);
        let peer_id = self.swarm.local_peer_id().to_base58();

        loop {
            tokio::select! {
                // Add metrics interval to the select
                _ = metrics_interval.tick() => {
                    self.network_metrics.update_metrics();

                    let network_info = self.swarm.network_info();

                    extract_gossipsub_metrics(&self.swarm.behaviour_mut().gossipsub);

                    let peer_id_kv = KeyValue::new("peerId", peer_id.clone());
                    let peer_id_kv_slice = &[peer_id_kv];

                    #[allow(clippy::cast_possible_wrap, clippy::cast_lossless)]
                    {
                        PEERS_CONNECTED.record(network_info.num_peers() as i64, peer_id_kv_slice);
                        TOTAL_INCOMING_CONNECTIONS.record(network_info.connection_counters().num_established_incoming() as u64, peer_id_kv_slice);
                        TOTAL_OUTGOING_CONNECTIONS.record(network_info.connection_counters().num_established_outgoing() as u64, peer_id_kv_slice);
                        TOTAL_CONNECTIONS.record(network_info.connection_counters().num_connections() as u64, peer_id_kv_slice);
                    }
                }

                event = self.swarm.select_next_some() => {
                    match event {
                        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                            message_id,
                            message,
                            propagation_source,
                        })) => {
                            match self.handle_gossipsub_message(message.data.into(), &message_id, &propagation_source).await {
                                Ok(()) => {
                                    TOTAL_GOSSIPSUB_MESSAGES_FORWARDED.add(1, &[KeyValue::new("peerId", self.swarm.local_peer_id().to_base58())]);
                                }
                                Err(e) => {
                                    TOTAL_INVALID_GOSSIPSUB_MESSAGES_RECEIVED.add(1, &[KeyValue::new("peerId", self.swarm.local_peer_id().to_base58())]);
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
                            // Record subscript metrics
                            TOTAL_GOSSIPSUB_SUBSCRIPTIONS.add(1, &[KeyValue::new("topic", topic.to_string())]);
                            metrics.record(&gossipsub::Event::Subscribed {
                                peer_id,
                                topic: topic.clone(),
                            });

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
                            // Record unsubscription metrics
                            TOTAL_GOSSIPSUB_SUBSCRIPTIONS.add(-1, &[KeyValue::new("topic", topic.to_string())]);
                            metrics.record(&gossipsub::Event::Unsubscribed {
                                peer_id,
                                topic: topic.clone(),
                            });

                            debug!(
                                target = "atoma-p2p",
                                event = "gossipsub_unsubscribed",
                                peer_id = %peer_id,
                                topic = %topic,
                                "Peer unsubscribed from topic"
                            );
                        }
                        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Mdns(mdns::Event::Discovered(discovered_peers))) => {
                            let peer_count = discovered_peers.len() as u64;
                            debug!(
                                target = "atoma-p2p",
                                event = "mdns_discovered",
                                peer_count = %peer_count,
                                "Mdns discovered peers"
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
                            // Record discovery metrics
                            TOTAL_MDNS_DISCOVERIES.add(peer_count, &[KeyValue::new("peerId", peer_id.clone())]);
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
                        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Kademlia(kad::Event::RoutingUpdated {
                            peer,
                            is_new_peer,
                            addresses,
                            bucket_range,
                            old_peer,
                        })) => {
                            debug!(
                                target = "atoma-p2p",
                                event = "kademlia_routing_updated",
                                peer = %peer,
                                is_new_peer = %is_new_peer,
                                addresses = ?addresses,
                                bucket_range = ?bucket_range,
                                old_peer = ?old_peer,
                                "Kademlia routing updated"
                            );
                            KAD_ROUTING_TABLE_SIZE.record(addresses.len() as u64, &[KeyValue::new("peerId", peer.to_base58())]);
                            metrics.record(&kad::Event::RoutingUpdated {
                                peer,
                                is_new_peer,
                                addresses,
                                bucket_range,
                                old_peer,
                            });
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
                            num_established,
                            ..
                        } => {
                            debug!(
                                target = "atoma-p2p",
                                event = "peer_connection_closed",
                                peer_id = %peer_id,
                                connection_id = %connection_id,
                                num_established = %num_established,
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
                            TOTAL_DIALS_ATTEMPTED.add(1, &[KeyValue::new("peerId", peer_id.unwrap().to_base58())]);
                        }
                        SwarmEvent::OutgoingConnectionError {
                            peer_id,
                            error,
                            ..
                        } => {
                            TOTAL_DIALS_FAILED.add(1, &[KeyValue::new("peerId", peer_id.unwrap().to_base58())]);
                            error!(
                                target = "atoma-p2p",
                                event = "outgoing_connection_error",
                                peer_id = ?peer_id,
                                error = %error,
                                "Outgoing connection error"
                            );
                        }
                        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Identify(identify_event)) => {
                            tracing::debug!(
                                target = "atoma-p2p",
                                event = "identify",
                                identify_event = ?identify_event,
                                "Identify event"
                            );
                            metrics.record(&identify_event);
                        }
                        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Kademlia(kad_event)) => {
                            tracing::debug!(
                                target = "atoma-p2p",
                                event = "kad",
                                kad_event = ?kad_event,
                                "Kad event"
                            );
                            metrics.record(&kad_event);
                        }
                        swarm_event => {
                            tracing::debug!(
                                target = "atoma-p2p",
                                event = "swarm_event",
                                swarm_event = ?swarm_event,
                                "Swarm event"
                            );
                            metrics.record(&swarm_event);
                        }
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
        message_data: Bytes,
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
        let signed_node_message = SignedNodeMessage::deserialize_with_signature(&message_data)?;
        let signature_len = signed_node_message.signature.len();
        debug!(
            target = "atoma-p2p",
            event = "gossipsub_message_data",
            message_id = %message_id,
            propagation_source = %propagation_source,
            "Received gossipsub message data"
        );
        let node_message = &signed_node_message.node_message;
        let node_message_hash = blake3::hash(&message_data[signature_len..]);
        let message_acceptance = match validate_signed_node_message(
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
        let mut buffer = BytesMut::new();
        ciborium::into_writer(&node_message, (&mut buffer).writer())?;
        let bytes = buffer.freeze();
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
            signature: Bytes::copy_from_slice(signature.as_ref()),
        };
        let serialized_signed_node_message = signed_node_message.serialize_with_signature()?;
        let topic = gossipsub::IdentTopic::new(METRICS_GOSPUBSUB_TOPIC);
        self.swarm
            .behaviour_mut()
            .gossipsub
            .publish(topic, serialized_signed_node_message)
            .map_err(|e| {
                error!(
                    target = "atoma-p2p",
                    event = "publish_metrics_error",
                    error = %e,
                    "Failed to publish metrics"
                );
                TOTAL_FAILED_GOSSIPSUB_PUBLISHES.add(
                    1,
                    &[KeyValue::new(
                        "peerId",
                        self.swarm.local_peer_id().to_base58(),
                    )],
                );
                AtomaP2pNodeError::PublishError(e.to_string())
            })?;

        TOTAL_GOSSIPSUB_PUBLISHES.add(
            1,
            &[KeyValue::new(
                "peerId",
                self.swarm.local_peer_id().to_base58(),
            )],
        );

        Ok(())
    }
}
