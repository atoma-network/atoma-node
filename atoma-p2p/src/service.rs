use crate::{
    config::P2pAtomaNodeConfig,
    types::{GossipMessage, PublicAddressMessage, RequestPublicAddressMessage},
};
use futures::StreamExt;
use libp2p::{
    gossipsub::{self, ConfigBuilderError},
    mdns, noise,
    swarm::{DialError, NetworkBehaviour, SwarmEvent},
    tcp, yamux, Multiaddr, Swarm, SwarmBuilder, TransportError,
};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use sui_keys::keystore::{AccountKeystore, FileBasedKeystore};
use thiserror::Error;
use tokio::sync::watch;
use tracing::{debug, error, instrument};

const GOSPUBSUB_TOPIC: &str = "atoma-p2p";

#[derive(NetworkBehaviour)]
struct MyBehaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
}

pub struct P2pAtomaNode {
    keystore: Arc<FileBasedKeystore>,
    swarm: Swarm<MyBehaviour>,
    public_url: String,
    node_small_id: u64,
}

impl P2pAtomaNode {
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
                Ok(MyBehaviour { gossipsub, mdns })
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
                            propagation_source,
                            message_id,
                            message,
                        })) => {
                            debug!(
                                target = "atoma-p2p",
                                event = "gossipsub_message",
                                message_id = %message_id,
                                "Received gossipsub message"
                            );
                            let gossip_message: GossipMessage = serde_json::from_slice(&message.data)?;
                            match gossip_message {
                                GossipMessage::PublicAddressMessage(_) => {
                                    debug!(
                                        target = "atoma-p2p",
                                        event = "gossipsub_message_data",
                                        "Received gossipsub message data"
                                    );

                                    // Rebroadcast the message to the network peers
                                    let topic = gossipsub::IdentTopic::new(GOSPUBSUB_TOPIC);
                                    if let Err(e) = self.swarm.behaviour_mut().gossipsub.publish(
                                        topic,
                                        message.data.clone()
                                    ) {
                                        error!(
                                            target = "atoma-p2p",
                                            event = "gossipsub_message_rebroadcast_error",
                                            error = %e,
                                            "Failed to rebroadcast gossipsub message"
                                        );
                                    }
                                },
                                GossipMessage::RequestPublicAddressMessage(_) => {
                                    debug!(
                                        target = "atoma-p2p",
                                        event = "gossipsub_message_data",
                                        "Received gossipsub message data"
                                    );
                                    let timestamp = std::time::Instant::now().elapsed().as_secs();
                                    let public_url_hash = blake3::hash(&self.public_url.as_bytes());
                                    let address = self.keystore.addresses()[0];
                                    let signature = self.keystore.sign_hashed(&address, public_url_hash.as_bytes()).map_err(|e| {
                                        error!(
                                            target = "atoma-p2p",
                                            event = "gossipsub_message_sign_hashed_error",
                                            error = %e,
                                            "Failed to sign hashed message"
                                        );
                                        P2pAtomaNodeError::SignHashedMessageError(e.to_string())
                                    })?;
                                    let response = GossipMessage::PublicAddressMessage(
                                        PublicAddressMessage {
                                            address: self.public_url.clone(),
                                            signature: signature.as_ref().to_vec(),
                                            timestamp,
                                            node_small_id: self.node_small_id,
                                        }
                                    );
                                    let serialized_response = serde_json::to_vec(&response)?;
                                    let topic = gossipsub::IdentTopic::new(GOSPUBSUB_TOPIC);
                                    if let Err(e) = self.swarm.behaviour_mut().gossipsub.publish(
                                        topic,
                                        serialized_response
                                    ) {
                                        error!(
                                            target = "atoma-p2p",
                                            event = "gossipsub_message_rebroadcast_error",
                                            error = %e,
                                            "Failed to rebroadcast gossipsub message"
                                        );
                                    }
                                }
                                _ => {}
                            }

                        }
                        _ => {}
                    }
                    // Handle incoming connections
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
}
