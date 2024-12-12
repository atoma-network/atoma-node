use crate::config::P2pAtomaNodeConfig;
use futures::StreamExt;
use libp2p::{
    gossipsub::{self, ConfigBuilderError},
    mdns, noise,
    swarm::NetworkBehaviour,
    tcp, yamux, Swarm, SwarmBuilder, TransportError,
};
use std::hash::{DefaultHasher, Hash, Hasher};
use thiserror::Error;
use tokio::sync::watch;
use tracing::{error, instrument};

const GOSPUBSUB_TOPIC: &str = "atoma-p2p";

#[derive(NetworkBehaviour)]
struct MyBehaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
}

pub struct P2pAtomaNode {
    swarm: Swarm<MyBehaviour>,
}

impl P2pAtomaNode {
    #[instrument(level = "info", skip(config))]
    pub fn start(config: P2pAtomaNodeConfig) -> Result<Self, P2pAtomaNodeError> {
        let mut swarm = SwarmBuilder::with_new_identity()
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )
            .map_err(|e| {
                error!("Failed to build swarm, with error: {e}");
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
                error!("Failed to build behaviour, with error: {e}");
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
                error!("Failed to subscribe to topic, with error: {e}");
                P2pAtomaNodeError::GossipsubSubscriptionError(e)
            })?;
        swarm.listen_on(config.listen_addr.parse()?).map_err(|e| {
            error!("Failed to listen on address, with error: {e}");
            P2pAtomaNodeError::SwarmListenOnError(e)
        })?;

        Ok(Self { swarm })
    }

    #[instrument(level = "info", skip(self))]
    pub async fn run(
        mut self,
        mut shutdown_signal: watch::Receiver<bool>,
    ) -> Result<(), P2pAtomaNodeError> {
        loop {
            tokio::select! {
                _ = self.swarm.select_next_some() => {
                    // Handle incoming connections
                }
                shutdown_signal_changed = shutdown_signal.changed() => {
                    match shutdown_signal_changed {
                        Ok(()) => {
                            if *shutdown_signal.borrow() {
                                tracing::trace!(
                                    target = "atoma-state-manager",
                                    event = "shutdown_signal",
                                    "Shutdown signal received, shutting down"
                                );
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                target = "atoma-state-manager",
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
}
