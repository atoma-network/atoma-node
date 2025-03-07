use anyhow::{Context, Result};
use atoma_p2p::{config::AtomaP2pNodeConfig, service::AtomaP2pNode, types::AtomaP2pEvent};
use atoma_sui::config::Config as AtomaSuiConfig;
use atoma_utils::spawn_with_shutdown;
use clap::Parser;
use std::time::Duration;
use std::{path::PathBuf, sync::Arc};
use sui_keys::keystore::FileBasedKeystore;
use tokio::signal;
use tokio::sync::watch;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the configuration file
    #[arg(short, long)]
    config_path: String,

    /// Node ID (used to identify this node in logs)
    #[arg(short, long, default_value = "1")]
    node_id: u64,
}

struct Config {
    sui: AtomaSuiConfig,
    p2p: AtomaP2pNodeConfig,
}

impl Config {
    pub fn load(config_path: &str) -> Self {
        let config = AtomaP2pNodeConfig::from_file_path(config_path);
        let sui_config = AtomaSuiConfig::from_file_path(config_path);
        Self {
            sui: sui_config,
            p2p: config,
        }
    }
}

#[tokio::main]
#[allow(clippy::too_many_lines)]
#[allow(clippy::redundant_pub_crate)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .try_init();

    info!(event = "startup", "Starting Atoma Proxy Service...");

    let args = Args::parse();
    tracing::info!("Loading configuration from: {}", args.config_path);

    let config = Config::load(&args.config_path);
    tracing::info!("Configuration loaded successfully");

    // Create a channel to receive events from the P2P node
    let (atoma_p2p_sender, atoma_p2p_receiver) = flume::unbounded();
    let (shutdown_sender, shutdown_receiver) = watch::channel(false);

    // Create and start the P2P node
    let keystore: FileBasedKeystore =
        FileBasedKeystore::new(&PathBuf::from(&config.sui.sui_keystore_path()))
            .with_context(|| "Failed to create keystore")?;

    let node = AtomaP2pNode::start(config.p2p, Arc::new(keystore), atoma_p2p_sender, false)
        .await
        .with_context(|| "Failed to start P2P node")?;

    let atoma_p2p_node_handle =
        spawn_with_shutdown(node.run(shutdown_receiver.clone()), shutdown_sender.clone());

    // Add a periodic heartbeat/metrics reporting task
    let heartbeat_task = {
        let shutdown_rx = shutdown_receiver.clone();
        let node_id = args.node_id;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            let mut shutdown_rx = shutdown_rx;
            let mut tick_count = 0;

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        tick_count += 1;
                        info!(
                            target = "atoma-p2p-tester",
                            event = "heartbeat",
                            node_id = node_id,
                            uptime_seconds = interval.period().as_secs() * tick_count,
                            "P2P tester node is healthy"
                        );
                    }
                    Ok(()) = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            break;
                        }
                    }
                }
            }

            info!("Heartbeat task shut down gracefully");
        })
    };

    // Spawn a task to handle events from the P2P node
    let event_handler = tokio::spawn(async move {
        let receiver = atoma_p2p_receiver;
        let mut shutdown_receiver = shutdown_receiver.clone();

        loop {
            tokio::select! {
                Ok(event) = receiver.recv_async() => {
                    match event {
                        (
                            AtomaP2pEvent::NodeMetricsRegistrationEvent {
                                public_url,
                                node_small_id,
                                timestamp,
                                country,
                                node_metrics,
                            },
                            _,
                        ) => {
                            info!(
                                "Received node metrics registration: node_id={}, public_url={}, country={}, timestamp={}",
                                node_small_id, public_url, country, timestamp
                            );
                            info!("Node metrics: {:?}", node_metrics);
                        }
                        (
                            AtomaP2pEvent::VerifyNodeSmallIdOwnership {
                                node_small_id,
                                sui_address,
                            },
                            response_sender,
                        ) => {
                            info!(
                                "Received node small ID ownership verification: node_id={}, sui_address={}",
                                node_small_id, sui_address
                            );
                            // For testing purposes, always verify as true
                            if let Some(sender) = response_sender {
                                let _ = sender.send(true);
                            }
                        }
                    }
                }
                Ok(()) = shutdown_receiver.changed() => {
                    if *shutdown_receiver.borrow() {
                        info!("Event handler received shutdown signal");
                        break;
                    }
                }
            }
        }
        info!("Event handler terminated gracefully");
    });

    // Wait for Ctrl+C
    info!(
        "P2P tester node {} is running. Press Ctrl+C to stop.",
        args.node_id
    );
    match signal::ctrl_c().await {
        Ok(()) => {
            info!("Shutting down P2P tester node {}", args.node_id);
        }
        Err(err) => {
            error!("Error waiting for Ctrl+C: {}", err);
        }
    }

    // Add timeout for graceful shutdown
    info!("Waiting for components to shut down gracefully...");
    let timeout = tokio::time::timeout(Duration::from_secs(5), atoma_p2p_node_handle).await;

    match timeout {
        Ok(result) => {
            if let Err(e) = result {
                error!("P2P node terminated with error: {:?}", e);
            } else {
                info!("P2P node shut down successfully");
            }
        }
        Err(_) => {
            error!("P2P node shutdown timed out after 5 seconds");
        }
    }

    let heartbeat_result = tokio::time::timeout(Duration::from_secs(2), heartbeat_task).await;
    let event_handler_result = tokio::time::timeout(Duration::from_secs(2), event_handler).await;

    if let Ok(result) = heartbeat_result {
        let _ = handle_task_result("Heartbeat", result);
    }
    if let Ok(result) = event_handler_result {
        let _ = handle_task_result("Event Handler", result);
    }

    Ok(())
}

fn handle_task_result<T>(
    task_name: &str,
    result: Result<T, tokio::task::JoinError>,
) -> Result<T, tokio::task::JoinError> {
    if let Err(e) = &result {
        error!(
            target = "atoma-p2p-tester",
            event = "component_error",
            component = task_name,
            error = ?e,
            "Component terminated with error"
        );
    } else {
        info!(
            target = "atoma-p2p-tester",
            event = "component_shutdown",
            component = task_name,
            "Component shut down gracefully"
        );
    }
    result
}
