use anyhow::{Context, Result};
use atoma_daemon::telemetry;
use atoma_p2p::{config::AtomaP2pNodeConfig, service::AtomaP2pNode, types::AtomaP2pEvent};
use atoma_sui::config::Config as AtomaSuiConfig;
use clap::Parser;
use std::time::Duration;
use std::{path::PathBuf, sync::Arc};
use sui_keys::keystore::FileBasedKeystore;
use tokio::signal;
use tracing::{error, info};

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
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Store both guards to keep logging active for the duration of the program
    let (_file_guard, _stdout_guard) =
        telemetry::setup_logging().context("Failed to setup logging")?;

    info!(event = "startup", "Starting Atoma Proxy Service...");

    let args = Args::parse();
    tracing::info!("Loading configuration from: {}", args.config_path);

    let config = Config::load(&args.config_path);
    tracing::info!("Configuration loaded successfully");

    // Create a channel to receive events from the P2P node
    let (atoma_p2p_sender, atoma_p2p_receiver) = flume::unbounded();

    // Create and start the P2P node
    let keystore: FileBasedKeystore =
        FileBasedKeystore::new(&PathBuf::from(&config.sui.sui_keystore_path()))?;

    let _node = AtomaP2pNode::start(config.p2p, Arc::new(keystore), atoma_p2p_sender, false)?;

    // Spawn a task to handle events from the P2P node
    tokio::spawn(async move {
        while let Ok(event) = atoma_p2p_receiver.recv_async().await {
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

    // Give some time for graceful shutdown
    tokio::time::sleep(Duration::from_secs(1)).await;

    Ok(())
}
