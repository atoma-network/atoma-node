use std::{str::FromStr, sync::Arc};

use anyhow::Result;
use atoma_daemon::{config::AtomaDaemonConfig, daemon::{run_daemon, DaemonState}};
use atoma_state::{config::StateManagerConfig, StateManager};
use atoma_sui::client::AtomaSuiClient;
use clap::Parser;
use sui_sdk::types::base_types::ObjectID;
use tokio::{net::TcpListener, sync::RwLock};
use tracing::info;

#[derive(Parser)]
struct DaemonArgs {
    #[arg(short, long)]
    config_file_path: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = DaemonArgs::parse();
    let daemon_config = AtomaDaemonConfig::from_file_path(args.config_file_path.clone());
    let state_manager_config = StateManagerConfig::from_file_path(args.config_file_path.clone());
    let client = Arc::new(RwLock::new(
        AtomaSuiClient::new(args.config_file_path).await?,
    ));
    
    info!("Starting a new StateManager instance...");
    
    let state_manager = StateManager::new_from_url(state_manager_config.database_url).await?;
    let tcp_listener = TcpListener::bind(daemon_config.service_bind_address.clone()).await?;
    let daemon_state = DaemonState {
        client,
        state_manager,
        node_badges: daemon_config
            .node_badges
            .iter()
            .map(|(id, value)| (ObjectID::from_str(id).unwrap(), *value))
            .collect(),
    };

    info!("Starting the Atoma daemon service, on {}", daemon_config.service_bind_address);
    run_daemon(daemon_state, tcp_listener).await?;
    info!("Atoma daemon service stopped gracefully...");
    
    Ok(())
}
