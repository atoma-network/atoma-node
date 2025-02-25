use std::{str::FromStr, sync::Arc};

use anyhow::{Context, Result};
use atoma_daemon::{
    config::AtomaDaemonConfig,
    server::{run_server, DaemonState},
    telemetry,
};
use atoma_state::{config::AtomaStateManagerConfig, AtomaState};
use atoma_sui::client::Client;
use atoma_utils::spawn_with_shutdown;
use clap::Parser;
use sui_sdk::types::base_types::ObjectID;
use tokio::{
    net::TcpListener,
    sync::{watch, RwLock},
    try_join,
};
use tracing::info;

#[derive(Parser)]
struct DaemonArgs {
    #[arg(short, long)]
    config_path: String,
}

#[tokio::main]
#[allow(clippy::redundant_pub_crate)]
async fn main() -> Result<()> {
    // Store both guards to keep logging active for the duration of the program
    let (_file_guard, _stdout_guard) =
        telemetry::setup_logging().context("Failed to setup logging")?;

    let args = DaemonArgs::parse();
    let daemon_config = AtomaDaemonConfig::from_file_path(args.config_path.clone());
    let state_manager_config = AtomaStateManagerConfig::from_file_path(args.config_path.clone());
    let client = Arc::new(RwLock::new(
        Client::new_from_config(args.config_path).await?,
    ));

    info!(
        target = "atoma_daemon",
        event = "atoma-daemon-start",
        "Starting a new AtomaStateManager instance..."
    );

    let atoma_state = AtomaState::new_from_url(&state_manager_config.database_url).await?;
    let tcp_listener = TcpListener::bind(daemon_config.service_bind_address.clone()).await?;
    let daemon_state = DaemonState {
        client,
        atoma_state,
        node_badges: daemon_config
            .node_badges
            .iter()
            .map(|(id, value)| (ObjectID::from_str(id).unwrap(), *value))
            .collect(),
    };

    info!(
        target = "atoma_daemon",
        event = "atoma-daemon-start",
        "Starting the Atoma daemon service, on {}",
        daemon_config.service_bind_address
    );
    let (shutdown_sender, mut shutdown_receiver) = watch::channel(false);

    let daemon_handle = spawn_with_shutdown(
        run_server(daemon_state, tcp_listener, shutdown_receiver.clone()),
        shutdown_sender.clone(),
    );
    info!(
        target = "atoma_daemon",
        event = "atoma-daemon-stop",
        "Atoma daemon service stopped gracefully..."
    );

    let ctrl_c = tokio::task::spawn(async move {
        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                info!(
                    target = "atoma_daemon",
                    event = "atoma-daemon-stop",
                    "ctrl-c received, sending shutdown signal"
                );
                shutdown_sender
                    .send(true)
                    .context("Failed to send shutdown signal")?;
                result.map_err(anyhow::Error::from)
            }
            _ = shutdown_receiver.changed() => {
                Ok(())
            }
        }
    });

    let (daemon_result, _) = try_join!(daemon_handle, ctrl_c)?;

    // Before the program exits, ensure all spans are exported
    telemetry::shutdown();

    daemon_result
}
