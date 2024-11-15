use std::{path::Path, str::FromStr, sync::Arc};

use anyhow::{Context, Result};
use atoma_daemon::{
    config::AtomaDaemonConfig,
    daemon::{run_daemon, DaemonState},
};
use atoma_state::{config::AtomaStateManagerConfig, AtomaState};
use atoma_sui::client::AtomaSuiClient;
use clap::Parser;
use sui_sdk::types::base_types::ObjectID;
use tokio::{
    net::TcpListener,
    sync::{watch, RwLock},
    try_join,
};
use tracing::info;
use tracing_appender::{
    non_blocking,
    rolling::{RollingFileAppender, Rotation},
};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan, time::UtcTime},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Registry,
};

/// The directory where the logs are stored.
const LOGS: &str = "./logs";
/// The log file name.
const LOG_FILE: &str = "atoma-daemon-service.log";

#[derive(Parser)]
struct DaemonArgs {
    #[arg(short, long)]
    config_path: String,
}

/// Spawns a task that will automatically trigger shutdown if it encounters an error
///
/// This helper function wraps a future in a tokio task that monitors its execution.
/// If the wrapped future returns an error, it will automatically trigger a shutdown
/// signal through the provided sender.
///
/// # Arguments
///
/// * `f` - The future to execute, which must return a `Result<()>`
/// * `shutdown_sender` - A channel sender used to signal shutdown to other parts of the application
///
/// # Returns
///
/// Returns a `JoinHandle` for the spawned task
///
/// # Example
///
/// ```
/// let (shutdown_tx, shutdown_rx) = watch::channel(false);
/// let handle = spawn_with_shutdown(some_fallible_task(), shutdown_tx);
/// ```
fn spawn_with_shutdown<F>(
    f: F,
    shutdown_sender: watch::Sender<bool>,
) -> tokio::task::JoinHandle<Result<()>>
where
    F: std::future::Future<Output = Result<()>> + Send + 'static,
{
    tokio::task::spawn(async move {
        let res = f.await;
        if res.is_err() {
            // Only send shutdown signal if the task failed
            shutdown_sender
                .send(true)
                .context("Failed to send shutdown signal")?;
        }
        res
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_logging()?;
    let args = DaemonArgs::parse();
    let daemon_config = AtomaDaemonConfig::from_file_path(args.config_path.clone());
    let state_manager_config = AtomaStateManagerConfig::from_file_path(args.config_path.clone());
    let client = Arc::new(RwLock::new(
        AtomaSuiClient::new_from_config(args.config_path).await?,
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
        run_daemon(daemon_state, tcp_listener, shutdown_receiver.clone()),
        shutdown_sender.clone(),
    );
    info!(
        target = "atoma_daemon",
        event = "atoma-daemon-stop",
        "Atoma daemon service stopped gracefully..."
    );

    let ctrl_c = tokio::task::spawn(async move {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!(
                    target = "atoma_daemon",
                    event = "atoma-daemon-stop",
                    "ctrl-c received, sending shutdown signal"
                );
                shutdown_sender
                    .send(true)
                    .context("Failed to send shutdown signal")?;
                Ok::<(), anyhow::Error>(())
            }
            _ = shutdown_receiver.changed() => {
                Ok::<(), anyhow::Error>(())
            }
        }
    });

    let (daemon_result, _) = try_join!(daemon_handle, ctrl_c)?;

    daemon_result
}

fn setup_logging() -> Result<()> {
    let log_dir = Path::new(LOGS);
    let file_appender = RollingFileAppender::new(Rotation::DAILY, log_dir, LOG_FILE);
    let (non_blocking_appender, _guard) = non_blocking(file_appender);
    let file_layer = fmt::layer()
        .json()
        .with_timer(UtcTime::rfc_3339())
        .with_current_span(true)
        .with_writer(non_blocking_appender);

    let console_layer = fmt::layer()
        .pretty()
        .with_target(true)
        .with_line_number(true)
        .with_file(true)
        .with_span_events(FmtSpan::FULL);

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,atoma_daemon=debug"));

    Registry::default()
        .with(env_filter)
        .with(console_layer)
        .with(file_layer)
        .init();
    Ok(())
}
