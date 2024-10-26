use std::{path::Path, str::FromStr, sync::Arc};

use anyhow::Result;
use atoma_daemon::{
    config::AtomaDaemonConfig,
    daemon::{run_daemon, DaemonState},
};
use atoma_state::{config::AtomaStateManagerConfig, AtomaState};
use atoma_sui::client::AtomaSuiClient;
use clap::Parser;
use sui_sdk::types::base_types::ObjectID;
use tokio::{net::TcpListener, sync::RwLock};
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
    config_file_path: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_logging()?;
    let args = DaemonArgs::parse();
    let daemon_config = AtomaDaemonConfig::from_file_path(args.config_file_path.clone());
    let state_manager_config =
        AtomaStateManagerConfig::from_file_path(args.config_file_path.clone());
    let client = Arc::new(RwLock::new(
        AtomaSuiClient::new(args.config_file_path).await?,
    ));

    info!("Starting a new AtomaStateManager instance...");

    let atoma_state = AtomaState::new_from_url(state_manager_config.database_url).await?;
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
        "Starting the Atoma daemon service, on {}",
        daemon_config.service_bind_address
    );
    run_daemon(daemon_state, tcp_listener).await?;
    info!("Atoma daemon service stopped gracefully...");

    Ok(())
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
