use std::{path::Path, sync::Arc};

use anyhow::{Context, Result};
use atoma_service::{
    config::AtomaServiceConfig,
    server::{run_server, AppState},
};
use atoma_state::{config::AtomaStateManagerConfig, AtomaStateManager};
use atoma_sui::{AtomaSuiConfig, SuiEventSubscriber};
use clap::Parser;
use dotenv::dotenv;
use futures::future::try_join_all;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use sui_keys::keystore::FileBasedKeystore;
use tokenizers::Tokenizer;
use tokio::{net::TcpListener, sync::watch, try_join};
use tracing::{error, info, instrument, warn};
use tracing_appender::{
    non_blocking,
    rolling::{RollingFileAppender, Rotation},
};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan, time::UtcTime},
    prelude::*,
    EnvFilter, Registry,
};

/// The name of the environment variable for the Hugging Face token
const HF_TOKEN: &str = "HF_TOKEN";
/// The directory where the logs are stored.
const LOGS: &str = "./logs";
/// The log file name.
const LOG_FILE: &str = "atoma-node-service.log";

/// Command line arguments for the Atoma node
#[derive(Parser)]
struct Args {
    /// Index of the address to use from the keystore
    #[arg(short, long, default_value_t = 0)]
    address_index: usize,

    /// Path to the configuration file
    #[arg(short, long)]
    config_path: String,
}

/// Configuration for the Atoma node.
///
/// This struct holds the configuration settings for various components
/// of the Atoma node, including the Sui, service, and state manager configurations.
#[derive(Debug)]
struct Config {
    /// Configuration for the Sui component.
    sui: AtomaSuiConfig,

    /// Configuration for the service component.
    service: AtomaServiceConfig,

    /// Configuration for the state manager component.
    state: AtomaStateManagerConfig,
}

impl Config {
    async fn load(path: String) -> Self {
        Self {
            sui: AtomaSuiConfig::from_file_path(path.clone()),
            service: AtomaServiceConfig::from_file_path(path.clone()),
            state: AtomaStateManagerConfig::from_file_path(path),
        }
    }
}

/// Initializes tokenizers for multiple models by fetching their configurations from HuggingFace.
///
/// This function concurrently fetches tokenizer configurations for multiple models from HuggingFace's
/// repository and initializes them. Each tokenizer is wrapped in an Arc for safe sharing across threads.
///
/// # Arguments
///
/// * `models` - A slice of model names/paths on HuggingFace (e.g., ["facebook/opt-125m"])
/// * `revisions` - A slice of revision/branch names corresponding to each model (e.g., ["main"])
///
/// # Returns
///
/// Returns a `Result` containing a vector of Arc-wrapped tokenizers on success, or an error if:
/// - Failed to fetch tokenizer configuration from HuggingFace
/// - Failed to parse the tokenizer JSON
/// - Any other network or parsing errors occur
///
/// # Examples
///
/// ```rust,ignore
/// use anyhow::Result;
///
/// #[tokio::main]
/// async fn example() -> Result<()> {
///     let models = vec!["facebook/opt-125m".to_string()];
///     let revisions = vec!["main".to_string()];
///     
///     let tokenizers = initialize_tokenizers(&models, &revisions).await?;
///     Ok(())
/// }
/// ```
#[instrument(level = "info", skip(models, revisions))]
async fn initialize_tokenizers(
    models: &[String],
    revisions: &[String],
    hf_token: String,
) -> Result<Vec<Arc<Tokenizer>>> {
    let api = ApiBuilder::new()
        .with_progress(true)
        .with_token(Some(hf_token))
        .build()?;
    let fetch_futures: Vec<_> = models
        .iter()
        .zip(revisions.iter())
        .map(|(model, revision)| {
            let api = api.clone();
            async move {
                let repo = api.repo(Repo::with_revision(
                    model.clone(),
                    RepoType::Model,
                    revision.clone(),
                ));

                let tokenizer_filename = repo
                    .get("tokenizer.json")
                    .expect("Failed to get tokenizer.json");

                Tokenizer::from_file(tokenizer_filename)
                    .map_err(|e| {
                        anyhow::anyhow!(format!(
                            "Failed to parse tokenizer for model {}, with error: {}",
                            model, e
                        ))
                    })
                    .map(Arc::new)
            }
        })
        .collect();

    try_join_all(fetch_futures).await
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_logging(LOGS).context("Failed to setup logging")?;
    dotenv().ok();

    let args = Args::parse();
    let config = Config::load(args.config_path).await;

    info!("Starting Atoma node service");

    let (shutdown_sender, shutdown_receiver) = watch::channel(false);
    let (event_subscriber_sender, event_subscriber_receiver) = flume::unbounded();
    let (state_manager_sender, state_manager_receiver) = flume::unbounded();

    info!(
        target = "atoma-node-service",
        event = "keystore_path",
        keystore_path = config.sui.sui_keystore_path(),
        "Starting with Sui's keystore instance"
    );

    let keystore = FileBasedKeystore::new(&config.sui.sui_config_path().into())
        .context("Failed to initialize keystore")?;

    info!(
        target = "atoma-node-service",
        event = "state_manager_service_spawn",
        database_url = config.state.database_url,
        "Spawning state manager service"
    );
    let state_manager_shutdown_receiver = shutdown_receiver.clone();
    let state_manager_handle = tokio::spawn(async move {
        let state_manager = AtomaStateManager::new_from_url(
            config.state.database_url,
            event_subscriber_receiver,
            state_manager_receiver,
        )
        .await?;
        state_manager.run(state_manager_shutdown_receiver).await?;
        Ok::<(), anyhow::Error>(())
    });

    let package_id = config.sui.atoma_package_id();
    info!(
        target = "atoma-node-service",
        event = "subscriber_service_spawn",
        package_id = package_id.to_string(),
        "Spawning subscriber service"
    );
    let subscriber =
        SuiEventSubscriber::new(config.sui, event_subscriber_sender, shutdown_receiver);

    info!(
        target = "atoma-node-service",
        event = "subscriber_service_spawn",
        package_id = package_id.to_string(),
        "Subscribing to Sui events"
    );
    let subscriber_handle = tokio::spawn(async move {
        info!(
            target = "atoma-node-service",
            event = "subscriber_service_run",
            package_id = package_id.to_string(),
            "Running Sui event subscriber"
        );
        let result = subscriber.run().await;
        info!(
            target = "atoma-node-service",
            event = "subscriber_service_finished",
            package_id = package_id.to_string(),
            "Sui event subscriber finished"
        );
        result.map_err(|e| anyhow::anyhow!(e))
    });

    let hf_token = std::env::var(HF_TOKEN)
        .context(format!("Variable {} not set in the .env file", HF_TOKEN))?;
    let tokenizers =
        initialize_tokenizers(&config.service.models, &config.service.revisions, hf_token).await?;

    let app_state = AppState {
        state_manager_sender,
        tokenizers: Arc::new(tokenizers),
        models: Arc::new(vec![]),
        inference_service_url: config
            .service
            .inference_service_url
            .context("Inference service URL not configured")?,
        keystore: Arc::new(keystore),
        address_index: args.address_index,
    };

    let tcp_listener = TcpListener::bind(&config.service.service_bind_address)
        .await
        .context("Failed to bind TCP listener")?;

    info!(
        target = "atoma-node-service",
        event = "atoma_node_service_spawn",
        bind_address = config.service.service_bind_address,
        "Starting Atoma node service"
    );

    let server_handle = tokio::spawn(run_server(app_state, tcp_listener, shutdown_sender));

    // Wait for shutdown signal and handle cleanup
    let (subscriber_result, state_manager_result, server_result) =
        try_join!(subscriber_handle, state_manager_handle, server_handle)?;
    handle_tasks_results(subscriber_result, state_manager_result, server_result)?;

    info!(
        target = "atoma-node-service",
        event = "atoma_node_service_shutdown",
        "Atoma node service shut down successfully"
    );
    Ok(())
}

/// Configure logging with JSON formatting, file output, and console output
fn setup_logging<P: AsRef<Path>>(log_dir: P) -> Result<()> {
    // Set up file appender with rotation
    let file_appender = RollingFileAppender::new(Rotation::DAILY, log_dir, LOG_FILE);

    // Create a non-blocking writer
    let (non_blocking_appender, _guard) = non_blocking(file_appender);

    // Create JSON formatter for file output
    let file_layer = fmt::layer()
        .json()
        .with_timer(UtcTime::rfc_3339())
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_target(true)
        .with_line_number(true)
        .with_file(true)
        .with_current_span(true)
        .with_span_list(true)
        .with_writer(non_blocking_appender);

    // Create console formatter for development
    let console_layer = fmt::layer()
        .pretty()
        .with_target(true)
        .with_thread_ids(true)
        .with_line_number(true)
        .with_file(true)
        .with_span_events(FmtSpan::ENTER);

    // Create filter from environment variable or default to info
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,atoma_node_service=debug"));

    // Combine layers with filter
    Registry::default()
        .with(env_filter)
        .with(console_layer)
        .with(file_layer)
        .init();

    Ok(())
}

/// Handles the results of various tasks (subscriber, state manager, and server).
///
/// This function checks the results of the subscriber, state manager, and server tasks.
/// If any of the tasks return an error, it logs the error and returns it.
/// This is useful for ensuring that the application can gracefully handle failures
/// in any of its components and provide appropriate logging for debugging.
///
/// # Arguments
///
/// * `subscriber_result` - The result of the subscriber task, which may contain an error.
/// * `state_manager_result` - The result of the state manager task, which may contain an error.
/// * `server_result` - The result of the server task, which may contain an error.
///
/// # Returns
///
/// Returns a `Result<()>`, which is `Ok(())` if all tasks succeeded, or an error if any task failed.
#[instrument(
    level = "info",
    skip(subscriber_result, state_manager_result, server_result)
)]
fn handle_tasks_results(
    subscriber_result: Result<()>,
    state_manager_result: Result<()>,
    server_result: Result<()>,
) -> Result<()> {
    let result_handler = |result: Result<()>, message: &str| {
        if let Err(e) = result {
            error!(
                target = "atoma-node-service",
                event = "atoma_node_service_shutdown",
                error = ?e,
                "{message}"
            );
            return Err(e);
        }
        Ok(())
    };
    result_handler(subscriber_result, "Subscriber terminated abruptly")?;
    result_handler(state_manager_result, "State manager terminated abruptly")?;
    result_handler(server_result, "Server terminated abruptly")?;
    Ok(())
}
