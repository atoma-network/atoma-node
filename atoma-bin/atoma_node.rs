use std::{path::Path, str::FromStr, sync::Arc};

use anyhow::{Context, Result};
use atoma_service::{
    config::AtomaServiceConfig,
    server::{run_server, AppState},
};
use atoma_state::config::StateManagerConfig;
use atoma_sui::{AtomaSuiConfig, SuiEventSubscriber};
use clap::Parser;
use futures::future::try_join_all;
use sqlx::SqlitePool;
use sui_keys::keystore::FileBasedKeystore;
use tokenizers::Tokenizer;
use tokio::{net::TcpListener, sync::watch};
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

/// Command line arguments for the Atoma node
#[derive(Parser)]
struct Args {
    /// Index of the address to use from the keystore
    #[arg(short, long, default_value_t = 0)]
    address_index: usize,

    /// Path to the configuration file
    #[arg(short, long)]
    config_path: String,

    /// Path to the keystore file containing account keys
    #[arg(short, long)]
    keystore_path: String,
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
    state: StateManagerConfig,
}

impl Config {
    async fn load(path: String) -> Self {
        Self {
            sui: AtomaSuiConfig::from_file_path(path.clone()),
            service: AtomaServiceConfig::from_file_path(path.clone()),
            state: StateManagerConfig::from_file_path(path),
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
) -> Result<Vec<Arc<Tokenizer>>> {
    let fetch_futures: Vec<_> = models
        .iter()
        .zip(revisions.iter())
        .map(|(model, revision)| async move {
            let url = format!(
                "https://huggingface.co/{}/blob/{}/tokenizer.json",
                model, revision
            );

            let tokenizer_json = reqwest::get(&url)
                .await
                .context(format!("Failed to fetch tokenizer from {}", url))?
                .text()
                .await
                .context("Failed to read tokenizer content")?;

            Tokenizer::from_str(&tokenizer_json)
                .map_err(|e| {
                    anyhow::anyhow!(format!(
                        "Failed to parse tokenizer for model {}, with error: {}",
                        model, e
                    ))
                })
                .map(Arc::new)
        })
        .collect();

    try_join_all(fetch_futures).await
}

/// Gracefully shuts down the Atoma node service by awaiting the completion of the subscriber task.
///
/// This function handles the shutdown process by waiting for the subscriber task to complete
/// and properly handling different shutdown scenarios:
/// - Normal completion: Returns the subscriber's result with added context
/// - Cancellation: Logs a warning and returns success
/// - Other errors: Returns the error with added context
///
/// # Arguments
///
/// * `subscriber_handle` - A `JoinHandle` for the Sui event subscriber task that needs to be
///   shut down gracefully
///
/// # Returns
///
/// Returns a `Result<()>` where:
/// - `Ok(())` indicates successful shutdown
/// - `Err(_)` indicates a failure during the shutdown process
///
/// # Examples
///
/// ```rust,ignore
/// use tokio;
/// use anyhow::Result;
///
/// #[tokio::main]
/// async fn example() -> Result<()> {
///     let subscriber_handle = tokio::spawn(async { Ok(()) });
///     shutdown(subscriber_handle).await?;
///     Ok(())
/// }
/// ```
#[instrument(level = "info", skip(subscriber_handle))]
async fn shutdown(subscriber_handle: tokio::task::JoinHandle<Result<()>>) -> Result<()> {
    match subscriber_handle.await {
        Ok(result) => result.context("Subscriber failed during shutdown"),
        Err(e) if e.is_cancelled() => {
            warn!("Subscriber was cancelled during shutdown");
            Ok(())
        }
        Err(e) => Err(e).context("Error while shutting down subscriber"),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_logging("./logs").context("Failed to setup logging")?;

    let args = Args::parse();
    let config = Config::load(args.config_path).await;

    info!("Starting Atoma node service");

    let (shutdown_sender, shutdown_receiver) = watch::channel(false);
    let subscriber = SuiEventSubscriber::new(config.sui, String::new(), shutdown_receiver.clone());

    info!("Subscribing to Sui events");
    let subscriber_handle = tokio::spawn(async move {
        info!("Running Sui event subscriber");
        let result = subscriber.run().await;
        info!("Sui event subscriber finished");
        result.map_err(|e| anyhow::anyhow!(e))
    });

    let keystore = FileBasedKeystore::new(&args.keystore_path.into())
        .context("Failed to initialize keystore")?;

    let sqlite_pool = SqlitePool::connect(&config.state.database_url)
        .await
        .context("Failed to connect to SQLite database")?;

    let tokenizers =
        initialize_tokenizers(&config.service.models, &config.service.revisions).await?;

    let app_state = AppState {
        state: sqlite_pool,
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

    info!("Starting server on {}", config.service.service_bind_address);

    let server_handle = tokio::spawn(run_server(app_state, tcp_listener, shutdown_sender));

    // Wait for shutdown signal and handle cleanup
    shutdown(subscriber_handle).await?;

    if let Err(e) = server_handle.await {
        error!("Server terminated with error: {:?}", e);
    }

    info!("Atoma node service shut down successfully");
    Ok(())
}

/// Configure logging with JSON formatting, file output, and console output
fn setup_logging<P: AsRef<Path>>(log_dir: P) -> Result<()> {
    // Set up file appender with rotation
    let file_appender = RollingFileAppender::new(Rotation::DAILY, log_dir, "atoma-node.log");

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
        .unwrap_or_else(|_| EnvFilter::new("info,atoma_service=debug"));

    // Combine layers with filter
    Registry::default()
        .with(env_filter)
        .with(console_layer)
        .with(file_layer)
        .init();

    Ok(())
}
