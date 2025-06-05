use std::{path::PathBuf, str::FromStr, sync::Arc};

use anyhow::{Context, Result};
use atoma_confidential::AtomaConfidentialCompute;
use atoma_daemon::{telemetry, AtomaDaemonConfig, DaemonState};
use atoma_p2p::{AtomaP2pNode, AtomaP2pNodeConfig};
use atoma_service::{
    config::AtomaServiceConfig, handlers::request_counter::RequestCounter, server::AppState,
};
use atoma_state::{config::AtomaStateManagerConfig, AtomaState, AtomaStateManager};
use atoma_sui::{client::Client, config::Config, subscriber::Subscriber};
use atoma_utils::spawn_with_shutdown;
use clap::Parser;
use dashmap::{DashMap, DashSet};
use futures::future::try_join_all;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use sui_keys::keystore::FileBasedKeystore;
use sui_sdk::{types::base_types::ObjectID, wallet_context::WalletContext};
use tokenizers::Tokenizer;
use tokio::{
    net::TcpListener,
    sync::{watch, RwLock},
    try_join,
};
use tracing::{error, info, instrument, warn};

/// The name of the environment variable for the Hugging Face token
const HF_TOKEN: &str = "HF_TOKEN";

/// Command line arguments for the Atoma node
#[derive(Parser)]
struct Args {
    /// Index of the address to use from the keystore
    #[arg(short, long)]
    address_index: Option<usize>,

    /// Path to the configuration file
    #[arg(short, long)]
    config_path: String,
}

/// Configuration for the Atoma node.
///
/// This struct holds the configuration settings for various components
/// of the Atoma node, including the Sui, service, and state manager configurations.
#[derive(Debug)]
struct NodeConfig {
    /// Configuration for the Sui component.
    sui: Config,

    /// Configuration for the p2p component.
    p2p: AtomaP2pNodeConfig,

    /// Configuration for the service component.
    service: AtomaServiceConfig,

    /// Configuration for the state manager component.
    state: AtomaStateManagerConfig,

    /// Configuration for the daemon component.
    daemon: AtomaDaemonConfig,
}

impl NodeConfig {
    fn load(path: &str) -> Self {
        let sui = Config::from_file_path(path);
        let p2p = AtomaP2pNodeConfig::from_file_path(path);
        let service = AtomaServiceConfig::from_file_path(path);
        let state = AtomaStateManagerConfig::from_file_path(path);
        let daemon = AtomaDaemonConfig::from_file_path(path);

        Self {
            sui,
            p2p,
            service,
            state,
            daemon,
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

                let tokenizer_filename = repo.get("tokenizer.json").unwrap_or_else(|e| {
                    panic!("Failed to get tokenizer.json for model {model}, with error: {e}");
                });

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
#[allow(clippy::too_many_lines)]
#[allow(clippy::redundant_pub_crate)]
async fn main() -> Result<()> {
    let _log_guards = telemetry::setup_logging().context("Failed to setup logging")?;

    dotenvy::dotenv().ok();

    let args = Args::parse();
    let config = NodeConfig::load(&args.config_path);

    info!("Starting Atoma node service");

    // Initialize Sentry only if DSN is provided
    let _guard = config.service.sentry_dsn.map_or_else(
        || {
            info!("No Sentry DSN provided, skipping Sentry initialization");
            None
        },
        |dsn| {
            info!("Initializing Sentry with provided DSN");
            Some(sentry::init((
                dsn,
                sentry::ClientOptions {
                    release: sentry::release_name!(),
                    // Capture user IPs and potentially sensitive headers when using HTTP server integrations
                    // see https://docs.sentry.io/platforms/rust/data-management/data-collected for more info
                    send_default_pii: false,
                    environment: Some(std::borrow::Cow::Owned(
                        config
                            .service
                            .environment
                            .clone()
                            .unwrap_or_else(|| "development".to_string()),
                    )),
                    traces_sample_rate: 0.2,
                    ..Default::default()
                },
            )))
        },
    );

    let (shutdown_sender, mut shutdown_receiver) = watch::channel(false);
    let (event_subscriber_sender, event_subscriber_receiver) = flume::unbounded();
    let (state_manager_sender, state_manager_receiver) = flume::unbounded();
    let (p2p_event_sender, p2p_event_receiver) = flume::unbounded();

    // Start the heartbeat service
    start_heartbeat_service(
        shutdown_receiver.clone(),
        config.service.heartbeat_url.clone(),
    );

    info!(
        target = "atoma-node-service",
        event = "keystore_path",
        keystore_path = config.sui.sui_keystore_path(),
        "Starting with Sui's keystore instance"
    );

    let keystore = FileBasedKeystore::new(&config.sui.sui_keystore_path().into())
        .context("Failed to initialize keystore")?;
    let mut wallet_ctx = WalletContext::new(&PathBuf::from(config.sui.sui_config_path()))?;
    if let Some(request_timeout) = config.sui.request_timeout() {
        wallet_ctx = wallet_ctx.with_request_timeout(request_timeout);
    }
    if let Some(max_concurrent_requests) = config.sui.max_concurrent_requests() {
        wallet_ctx = wallet_ctx.with_max_concurrent_requests(max_concurrent_requests);
    }
    let address = wallet_ctx.active_address()?;
    let address_index = args.address_index.unwrap_or_else(|| {
        wallet_ctx
            .get_addresses()
            .iter()
            .position(|a| a == &address)
            .unwrap()
    });

    info!(
        target = "atoma-node-service",
        event = "p2p_node_spawn",
        "Spawning Atoma's p2p node service"
    );
    let p2p_node_service_shutdown_receiver = shutdown_receiver.clone();
    let p2p_node_service_handle = spawn_with_shutdown(
        async move {
            let p2p_node =
                AtomaP2pNode::start(config.p2p, Arc::new(keystore), p2p_event_sender, false)?;
            let pinned_future = Box::pin(p2p_node.run(p2p_node_service_shutdown_receiver));
            pinned_future.await
        },
        shutdown_sender.clone(),
    );

    info!(
        target = "atoma-node-service",
        event = "state_manager_service_spawn",
        database_url = config.state.database_url,
        "Spawning state manager service"
    );

    let client = Arc::new(RwLock::new(
        Client::new_from_config(args.config_path).await?,
    ));
    let state_manager_shutdown_receiver = shutdown_receiver.clone();
    let database_url = config.state.database_url.clone();
    let client_clone = client.clone();
    let state_manager_handle = spawn_with_shutdown(
        async move {
            let state_manager = AtomaStateManager::new_from_url(
                &database_url,
                client_clone,
                event_subscriber_receiver,
                state_manager_receiver,
                p2p_event_receiver,
            )
            .await?;
            state_manager.run(state_manager_shutdown_receiver).await
        },
        shutdown_sender.clone(),
    );

    let (subscriber_confidential_compute_sender, subscriber_confidential_compute_receiver) =
        tokio::sync::mpsc::unbounded_channel();
    let (app_state_decryption_sender, app_state_decryption_receiver) =
        tokio::sync::mpsc::unbounded_channel();
    let (app_state_encryption_sender, app_state_encryption_receiver) =
        tokio::sync::mpsc::unbounded_channel();

    info!(
        target = "atoma-node-service",
        event = "confidential_compute_service_spawn",
        "Spawning confidential compute service"
    );

    let (compute_shared_secret_sender, compute_shared_secret_receiver) =
        tokio::sync::mpsc::unbounded_channel();

    let confidential_compute_service_handle = spawn_with_shutdown(
        AtomaConfidentialCompute::start_confidential_compute_service(
            client.clone(),
            subscriber_confidential_compute_receiver,
            app_state_decryption_receiver,
            app_state_encryption_receiver,
            compute_shared_secret_receiver,
            shutdown_receiver.clone(),
        ),
        shutdown_sender.clone(),
    );

    let (stack_retrieve_sender, stack_retrieve_receiver) = tokio::sync::mpsc::unbounded_channel();
    let package_id = config.sui.atoma_package_id();
    info!(
        target = "atoma-node-service",
        event = "subscriber_service_spawn",
        package_id = package_id.to_string(),
        "Spawning subscriber service"
    );

    let subscriber = Subscriber::new(
        config.sui.clone(),
        false,
        event_subscriber_sender,
        stack_retrieve_receiver,
        subscriber_confidential_compute_sender,
        shutdown_receiver.clone(),
    );

    info!(
        target = "atoma-node-service",
        event = "subscriber_service_spawn",
        package_id = package_id.to_string(),
        "Subscribing to Sui events"
    );
    let subscriber_handle = spawn_with_shutdown(
        async move {
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
            result
        },
        shutdown_sender.clone(),
    );

    let hf_token =
        std::env::var(HF_TOKEN).context(format!("Variable {HF_TOKEN} not set in the .env file"))?;
    let tokenizers =
        initialize_tokenizers(&config.service.models, &config.service.revisions, hf_token).await?;

    let keystore = FileBasedKeystore::new(&config.sui.sui_keystore_path().into())
        .context("Failed to initialize keystore")?;

    let app_state = AppState {
        concurrent_requests_per_stack: Arc::new(DashMap::new()),
        client_dropped_streamer_connections: Arc::new(DashSet::new()),
        state_manager_sender,
        stack_retrieve_sender,
        decryption_sender: app_state_decryption_sender,
        encryption_sender: app_state_encryption_sender,
        compute_shared_secret_sender,
        tokenizers: Arc::new(tokenizers),
        models: Arc::new(
            config
                .service
                .models
                .into_iter()
                .map(String::to_lowercase)
                .collect(),
        ),
        chat_completions_service_urls: config.service.chat_completions_service_urls,
        embeddings_service_url: config
            .service
            .embeddings_service_url
            .context("Embeddings service URL not configured")?,
        image_generations_service_url: config
            .service
            .image_generations_service_url
            .context("Image generations service URL not configured")?,
        keystore: Arc::new(keystore),
        address_index,
        whitelist_sui_addresses_for_fiat: config.service.whitelist_sui_addresses_for_fiat,
        too_many_requests: Arc::new(DashMap::new()),
        too_many_requests_timeout_ms: u128::from(config.service.too_many_requests_timeout_ms),
        running_num_requests: Arc::new(RequestCounter::new()),
        memory_lower_threshold: config.service.memory_lower_threshold,
        memory_upper_threshold: config.service.memory_upper_threshold,
        max_num_queued_requests: config.service.max_num_queued_requests,
    };

    let chat_completions_service_urls = app_state
        .chat_completions_service_urls
        .iter()
        .flat_map(|(model, urls)| {
            urls.iter()
                .map(|(url, job, max_number_of_running_requests)| {
                    (
                        model.clone(),
                        url.clone(),
                        job.clone(),
                        *max_number_of_running_requests,
                    )
                })
        })
        .collect();
    atoma_service::handlers::inference_service_metrics::start_metrics_updater(
        chat_completions_service_urls,
        config.service.metrics_update_interval,
    );

    let daemon_app_state = DaemonState {
        atoma_state: AtomaState::new_from_url(&config.state.database_url).await?,
        client,
        node_badges: config
            .daemon
            .node_badges
            .iter()
            .map(|(id, value)| (ObjectID::from_str(id).unwrap(), *value))
            .collect(),
    };

    let tcp_listener = TcpListener::bind(&config.service.service_bind_address)
        .await
        .context("Failed to bind TCP listener")?;
    let daemon_tcp_listener = TcpListener::bind(&config.daemon.service_bind_address)
        .await
        .context("Failed to bind daemon TCP listener")?;

    info!(
        target = "atoma-node-service",
        event = "atoma_node_service_spawn",
        bind_address = config.service.service_bind_address,
        "Starting Atoma node service"
    );

    let service_handle = spawn_with_shutdown(
        atoma_service::server::run_server(app_state, tcp_listener, shutdown_receiver.clone()),
        shutdown_sender.clone(),
    );

    info!(
        target = "atoma-daemon-service",
        event = "atoma_daemon_service_spawn",
        bind_address = config.daemon.service_bind_address,
        "Starting Atoma daemon service"
    );
    let daemon_handle = spawn_with_shutdown(
        atoma_daemon::server::run_server(
            daemon_app_state,
            daemon_tcp_listener,
            shutdown_receiver.clone(),
        ),
        shutdown_sender.clone(),
    );

    let ctrl_c = tokio::task::spawn(async move {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!(
                    target = "atoma-node-service",
                    event = "atoma-node-stop",
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

    // Wait for shutdown signal and handle cleanup
    let (
        subscriber_result,
        state_manager_result,
        server_result,
        daemon_result,
        p2p_node_service_result,
        confidential_compute_service_result,
        _,
    ) = try_join!(
        subscriber_handle,
        state_manager_handle,
        service_handle,
        daemon_handle,
        p2p_node_service_handle,
        confidential_compute_service_handle,
        ctrl_c
    )?;
    handle_tasks_results(
        subscriber_result,
        state_manager_result,
        server_result,
        daemon_result,
        p2p_node_service_result,
        confidential_compute_service_result,
    )?;

    info!(
        target = "atoma-node-service",
        event = "atoma_node_service_shutdown",
        "Atoma node service shut down successfully"
    );

    telemetry::shutdown();

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
    daemon_result: Result<()>,
    p2p_node_service_result: Result<()>,
    confidential_compute_service_result: Result<()>,
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
    result_handler(daemon_result, "Daemon terminated abruptly")?;
    result_handler(
        p2p_node_service_result,
        "P2P node service terminated abruptly",
    )?;
    result_handler(
        confidential_compute_service_result,
        "Confidential compute service terminated abruptly",
    )?;
    Ok(())
}

/// Starts a heartbeat service that pings a health check endpoint every minute.
///
/// This function spawns a background task that sends a GET request to a health check
/// service at regular intervals to indicate the daemon is still running.
///
/// # Arguments
/// * `shutdown_receiver` - A receiver that signals when the service should shut down
/// * `heartbeat_url` - The URL of the heartbeat service
#[allow(clippy::redundant_pub_crate)]
fn start_heartbeat_service(mut shutdown_receiver: watch::Receiver<bool>, heartbeat_url: String) {
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let interval = std::time::Duration::from_secs(60);

        tracing::info!(
            target = "atoma_daemon",
            event = "heartbeat-service-start",
            url = %heartbeat_url.clone(),
            interval_secs = %interval.as_secs(),
            "Starting heartbeat service"
        );

        loop {
            tokio::select! {
                () = tokio::time::sleep(interval) => {
                    // Send heartbeat ping
                    match client.get(heartbeat_url.clone()).send().await {
                        Ok(response) => {
                            if response.status().is_success() {
                                tracing::debug!(
                                    target = "atoma_daemon",
                                    event = "heartbeat-ping",
                                    status = %response.status(),
                                    "Sent heartbeat ping successfully"
                                );
                            } else {
                                tracing::warn!(
                                    target = "atoma_daemon",
                                    event = "heartbeat-ping-failed",
                                    status = %response.status(),
                                    "Heartbeat ping returned non-success status"
                                );
                            }
                        },
                        Err(e) => {
                            tracing::error!(
                                target = "atoma_daemon",
                                event = "heartbeat-ping-error",
                                error = %e,
                                "Failed to send heartbeat ping"
                            );
                        }
                    }
                }
                result = shutdown_receiver.changed() => {
                    if result.is_err() || *shutdown_receiver.borrow() {
                        tracing::info!(
                            target = "atoma_daemon",
                            event = "heartbeat-service-shutdown",
                            "Heartbeat service shutting down"
                        );
                        break;
                    }
                }
            }
        }
    });
}
