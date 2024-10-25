use std::{str::FromStr, sync::Arc};

use anyhow::{Context, Result};
use atoma_service::{
    config::AtomaServiceConfig,
    server::{run_server, AppState},
};
use atoma_state::config::StateManagerConfig;
use atoma_sui::{AtomaSuiConfig, SuiEventSubscriber};
use clap::Parser;
use sqlx::SqlitePool;
use sui_keys::keystore::FileBasedKeystore;
use tokenizers::Tokenizer;
use tokio::{net::TcpListener, sync::watch};
use tracing::{error, info};

/// Command line arguments for the Atoma node
#[derive(Parser)]
struct Args {
    /// Index of the address to use from the keystore
    #[arg(short, long)]
    address_index: Option<usize>,

    /// Path to the configuration file
    #[arg(short, long)]
    config_path: String,

    /// Path to the keystore file containing account keys
    #[arg(short, long)]
    keystore_path: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let sui_config = AtomaSuiConfig::from_file_path(args.config_path.clone());
    let service_config = AtomaServiceConfig::from_file_path(args.config_path.clone());
    let state_manager_config = StateManagerConfig::from_file_path(args.config_path);

    info!("Starting Atoma node service");

    let (shutdown_sender, mut shutdown_receiver) = watch::channel(false);
    let subscriber = SuiEventSubscriber::new(sui_config, "".to_string(), shutdown_receiver.clone());

    info!("Subscribing to Sui events");

    let subscriber_handle = tokio::spawn(async move {
        info!("Running Sui event subscriber");
        subscriber.run().await?;
        info!("Sui event subscriber finished");
        Ok::<(), anyhow::Error>(())
    });

    let keystore = FileBasedKeystore::new(&args.keystore_path.into())?;
    let sqlite_pool = SqlitePool::connect(&state_manager_config.database_url).await?;
    let models = service_config.models;
    let mut tokenizers = Vec::with_capacity(models.len());
    for (model, revision) in models.iter().zip(service_config.revisions.iter()) {
        let url = format!(
            "https://huggingface.co/{}/blob/{}/tokenizer.json",
            model, revision
        );
        let tokenizer_json = reqwest::get(url)
            .await
            .context("Failed to fetch tokenizer")?
            .text()
            .await
            .context("Failed to read tokenizer content")?;

        tokenizers.push(Arc::new(Tokenizer::from_str(&tokenizer_json).map_err(
            |e| {
                error!("Failed to load tokenizer: {}", e);
                anyhow::anyhow!(e)
            },
        )?));
    }
    let app_state = AppState {
        state: sqlite_pool,
        tokenizers: Arc::new(tokenizers),
        models: Arc::new(vec![]),
        inference_service_url: service_config.inference_service_url.unwrap(),
        keystore: Arc::new(keystore),
        address_index: args.address_index.unwrap_or(0),
    };
    let tcp_listener = TcpListener::bind(service_config.service_bind_address).await?;
    info!("Starting Atoma node service");
    run_server(app_state, tcp_listener, shutdown_sender).await?;

    if let Err(e) = shutdown_receiver.changed().await {
        error!("Failed to receive shutdown signal: {}", e);
        return Err(anyhow::anyhow!(e));
    }

    if let Err(e) = subscriber_handle.await {
        error!("Error while shutting down subscriber: {:?}", e);
    }

    info!("Shutting down Atoma node service");

    Ok(())
}
