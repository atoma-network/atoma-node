use std::time::Duration;

use atoma_input_manager::AtomaInputManagerError;
use atoma_sui::subscriber::{SuiSubscriber, SuiSubscriberError};
use atoma_types::{InputSource, ModelInput};
use clap::Parser;
use sui_sdk::types::base_types::ObjectID;
use tokio::sync::oneshot;
use tracing::{error, info};

#[derive(Debug, Parser)]
struct Args {
    /// The Sui package id associated with the Atoma call contract
    #[arg(long)]
    pub package_id: String,
    /// HTTP node's address for Sui client
    #[arg(long, default_value = "https://fullnode.testnet.sui.io:443")]
    pub http_addr: String,
    /// RPC node's web socket address for Sui client
    #[arg(long, default_value = "wss://fullnode.testnet.sui.io:443")]
    pub ws_addr: String,
}

#[tokio::main]
async fn main() -> Result<(), SuiSubscriberError> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let package_id = ObjectID::from_hex_literal(&args.package_id)?;
    let http_url = args.http_addr;
    let ws_url = args.ws_addr;

    let (event_sender, mut event_receiver) = tokio::sync::mpsc::channel(32);
    let (input_manager_tx, mut input_manager_rx) = tokio::sync::mpsc::channel::<(
        InputSource,
        oneshot::Sender<Result<ModelInput, AtomaInputManagerError>>,
    )>(32);
    let (chat_request_sender, _) = tokio::sync::mpsc::channel(32);
    // Spawn a task to discard messages
    tokio::spawn(async move {
        while let Some((input_source, oneshot)) = input_manager_rx.recv().await {
            info!("Received input from source: {:?}", input_source);
            let data = match input_source {
                InputSource::Firebase { request_id } => request_id,
                InputSource::Ipfs { cid, format } => format!("{cid}.{format:?}"),
                InputSource::Raw { prompt } => prompt,
            };
            if let Err(err) = oneshot.send(Ok(ModelInput::Text(data))) {
                error!("Failed to send response: {:?}", err);
            }
        }
    });

    let event_subscriber = SuiSubscriber::new(
        1,
        &http_url,
        Some(&ws_url),
        package_id,
        event_sender,
        Some(Duration::from_secs(5 * 60)),
        input_manager_tx,
        chat_request_sender,
    )
    .await?;

    tokio::spawn(async move {
        info!("initializing subscribe");
        if let Err(err) = event_subscriber.subscribe().await {
            error!("Failed to subscribe: {:?}", err);
        }
        Ok::<_, SuiSubscriberError>(())
    });

    while let Some(event) = event_receiver.recv().await {
        info!("Processed a new event: {:?}", event);
    }

    Ok(())
}
