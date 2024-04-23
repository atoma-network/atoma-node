use std::time::Duration;

use atoma_sui::subscriber::{SuiSubscriber, SuiSubscriberError};
use clap::Parser;
use sui_sdk::types::base_types::ObjectID;
use tracing::info;

#[derive(Debug, Parser)]
struct Args {
    /// The Sui package id associated with the Atoma call contract
    #[arg(long)]
    pub package_id: String,
    /// HTTP node's address for Sui client
    #[arg(long, default_value = "https://fullnode.devnet.sui.io:443")]
    pub http_addr: String,
    /// RPC node's web socket address for Sui client
    #[arg(long, default_value = "wss://rpc.devnet.sui.io:443")]
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

    let event_subscriber = SuiSubscriber::new(
        0,
        &http_url,
        Some(&ws_url),
        package_id,
        event_sender,
        Some(Duration::from_secs(5 * 60)),
    )
    .await?;

    tokio::spawn(async move {
        info!("initializing subscribe");
        event_subscriber.subscribe().await?;
        Ok::<_, SuiSubscriberError>(())
    });

    while let Some(event) = event_receiver.recv().await {
        info!("Processed a new event: {:?}", event);
    }

    Ok(())
}
