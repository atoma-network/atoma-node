use clap::Parser;
use sui::subscriber::{SuiSubscriber, SuiSubscriberError};
use sui_sdk::types::base_types::ObjectID;

#[derive(Debug, Parser)]
struct Args {
    /// The Sui package id associated with the Atoma call contract
    #[arg(long)]
    pub package_id: String,
}

#[tokio::main]
async fn main() -> Result<(), SuiSubscriberError> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let package_id = ObjectID::from_hex_literal(&args.package_id)?;

    let devnet_http_url = "https://fullnode.devnet.sui.io:443";
    let devnet_ws_url = "wss://fullnode.devnet.sui.io:443";
    let event_subscriber = SuiSubscriber::new(devnet_http_url, Some(devnet_ws_url), package_id).await?;
    event_subscriber.subscribe().await?;

    Ok(())
}
