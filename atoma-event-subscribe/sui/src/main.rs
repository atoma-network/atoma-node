use clap::Parser;
use sui::subscriber::{SuiSubscriber, SuiSubscriberError};
use sui_sdk::types::base_types::ObjectID;

#[derive(Debug, Parser)]
struct Args {
    /// The Sui package id associated with the Atoma call contract
    #[arg(long)]
    pub package_id: String,
    /// HTTP node's address for Sui client
    #[arg(long, default_value = "https://fullnode.devnet.sui.io:443")]
    pub http_addr: String,
    /// RPC node's web socket address for Sui client
    #[arg(long, default_value = "wss://fullnode.devnet.sui.io:443")]
    pub ws_socket_addr: String,
}

#[tokio::main]
async fn main() -> Result<(), SuiSubscriberError> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let package_id = ObjectID::from_hex_literal(&args.package_id)?;

    let http_url = args.http_addr;
    let ws_url = args.ws_socket_addr;
    let event_subscriber = SuiSubscriber::new(&http_url, Some(&ws_url), package_id).await?;
    event_subscriber.subscribe().await?;

    Ok(())
}
