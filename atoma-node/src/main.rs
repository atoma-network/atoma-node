use std::time::Duration;

use atoma_node::{AtomaNode, AtomaNodeError};
use clap::Parser;
use tokio::sync::mpsc;

const CHANNEL_BUFFER: usize = 32;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    model_config_path: String,
    #[arg(long)]
    private_key_path: String,
    #[arg(long)]
    sui_subscriber_path: String,
}

#[tokio::main]
async fn main() -> Result<(), AtomaNodeError> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let model_config_path = args.model_config_path;
    let private_key_path = args.private_key_path;
    let sui_subscriber_path = args.sui_subscriber_path;

    let (_, json_rpc_server_rx) = mpsc::channel(CHANNEL_BUFFER);
    tokio::spawn(async move {
        let _atoma_node = AtomaNode::start(
            model_config_path,
            private_key_path,
            sui_subscriber_path,
            json_rpc_server_rx,
        )
        .await?;
        Ok::<(), AtomaNodeError>(())
    });

    tokio::time::sleep(Duration::from_secs(20)).await;
    Ok(())
}
