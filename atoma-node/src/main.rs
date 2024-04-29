use atoma_node::{AtomaNode, AtomaNodeError};
use clap::Parser;
use tokio::sync::mpsc;

const CHANNEL_BUFFER: usize = 32;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    atoma_sui_client_config_path: String,
    #[arg(long)]
    model_config_path: String,
    #[arg(long)]
    sui_subscriber_path: String,
}

#[tokio::main]
async fn main() -> Result<(), AtomaNodeError> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let atoma_sui_client_config_path = args.atoma_sui_client_config_path;
    let model_config_path = args.model_config_path;
    let sui_subscriber_path = args.sui_subscriber_path;

    let (_, json_rpc_server_rx) = mpsc::channel(CHANNEL_BUFFER);
    let _atoma_node = AtomaNode::start(
        atoma_sui_client_config_path,
        model_config_path,
        sui_subscriber_path,
        json_rpc_server_rx,
    )
    .await?;

    loop {}
}
