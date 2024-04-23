use atoma_node::{AtomaNode, AtomaNodeError};
use clap::Parser;
use tokio::sync::mpsc;

const CHANNEL_BUFFER: usize = 32;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    node_id: u64,
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
    let node_id = args.node_id;
    let model_config_path = args.model_config_path;
    let private_key_path = args.private_key_path;
    let sui_subscriber_path = args.sui_subscriber_path;

    let (_, json_rpc_server_rx) = mpsc::channel(CHANNEL_BUFFER);
    let _atoma_node = AtomaNode::start(
        node_id,
        model_config_path,
        private_key_path,
        sui_subscriber_path,
        json_rpc_server_rx,
    )
    .await?;

    Ok(())
}
