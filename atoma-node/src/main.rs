use atoma_node::{AtomaNode, AtomaNodeError};
use clap::Parser;
use tokio::sync::mpsc;

const CHANNEL_BUFFER: usize = 32;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    config_path: String,
}

#[tokio::main]
async fn main() -> Result<(), AtomaNodeError> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let config_path = args.config_path;

    let (_, json_rpc_server_rx) = mpsc::channel(CHANNEL_BUFFER);
    AtomaNode::start(config_path, json_rpc_server_rx).await?;

    Ok(())
}
