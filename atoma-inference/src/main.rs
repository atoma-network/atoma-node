use atoma_inference::{
    jrpc_server,
    models::config::ModelsConfig,
    service::{ModelService, ModelServiceError},
};
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    config_path: String,
}

#[tokio::main]
async fn main() -> Result<(), ModelServiceError> {
    tracing_subscriber::fmt::init();

    let (req_sender, req_receiver) = tokio::sync::mpsc::channel(32);
    let (_, subscriber_req_rx) = tokio::sync::mpsc::channel(32);
    let (atoma_node_resp_tx, _) = tokio::sync::mpsc::channel(32);
    let (stream_tx, _) = tokio::sync::mpsc::channel(32);
    let (_, chat_service_receiver) = tokio::sync::mpsc::channel(32);

    let args = Args::parse();
    let config_path = args.config_path;

    let model_config = ModelsConfig::from_file_path(config_path);
    let jrpc_port = model_config.jrpc_port();

    let mut service = ModelService::start(
        model_config,
        req_receiver,
        subscriber_req_rx,
        atoma_node_resp_tx,
        stream_tx,
        chat_service_receiver,
    )
    .expect("Failed to start inference service");

    tokio::spawn(async move {
        service.run().await?;
        Ok::<(), ModelServiceError>(())
    });

    jrpc_server::run(req_sender, jrpc_port).await;

    Ok(())
}
