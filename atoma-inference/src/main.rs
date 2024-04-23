use atoma_inference::{
    jrpc_server,
    models::config::ModelsConfig,
    service::{ModelService, ModelServiceError},
};

#[tokio::main]
async fn main() -> Result<(), ModelServiceError> {
    tracing_subscriber::fmt::init();

    let (req_sender, req_receiver) = tokio::sync::mpsc::channel(32);
    let (_, subscriber_req_rx) = tokio::sync::mpsc::channel(32);
    let (atoma_node_resp_tx, _) = tokio::sync::mpsc::channel(32);

    let model_config = ModelsConfig::from_file_path("../inference.toml");
    let jrpc_port = model_config.jrpc_port();

    let mut service = ModelService::start(
        model_config,
        req_receiver,
        subscriber_req_rx,
        atoma_node_resp_tx,
    )
    .expect("Failed to start inference service");

    tokio::spawn(async move {
        service.run().await?;
        Ok::<(), ModelServiceError>(())
    });

    jrpc_server::run(req_sender, jrpc_port).await;

    Ok(())
}
