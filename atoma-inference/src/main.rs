use ed25519_consensus::SigningKey as PrivateKey;
use inference::{
    jrpc_server,
    models::config::ModelsConfig,
    service::{ModelService, ModelServiceError},
};

#[tokio::main]
async fn main() -> Result<(), ModelServiceError> {
    tracing_subscriber::fmt::init();

    let (req_sender, req_receiver) = tokio::sync::mpsc::channel(32);
    // let (resp_sender, mut resp_receiver) = tokio::sync::broadcast::channel(32);

    let model_config = ModelsConfig::from_file_path("../inference.toml".parse().unwrap());
    let private_key_bytes =
        std::fs::read("../private_key").map_err(ModelServiceError::PrivateKeyError)?;
    let private_key_bytes: [u8; 32] = private_key_bytes
        .try_into()
        .expect("Incorrect private key bytes length");

    let private_key = PrivateKey::from(private_key_bytes);
    let mut service = ModelService::start(model_config, private_key, req_receiver)
        .expect("Failed to start inference service");

    // let pk = service.public_key();

    tokio::spawn(async move {
        service.run().await?;
        Ok::<(), ModelServiceError>(())
    });
    dbg!("DEBUG");

    jrpc_server::run(req_sender).await;

    Ok(())
}
