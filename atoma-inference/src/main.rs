use std::time::Duration;

use ed25519_consensus::SigningKey as PrivateKey;
use inference::{
    models::{
        candle::stable_diffusion::StableDiffusion,
        config::ModelsConfig,
        types::{ModelType, StableDiffusionRequest, StableDiffusionResponse},
    },
    service::{ModelService, ModelServiceError},
};

#[tokio::main]
async fn main() -> Result<(), ModelServiceError> {
    tracing_subscriber::fmt::init();

    let (req_sender, req_receiver) = tokio::sync::mpsc::channel::<StableDiffusionRequest>(32);
    let (resp_sender, mut resp_receiver) =
        tokio::sync::mpsc::channel::<StableDiffusionResponse>(32);

    let model_config = ModelsConfig::from_file_path("../inference.toml".parse().unwrap());
    let private_key_bytes =
        std::fs::read("../private_key").map_err(ModelServiceError::PrivateKeyError)?;
    let private_key_bytes: [u8; 32] = private_key_bytes
        .try_into()
        .expect("Incorrect private key bytes length");

    let private_key = PrivateKey::from(private_key_bytes);
    let mut service: ModelService<StableDiffusionRequest, StableDiffusionResponse> =
        ModelService::start::<StableDiffusion>(
            model_config,
            private_key,
            req_receiver,
            resp_sender,
        )
        .expect("Failed to start inference service");

    let pk = service.public_key();

    tokio::spawn(async move {
        service.run().await?;
        Ok::<(), ModelServiceError>(())
    });

    tokio::time::sleep(Duration::from_millis(5_000)).await;

    // req_sender
    //     .send(TextRequest {
    //         request_id: 0,
    //         prompt: "Leon, the professional is a movie".to_string(),
    //         model: "falcon_7b".to_string(),
    //         max_tokens: 512,
    //         temperature: Some(0.0),
    //         random_seed: 42,
    //         repeat_last_n: 64,
    //         repeat_penalty: 1.1,
    //         sampled_nodes: vec![pk],
    //         top_p: Some(1.0),
    //         top_k: 10,
    //     })
    //     .await
    //     .expect("Failed to send request");

    req_sender
        .send(StableDiffusionRequest {
            request_id: 0,
            prompt: "A depiction of Natalie Portman".to_string(),
            uncond_prompt: "".to_string(),
            height: None,
            width: None,
            num_samples: 1,
            n_steps: None,
            model_type: ModelType::StableDiffusionV1_5,
            guidance_scale: None,
            img2img: None,
            img2img_strength: 0.8,
            random_seed: Some(42),
            sampled_nodes: vec![pk],
        })
        .await
        .expect("Failed to send request");

    if let Some(response) = resp_receiver.recv().await {
        println!("Got a response: {:?}", response);
    }

    Ok(())
}
