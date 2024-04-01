use hf_hub::api::sync::Api;
use inference::{
    models::candle::{
        mamba::MambaModel,
        types::{TextRequest, TextResponse},
    },
    service::{ModelService, ModelServiceError},
};

#[tokio::main]
async fn main() -> Result<(), ModelServiceError> {
    tracing_subscriber::fmt::init();

    let (req_sender, req_receiver) = tokio::sync::mpsc::channel::<TextRequest>(32);
    let (resp_sender, mut resp_receiver) = tokio::sync::mpsc::channel::<TextResponse>(32);

    let mut service = ModelService::start::<MambaModel, Api>(
        "../inference.toml".parse().unwrap(),
        "../private_key".parse().unwrap(),
        req_receiver,
        resp_sender,
    )
    .expect("Failed to start inference service");

    tokio::spawn(async move {
        service.run().await?;
        Ok::<(), ModelServiceError>(())
    });

    let sampled_nodes = vec![];

    req_sender
        .send(TextRequest {
            request_id: 0,
            prompt: "Natalie Portman".to_string(),
            model: "state-spaces/mamba-2.8b".to_string(),
            max_tokens: 512,
            temperature: Some(0.6),
            random_seed: 42,
            repeat_last_n: 15,
            repeat_penalty: 0.6,
            sampled_nodes,
            top_p: Some(1.0),
            top_k: 10,
        })
        .await
        .expect("Failed to send request");

    if let Some(response) = resp_receiver.recv().await {
        println!("Got a response: {:?}", response);
    }

    Ok(())
}
