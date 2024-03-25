use hf_hub::api::tokio::Api;
use inference::service::InferenceService;

#[tokio::main]
async fn main() {
    let (_, request_receiver) = tokio::sync::mpsc::channel(32);

    let _ = InferenceService::start::<Api>(
        "/Users/jorgeantonio/dev/atoma-node/inference.toml"
            .parse()
            .unwrap(),
        "/Users/jorgeantonio/dev/atoma-node/atoma-inference/private_key"
            .parse()
            .unwrap(),
        request_receiver,
    )
    .await
    .expect("Failed to start inference service");
}
