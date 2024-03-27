use inference::models::Model;
use inference::service::InferenceService;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // let (_, receiver) = tokio::sync::mpsc::channel(32);

    // let _ = InferenceService::start::<Model>(
    //     "../inference.toml".parse().unwrap(),
    //     "../private_key".parse().unwrap(),
    //     receiver,
    // )
    // .expect("Failed to start inference service");

    // inference_service
    //     .run_inference(InferenceRequest {
    //         prompt: String::from("Which protocols are faster, zk-STARKs or zk-SNARKs ?"),
    //         max_tokens: 512,
    //         model: inference::models::ModelType::Llama2_7b,
    //         random_seed: 42,
    //         sampled_nodes: vec![],
    //         repeat_penalty: 1.0,
    //         temperature: Some(0.6),
    //         top_k: 10,
    //         top_p: None,
    //     })
    //     .await
    //     .unwrap();
}
