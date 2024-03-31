// use hf_hub::api::sync::Api;
// use inference::service::ModelService;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // let (_, receiver) = tokio::sync::mpsc::channel(32);

    // let _ = ModelService::start::<Model, Api>(
    //     "../inference.toml".parse().unwrap(),
    //     "../private_key".parse().unwrap(),
    //     receiver,
    // )
    // .expect("Failed to start inference service");
}
