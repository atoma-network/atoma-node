use crate::models::{config::ModelConfig, types::ModelType, ModelError, ModelTrait};
use ed25519_consensus::SigningKey as PrivateKey;
use std::{path::PathBuf, time::Duration};

mod prompts;
use prompts::PROMPTS;

use std::{collections::HashMap, sync::mpsc};

use futures::{stream::FuturesUnordered, StreamExt};
use rand::rngs::OsRng;
use reqwest::Client;
use serde_json::json;
use serde_json::Value;
use tokio::sync::oneshot;

use crate::{
    jrpc_server,
    model_thread::{spawn_model_thread, ModelThreadCommand, ModelThreadDispatcher},
    models::config::ModelsConfig,
    service::ModelService,
};

struct TestModel {
    duration: Duration,
}

impl ModelTrait for TestModel {
    type Input = Value;
    type Output = Value;
    type LoadData = Duration;

    fn fetch(
        duration: String,
        _cache_dir: PathBuf,
        _config: ModelConfig,
    ) -> Result<Self::LoadData, ModelError> {
        Ok(Duration::from_secs(duration.parse().unwrap()))
    }

    fn load(duration: Self::LoadData) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        Ok(Self { duration })
    }

    fn model_type(&self) -> ModelType {
        todo!()
    }

    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        std::thread::sleep(self.duration);
        println!(
            "Finished waiting time for {:?} and input = {}",
            self.duration, input
        );
        Ok(input)
    }
}

impl ModelThreadDispatcher {
    fn test_start() -> Self {
        let duration_in_secs = vec![1, 2, 5, 10];
        let mut model_senders = HashMap::with_capacity(4);

        for i in duration_in_secs {
            let model_name = format!("test_model_{:?}", i);

            let (model_sender, model_receiver) = mpsc::channel::<ModelThreadCommand>();
            model_senders.insert(model_name.clone(), model_sender.clone());

            let duration = format!("{i}");
            let cache_dir = "./".parse().unwrap();
            let model_config =
                ModelConfig::new(model_name.clone(), "".to_string(), "".to_string(), 0, false);

            let private_key = PrivateKey::new(OsRng);
            let public_key = private_key.verification_key();

            let _join_handle = spawn_model_thread::<TestModel>(
                model_name,
                duration,
                cache_dir,
                model_config,
                public_key,
                model_receiver,
            );
        }
        Self { model_senders }
    }
}

#[tokio::test]
async fn test_mock_model_thread() {
    const NUM_REQUESTS: usize = 16;

    let model_thread_dispatcher = ModelThreadDispatcher::test_start();
    let mut responses = FuturesUnordered::new();

    let mut should_be_received_responses = vec![];
    for i in 0..NUM_REQUESTS {
        for sender in model_thread_dispatcher.model_senders.values() {
            let (response_sender, response_receiver) = oneshot::channel();
            let request = json!(i);
            let command = ModelThreadCommand {
                request: request.clone(),
                response_sender,
            };
            sender.send(command).expect("Failed to send command");
            responses.push(response_receiver);
            should_be_received_responses.push(request.as_u64().unwrap());
        }
    }

    let mut received_responses = vec![];
    while let Some(response) = responses.next().await {
        if let Ok(value) = response {
            received_responses.push(value.as_u64().unwrap());
        }
    }

    received_responses.sort();
    should_be_received_responses.sort();

    assert_eq!(received_responses, should_be_received_responses);
}

#[tokio::test]
async fn test_inference_service() {
    const CHANNEL_BUFFER: usize = 32;
    const JRPC_PORT: u64 = 3000;

    let private_key = PrivateKey::new(OsRng);
    let model_ids = ["mamba_130m", "mamba_370m", "llama_tiny_llama_1_1b_chat"];
    let model_configs = vec![
        ModelConfig::new(
            "mamba_130m".to_string(),
            "f32".to_string(),
            "refs/pr/1".to_string(),
            0,
            false,
        ),
        ModelConfig::new(
            "mamba_370m".to_string(),
            "f32".to_string(),
            "refs/pr/1".to_string(),
            0,
            false,
        ),
        ModelConfig::new(
            "llama_tiny_llama_1_1b_chat".to_string(),
            "f32".to_string(),
            "main".to_string(),
            0,
            false,
        ),
    ];
    let config = ModelsConfig::new(
        "".to_string(),
        "./cache_dir".parse().unwrap(),
        true,
        model_configs,
        true,
        JRPC_PORT,
    );

    let (req_sender, req_receiver) = tokio::sync::mpsc::channel(CHANNEL_BUFFER);

    println!("Starting model service..");
    let mut service =
        ModelService::start(config.clone(), private_key.clone(), req_receiver).unwrap();
    
    let _service_join_handle = tokio::spawn(async move {
        service.run().await.expect("Failed to run service");
    });
    let _jrpc_server_join_handle =
        tokio::spawn(async move { jrpc_server::run(req_sender.clone(), JRPC_PORT).await });

    let client = Client::new();

    std::thread::sleep(Duration::from_secs(100));

    let mut responses = vec![];
    for (idx, prompt) in PROMPTS.iter().enumerate() {
        let model_id = model_ids[idx % 3];
        println!("model_id = {model_id}");
        let request = json!({
            "request_id": idx,
            "prompt": prompt.to_string(),
            "model": model_id.to_string(),
            "sampled_nodes": private_key.verification_key(),
            "temperature": 0.5,
            "random_seed": 42,
            "repeat_penalty": 1.0,
            "repeat_last_n": 64,
            "max_tokens": 32,
            "_top_k": 10,
            "top_p": 1.0
        });

        let request = json!({
            "jsonrpc": "2.0",
            "request": request,
            "id": idx
        });

        let response = client
            .post(format!("http://localhost:{}/", JRPC_PORT))
            .json(&request)
            .send()
            .await
            .expect("Failed to receive response from JRPCs server");

        let response_json: Value = response
            .json()
            .await
            .expect("Failed to parse response to JSON");
        println!("{}", response_json);
        responses.push(response_json);
    }
    assert_eq!(responses.len(), PROMPTS.len());
}
