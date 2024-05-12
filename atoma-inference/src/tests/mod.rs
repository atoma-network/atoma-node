use crate::models::{config::ModelConfig, types::ModelType, ModelError, ModelTrait};
use std::{path::PathBuf, time::Duration};

mod prompts;
use atoma_types::Text2TextPromptParams;
use prompts::PROMPTS;
use serde::Serialize;

use std::{collections::HashMap, sync::mpsc};

use atoma_types::{PromptParams, Request};
use futures::{stream::FuturesUnordered, StreamExt};
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

#[derive(Debug, Serialize)]
struct MockInputOutput {
    id: u64,
}

impl TryFrom<PromptParams> for MockInputOutput {
    type Error = ModelError;

    fn try_from(value: PromptParams) -> Result<Self, Self::Error> {
        Ok(Self {
            id: value.into_text2text_prompt_params().unwrap().max_tokens(),
        })
    }
}

impl ModelTrait for TestModel {
    type Input = MockInputOutput;
    type Output = MockInputOutput;
    type LoadData = Duration;

    fn fetch(
        duration: String,
        _cache_dir: PathBuf,
        _config: ModelConfig,
    ) -> Result<Self::LoadData, ModelError> {
        Ok(Duration::from_secs(duration.parse().unwrap()))
    }

    fn load(
        duration: Self::LoadData,
        _: std::sync::mpsc::Sender<String>,
    ) -> Result<Self, ModelError>
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
            "Finished waiting time for {:?} and input = {:?}",
            self.duration, input
        );
        Ok(input)
    }
}

impl ModelThreadDispatcher {
    fn test_start(stream_tx: std::sync::mpsc::Sender<String>) -> Self {
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

            let _join_handle = spawn_model_thread::<TestModel>(
                model_name,
                duration,
                cache_dir,
                model_config,
                model_receiver,
                stream_tx,
            );
        }
        Self {
            model_senders,
            responses: FuturesUnordered::new(),
        }
    }
}

#[tokio::test]
async fn test_mock_model_thread() {
    const NUM_REQUESTS: usize = 16;

    let (stream_tx, _) = std::sync::mpsc::channel::<String>();
    let model_thread_dispatcher = ModelThreadDispatcher::test_start(stream_tx);
    let mut responses = FuturesUnordered::new();

    let mut should_be_received_responses = vec![];
    for i in 0..NUM_REQUESTS {
        for sender in model_thread_dispatcher.model_senders.values() {
            let (response_sender, response_receiver) = oneshot::channel();
            let max_tokens = i as u64;
            let prompt_params = PromptParams::Text2TextPromptParams(Text2TextPromptParams::new(
                "".to_string(),
                "".to_string(),
                0.0,
                1,
                1.0,
                0,
                max_tokens,
                Some(0),
                Some(1.0),
            ));
            let request = Request::new(vec![0], 0, 1, prompt_params);
            let command = ModelThreadCommand {
                request: request.clone(),
                sender: response_sender,
            };
            sender.send(command).expect("Failed to send command");
            responses.push(response_receiver);
            should_be_received_responses
                .push(MockInputOutput::try_from(request.params()).unwrap().id);
        }
    }

    let mut received_responses = vec![];
    while let Some(response) = responses.next().await {
        if let Ok(value) = response {
            received_responses.push(value.response().as_u64().unwrap());
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

    let (json_server_req_sender, json_server_req_receiver) =
        tokio::sync::mpsc::channel(CHANNEL_BUFFER);
    let (_, subscriber_req_rx) = tokio::sync::mpsc::channel(CHANNEL_BUFFER);
    let (atoma_node_resp_tx, _) = tokio::sync::mpsc::channel(CHANNEL_BUFFER);
    let (stream_tx, _) = std::sync::mpsc::channel();

    println!("Starting model service..");
    let mut service = ModelService::start(
        config.clone(),
        json_server_req_receiver,
        subscriber_req_rx,
        atoma_node_resp_tx,
        stream_tx,
    )
    .unwrap();

    let _service_join_handle = tokio::spawn(async move {
        service.run().await.expect("Failed to run service");
    });
    let _jrpc_server_join_handle = tokio::spawn(async move {
        jrpc_server::run(json_server_req_sender.clone(), JRPC_PORT).await;
    });

    let client = Client::new();

    let mut responses = vec![];
    for (idx, prompt) in PROMPTS.iter().enumerate() {
        let model_id = model_ids[idx % 3];
        println!("model_id = {model_id}");
        let request = json!({
            "request_id": idx,
            "prompt": prompt.to_string(),
            "model": model_id.to_string(),
            "sampled_nodes": Vec::<Vec<u8>>::new(),
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

        println!("Sending new request to client..");
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
