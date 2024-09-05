use atoma_types::{AtomaStreamingData, Request, Response};
use candle::Error as CandleError;
use futures::StreamExt;
use std::fmt::Debug;
use std::{io, path::PathBuf, time::Instant};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::oneshot;
use tracing::{error, info, instrument};

use thiserror::Error;

use crate::{
    model_thread::{ModelThreadDispatcher, ModelThreadError, ModelThreadHandle},
    models::config::ModelsConfig,
};

/// `ModelService` - Responsible for listening to new AI inference requests, potentially
/// to different hosted AI large language models.
pub struct ModelService {
    /// Vector containing each model's thread join handle.
    model_thread_handle: Vec<ModelThreadHandle>,
    /// A model thread dispatcher.
    dispatcher: ModelThreadDispatcher,
    /// Start time of the `ModelService`.
    start_time: Instant,
    /// Boolean parameter that specifies if the service should flush all
    /// stored AI models, on shutdown.
    flush_storage: bool,
    /// The model weights, tokenizer and configuration data storage path.
    cache_dir: PathBuf,
    /// A `mpsc` end `Receiver`, listening to new requests, from the node's
    /// JRPC service.
    json_server_req_rx: Receiver<(Request, oneshot::Sender<Response>)>,
    /// A `mpsc` end `Receiver`, listening to new requests, from the node's
    /// event listener service (requests coming from the Atoma's smart contract).
    subscriber_req_rx: Receiver<Request>,
    /// Atoma's node response sender. Responsible for sending the generated output to
    /// different the Atoma's client service (for on-chain submission of the
    /// cryptographic commitment to the output).
    atoma_node_resp_tx: Sender<Response>,
}

impl ModelService {
    /// Starts a new instance of a `ModelService`.
    ///
    /// It includes starting a new `ModelThread` for the `Model` being hold.
    pub fn start(
        model_config: ModelsConfig,
        json_server_req_rx: Receiver<(Request, oneshot::Sender<Response>)>,
        subscriber_req_rx: Receiver<Request>,
        atoma_node_resp_tx: Sender<Response>,
        stream_tx: Sender<AtomaStreamingData>,
    ) -> Result<Self, ModelServiceError> {
        let flush_storage = model_config.flush_storage();
        let cache_dir = model_config.cache_dir();

        let (dispatcher, model_thread_handle) =
            ModelThreadDispatcher::start(model_config, stream_tx)?;
        let start_time = Instant::now();

        Ok(Self {
            dispatcher,
            model_thread_handle,
            start_time,
            flush_storage,
            cache_dir,
            json_server_req_rx,
            subscriber_req_rx,
            atoma_node_resp_tx,
        })
    }

    /// Main loop for `ModelService`.
    ///
    /// Listens to requests coming from either the node's JRPC service, or the
    /// node's blockchain event listener. It also processes newly processed responses
    /// containing the AI generated output (for a given request).
    #[instrument(skip_all)]
    pub async fn run(&mut self) -> Result<(), ModelServiceError> {
        loop {
            tokio::select! {
                Some(request) = self.json_server_req_rx.recv() => {
                    info!("Received a new request, with id = {:?}", request.0.id());
                    self.dispatcher.run_json_inference(request);
                },
                Some(request) = self.subscriber_req_rx.recv() => {
                    self.dispatcher.run_subscriber_inference(request);
                    let counter = metrics::counter!("atoma-inference-service-request");
                    counter.increment(1);
                },
                Some(resp) = self.dispatcher.responses.next() => match resp {
                    Ok(response) => {
                        info!("Received a new inference response: {:?}", response);
                        if let Err(e) = self.atoma_node_resp_tx.send(response).await {
                            return Err(ModelServiceError::SendError(e.to_string()));
                        }
                    },
                    Err(e) => {
                        error!("Found error in generating inference response: {}", e);
                    }
                },
                else => continue,
            }
        }
    }
}

impl ModelService {
    /// Stops the `ModelService`
    #[instrument(skip_all)]
    pub async fn stop(mut self) {
        info!(
            "Stopping Inference Service, running time: {:?}",
            self.start_time.elapsed()
        );

        if self.flush_storage {
            match std::fs::remove_dir(self.cache_dir) {
                Ok(()) => {}
                Err(e) => error!("Failed to remove storage folder, on shutdown: {e}"),
            };
        }

        let _ = self
            .model_thread_handle
            .drain(..)
            .map(|h| h.stop())
            .collect::<Vec<_>>();
    }
}

#[derive(Debug, Error)]
pub enum ModelServiceError {
    #[error("Failed to run inference: `{0}`")]
    FailedInference(Box<dyn std::error::Error + Send + Sync>),
    #[error("Failed to fecth model: `{0}`")]
    FailedModelFetch(String),
    #[error("Failed to generate private key: `{0}`")]
    PrivateKeyError(io::Error),
    #[error("Core error: `{0}`")]
    ModelThreadError(#[from] ModelThreadError),
    #[error("Candle error: `{0}`")]
    CandleError(#[from] CandleError),
    #[error("Sender error: `{0}`")]
    SendError(String),
}

#[cfg(test)]
mod tests {
    use atoma_types::{Digest, ModelParams};
    use serde::Serialize;
    use std::io::Write;
    use toml::{toml, Value};

    use crate::models::{config::ModelConfig, types::LlmOutput, ModelError, ModelTrait};

    use super::*;

    #[derive(Clone)]
    struct TestModelInstance {}

    struct MockInput {}

    #[derive(Serialize)]
    struct MockOutput {}

    impl LlmOutput for MockOutput {
        fn num_input_tokens(&self) -> usize {
            0
        }

        fn num_output_tokens(&self) -> Option<usize> {
            None
        }

        fn time_to_generate(&self) -> f64 {
            0.0
        }
        fn tokens(&self) -> Vec<u32> {
            vec![]
        }
    }

    impl TryFrom<(Digest, ModelParams)> for MockInput {
        type Error = ModelError;

        fn try_from(_: (Digest, ModelParams)) -> Result<Self, Self::Error> {
            Ok(Self {})
        }
    }

    impl ModelTrait for TestModelInstance {
        type Input = MockInput;
        type Output = MockOutput;
        type LoadData = ();

        fn fetch(_: String, _: PathBuf, _: ModelConfig) -> Result<(), crate::models::ModelError> {
            Ok(())
        }

        fn load(
            _: Self::LoadData,
            _: tokio::sync::mpsc::Sender<AtomaStreamingData>,
        ) -> Result<Self, crate::models::ModelError> {
            Ok(Self {})
        }

        fn model_type(&self) -> crate::models::types::ModelType {
            crate::models::types::ModelType::LlamaV1
        }

        fn run(&mut self, _: Self::Input) -> Result<Self::Output, crate::models::ModelError> {
            Ok(MockOutput {})
        }
    }

    #[tokio::test]
    async fn test_inference_service_initialization() {
        const CONFIG_FILE_PATH: &str = "./inference.toml";

        let config_data = Value::Table(toml! {
            api_key = "your_api_key"
            cache_dir = "./cache_dir/"
            flush_storage = true
            models = [
            [
                0,
                "bf16",
                "mamba_370m",
                "",
                false
            ]]
            tracing = true
            jrpc_port = 3000
        });
        let toml_string =
            toml::to_string_pretty(&config_data).expect("Failed to serialize to TOML");

        let mut file = std::fs::File::create(CONFIG_FILE_PATH).expect("Failed to create file");
        file.write_all(toml_string.as_bytes())
            .expect("Failed to write to file");

        let (_, json_server_req_rx) = tokio::sync::mpsc::channel(1);
        let (_, subscriber_req_rx) = tokio::sync::mpsc::channel(1);
        let (atoma_node_resp_tx, _) = tokio::sync::mpsc::channel(1);
        let (stream_tx, _) = tokio::sync::mpsc::channel(1);

        let config = ModelsConfig::from_file_path(CONFIG_FILE_PATH);

        let _ = ModelService::start(
            config,
            json_server_req_rx,
            subscriber_req_rx,
            atoma_node_resp_tx,
            stream_tx,
        )
        .unwrap();

        std::fs::remove_file(CONFIG_FILE_PATH).unwrap();
    }
}
