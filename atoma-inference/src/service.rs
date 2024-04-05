use candle::Error as CandleError;
use ed25519_consensus::{SigningKey as PrivateKey, VerificationKey as PublicKey};
use futures::StreamExt;
use std::{io, path::PathBuf, time::Instant};
use tokio::sync::mpsc::{Receiver, Sender};
use tracing::{error, info};

use thiserror::Error;

use crate::{
    apis::{ApiError, ApiTrait},
    model_thread::{ModelThreadDispatcher, ModelThreadError, ModelThreadHandle},
    models::{config::ModelsConfig, ModelTrait},
};

pub struct ModelService {
    model_thread_handle: Vec<ModelThreadHandle>,
    dispatcher: ModelThreadDispatcher,
    start_time: Instant,
    flush_storage: bool,
    public_key: PublicKey,
    storage_path: PathBuf,
    request_receiver: Receiver<serde_json::Value>,
    response_sender: Sender<serde_json::Value>,
}

impl ModelService {
    pub fn start<M, F>(
        model_config: ModelsConfig,
        private_key: PrivateKey,
        request_receiver: Receiver<serde_json::Value>,
        response_sender: Sender<serde_json::Value>,
    ) -> Result<Self, ModelServiceError>
    where
        M: ModelTrait + Send + 'static,
        F: ApiTrait + Send + Sync + 'static,
    {
        let public_key = private_key.verification_key();

        let flush_storage = model_config.flush_storage();
        let storage_path = model_config.storage_path();

        let (dispatcher, model_thread_handle) =
            ModelThreadDispatcher::start::<M, F>(model_config, public_key)
                .map_err(ModelServiceError::ModelThreadError)?;
        let start_time = Instant::now();

        Ok(Self {
            dispatcher,
            model_thread_handle,
            start_time,
            flush_storage,
            storage_path,
            public_key,
            request_receiver,
            response_sender,
        })
    }

    pub async fn run(&mut self) -> Result<serde_json::Value, ModelServiceError> {
        loop {
            tokio::select! {
                message = self.request_receiver.recv() => {
                    if let Some(request) = message {
                        self.dispatcher.run_inference(request);
                    }
                }
                response = self.dispatcher.responses.next() => {
                    if let Some(resp) = response {
                        match resp {
                            Ok(response) => {
                                info!("Received a new inference response: {:?}", response);
                                self.response_sender.send(response).await.map_err(|e| ModelServiceError::SendError(e.to_string()))?;
                            }
                            Err(e) => {
                                error!("Found error in generating inference response: {e}");
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn public_key(&self) -> PublicKey {
        self.public_key
    }
}

impl ModelService {
    pub async fn stop(mut self) {
        info!(
            "Stopping Inference Service, running time: {:?}",
            self.start_time.elapsed()
        );

        if self.flush_storage {
            match std::fs::remove_dir(self.storage_path) {
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
    ModelThreadError(ModelThreadError),
    #[error("Api error: `{0}`")]
    ApiError(ApiError),
    #[error("Candle error: `{0}`")]
    CandleError(CandleError),
    #[error("Sender error: `{0}`")]
    SendError(String),
}

impl From<ApiError> for ModelServiceError {
    fn from(error: ApiError) -> Self {
        Self::ApiError(error)
    }
}

impl From<CandleError> for ModelServiceError {
    fn from(error: CandleError) -> Self {
        Self::CandleError(error)
    }
}

#[cfg(test)]
mod tests {
    use ed25519_consensus::VerificationKey as PublicKey;
    use rand::rngs::OsRng;
    use std::io::Write;
    use toml::{toml, Value};

    use crate::{models::types::PrecisionBits, models::ModelId};

    use super::*;

    struct MockApi {}

    impl ApiTrait for MockApi {
        fn create(_: String, _: PathBuf) -> Result<Self, ApiError>
        where
            Self: Sized,
        {
            Ok(Self {})
        }

        fn fetch(&self, _: ModelId, _: String) -> Result<Vec<PathBuf>, ApiError> {
            Ok(vec![])
        }
    }

    impl Request for () {
        type ModelInput = ();

        fn into_model_input(self) -> Self::ModelInput {}

        fn is_node_authorized(&self, _: &PublicKey) -> bool {
            true
        }

        fn request_id(&self) -> usize {
            0
        }

        fn requested_model(&self) -> crate::models::ModelId {
            String::from("")
        }
    }

    impl Response for () {
        type ModelOutput = ();

        fn from_model_output(_: Self::ModelOutput) -> Self {}
    }

    #[derive(Clone)]
    struct TestModelInstance {}

    impl ModelTrait for TestModelInstance {
        type Input = ();
        type Output = ();
        type Fetch = ();

        fn fetch(_fetch: &Self::Fetch) -> Result<(), crate::models::ModelError> {
            Ok(())
        }

        fn load(
            _: Vec<PathBuf>,
            _: PrecisionBits,
            _device_id: usize,
        ) -> Result<Self, crate::models::ModelError> {
            Ok(Self {})
        }

        fn model_id(&self) -> crate::models::ModelId {
            String::from("")
        }

        fn run(&mut self, _: Self::Input) -> Result<Self::Output, crate::models::ModelError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_inference_service_initialization() {
        const CONFIG_FILE_PATH: &str = "./inference.toml";

        let private_key = PrivateKey::new(OsRng);

        let config_data = Value::Table(toml! {
            api_key = "your_api_key"
            models = [["Mamba3b", "F16", "", ""]]
            storage_path = "./storage_path/"
            tokenizer_file_path = "./tokenizer_file_path/"
            flush_storage = true
            tracing = true
        });
        let toml_string =
            toml::to_string_pretty(&config_data).expect("Failed to serialize to TOML");

        let mut file = std::fs::File::create(CONFIG_FILE_PATH).expect("Failed to create file");
        file.write_all(toml_string.as_bytes())
            .expect("Failed to write to file");

        let (_, req_receiver) = tokio::sync::mpsc::channel::<()>(1);
        let (resp_sender, _) = tokio::sync::mpsc::channel::<()>(1);

        let config = ModelsConfig::from_file_path(CONFIG_FILE_PATH.parse().unwrap());

        let _ = ModelService::<(), ()>::start::<TestModelInstance, MockApi>(
            config,
            private_key,
            req_receiver,
            resp_sender,
        )
        .unwrap();

        std::fs::remove_file(CONFIG_FILE_PATH).unwrap();
    }
}
