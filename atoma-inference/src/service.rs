use candle::Device;
use candle_transformers::quantized_var_builder::VarBuilder;
use ed25519_consensus::SigningKey as PrivateKey;
use hf_hub::api::sync::Api;
use std::{io, path::PathBuf, time::Instant};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{error::SendError, Receiver};
use tracing::info;

use thiserror::Error;

use crate::{
    apis::{ApiError, ApiTrait},
    config::InferenceConfig,
    model_thread::{ModelThreadDispatcher, ModelThreadError, ModelThreadHandle},
    types::{InferenceRequest, InferenceResponse},
};

pub struct InferenceService {
    model_thread_handle: Vec<ModelThreadHandle>,
    dispatcher: ModelThreadDispatcher,
    start_time: Instant,
    _request_receiver: Receiver<InferenceRequest>,
}

impl InferenceService {
    pub fn start<T: ApiTrait + Clone + Send + 'static>(
        config_file_path: PathBuf,
        private_key_path: PathBuf,
        _request_receiver: Receiver<InferenceRequest>,
    ) -> Result<Self, InferenceServiceError> {
        let private_key_bytes =
            std::fs::read(&private_key_path).map_err(InferenceServiceError::PrivateKeyError)?;
        let private_key_bytes: [u8; 32] = private_key_bytes
            .try_into()
            .expect("Incorrect private key bytes length");

        let private_key = PrivateKey::from(private_key_bytes);
        let public_key = private_key.verification_key();
        let inference_config = InferenceConfig::from_file_path(config_file_path);
        let api_key = inference_config.api_key();
        let storage_folder = inference_config.storage_folder();
        let models = inference_config.models();

        let api = Api::create(api_key, storage_folder)?;

        let mut handles = Vec::with_capacity(models.len());
        for model in models.iter() {
            let api = api.clone();
            let handle = std::thread::spawn(move || {
                api.fetch(model.clone()).expect("Failed to fetch model")
            });
            handles.push(handle);
        }

        let path_bufs = handles
            .into_iter()
            .map(|h| {
                let path_bufs = h.join().unwrap();
            })
            .collect();

        info!("Starting Core Dispatcher..");

        let model_configs = models.iter().map(|mt| mt.model_config()).collect();
        let device = Device::new_metal(ordinal);

        let tokenizer_file = inference_config.tokenizer_file_path();
        let tokenizer =
            Tokenizer::from_file(tokenizer_file).map_err(InferenceServiceError::TokenizerError)?;

        let var_builder = VarBuilder::pp(&self, s);

        let (dispatcher, model_thread_handle) =
            ModelThreadDispatcher::start(inference_core, public_key)
                .map_err(InferenceServiceError::ModelThreadError)?;
        let start_time = Instant::now();

        Ok(Self {
            dispatcher,
            model_thread_handle,
            start_time,
            _request_receiver,
        })
    }

    pub async fn run_inference(
        &self,
        inference_request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceServiceError> {
        self.dispatcher
            .run_inference(inference_request)
            .await
            .map_err(InferenceServiceError::ModelThreadError)
    }
}

impl InferenceService {
    pub async fn stop(mut self) {
        info!(
            "Stopping Inference Service, running time: {:?}",
            self.start_time.elapsed()
        );

        self.model_thread_handle.drain(..).map(|h| h.stop());
    }
}

#[derive(Debug, Error)]
pub enum InferenceServiceError {
    #[error("Failed to connect to API: `{0}`")]
    FailedApiConnection(ApiError),
    #[error("Failed to run inference: `{0}`")]
    FailedInference(Box<dyn std::error::Error + Send + Sync>),
    #[error("Failed to fecth model: `{0}`")]
    FailedModelFetch(String),
    #[error("Failed to generate private key: `{0}`")]
    PrivateKeyError(io::Error),
    #[error("Core error: `{0}`")]
    ModelThreadError(ModelThreadError),
    #[error("Send error: `{0}`")]
    SendError(SendError<InferenceResponse>),
    #[error("Api error: `{0}`")]
    ApiError(ApiError),
    #[error("Tokenizer error: `{0}`")]
    TokenizerError(Box<dyn std::error::Error + Send + Sync + 'static>),
}

impl From<ApiError> for InferenceServiceError {
    fn from(error: ApiError) -> Self {
        Self::ApiError(error)
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::OsRng;
    use std::io::Write;
    use toml::{toml, Value};

    use crate::models::ModelType;

    use super::*;

    #[derive(Clone)]
    struct TestApiInstance {}

    impl ApiTrait for TestApiInstance {
        fn create(_: String, _: PathBuf) -> Result<Self, ApiError>
        where
            Self: Sized,
        {
            Ok(Self {})
        }

        fn fetch(&self, _: ModelType) -> Result<Vec<PathBuf>, ApiError> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn test_inference_service_initialization() {
        const CONFIG_FILE_PATH: &str = "./inference.toml";
        const PRIVATE_KEY_FILE_PATH: &str = "./private_key";

        let private_key = PrivateKey::new(&mut OsRng);
        std::fs::write(PRIVATE_KEY_FILE_PATH, private_key.to_bytes()).unwrap();

        let config_data = Value::Table(toml! {
            api_key = "your_api_key"
            models = ["Mamba3b"]
            storage_folder = "./storage_folder/"
            tokenizer_file_path = "./tokenizer_file_path/"
            tracing = true
        });
        let toml_string =
            toml::to_string_pretty(&config_data).expect("Failed to serialize to TOML");

        let mut file = std::fs::File::create(CONFIG_FILE_PATH).expect("Failed to create file");
        file.write_all(toml_string.as_bytes())
            .expect("Failed to write to file");

        let (_, receiver) = tokio::sync::mpsc::channel(1);

        let _ = InferenceService::start::<TestApiInstance>(
            PathBuf::try_from(CONFIG_FILE_PATH).unwrap(),
            PathBuf::try_from(PRIVATE_KEY_FILE_PATH).unwrap(),
            receiver,
        )
        .unwrap();

        std::fs::remove_file(CONFIG_FILE_PATH).unwrap();
        std::fs::remove_file(PRIVATE_KEY_FILE_PATH).unwrap();
    }
}
