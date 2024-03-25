use ed25519_consensus::SigningKey as PrivateKey;
use std::{io, path::PathBuf, time::Instant};
use tokio::sync::mpsc::{error::SendError, Receiver};
use tracing::{info, warn};

use thiserror::Error;

use crate::{
    apis::{ApiError, ApiTrait},
    config::InferenceConfig,
    core::{InferenceCore, InferenceCoreError},
    core_thread::{CoreError, CoreThreadDispatcher, CoreThreadHandle},
    types::{InferenceRequest, InferenceResponse, ModelRequest, ModelResponse},
};

pub struct InferenceService {
    core_thread_handle: CoreThreadHandle,
    dispatcher: CoreThreadDispatcher,
    start_time: Instant,
    request_receiver: Receiver<InferenceRequest>,
}

impl InferenceService {
    pub async fn start<T: ApiTrait + Send + 'static>(
        config_file_path: PathBuf,
        private_key_path: PathBuf,
        request_receiver: Receiver<InferenceRequest>,
    ) -> Result<Self, InferenceServiceError> {
        let private_key_bytes =
            std::fs::read(&private_key_path).map_err(InferenceServiceError::PrivateKeyError)?;
        let private_key_bytes: [u8; 32] = private_key_bytes
            .try_into()
            .expect("Incorrect private key bytes length");

        let private_key = PrivateKey::from(private_key_bytes);
        let inference_config = InferenceConfig::from_file_path(config_file_path);
        let models = inference_config.models();
        let inference_core = InferenceCore::<T>::new(inference_config, private_key)?;

        let (dispatcher, core_thread_handle) = CoreThreadDispatcher::start(inference_core);
        let start_time = Instant::now();

        let inference_service = Self {
            dispatcher,
            core_thread_handle,
            start_time,
            request_receiver,
        };

        for model in models {
            let response = inference_service
                .fetch_model(ModelRequest {
                    model: model.clone(),
                    quantization_method: None,
                })
                .await?;
            if !response.is_success {
                warn!(
                    "Failed to fetch model: {:?}, with error: {:?}",
                    model, response.error
                );
            }
        }

        Ok(inference_service)
    }

    async fn run_inference(
        &self,
        inference_request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceServiceError> {
        self.dispatcher
            .run_inference(inference_request)
            .await
            .map_err(InferenceServiceError::CoreError)
    }

    async fn fetch_model(
        &self,
        model_request: ModelRequest,
    ) -> Result<ModelResponse, InferenceServiceError> {
        self.dispatcher
            .fetch_model(model_request)
            .await
            .map_err(InferenceServiceError::CoreError)
    }
}

impl InferenceService {
    pub async fn stop(self) {
        info!(
            "Stopping Inference Service, running time: {:?}",
            self.start_time.elapsed()
        );

        self.core_thread_handle.stop().await;
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
    CoreError(CoreError),
    #[error("Send error: `{0}`")]
    SendError(SendError<InferenceResponse>),
}

impl From<InferenceCoreError> for InferenceServiceError {
    fn from(error: InferenceCoreError) -> Self {
        match error {
            InferenceCoreError::FailedApiConnection(e) => Self::FailedApiConnection(e),
            InferenceCoreError::FailedInference(e) => Self::FailedInference(e),
            InferenceCoreError::FailedModelFetch(e) => Self::FailedModelFetch(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::OsRng;
    use std::io::Write;
    use toml::{toml, Value};

    use super::*;

    struct TestApiInstance {}

    impl ApiTrait for TestApiInstance {
        fn call(&mut self) -> Result<(), ApiError> {
            Ok(())
        }

        fn connect(_: &str) -> Result<Self, ApiError>
        where
            Self: Sized,
        {
            Ok(Self {})
        }

        fn fetch(&mut self) -> Result<(), ApiError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_inference_service_initialization() {
        const CONFIG_FILE_PATH: &str = "./config.toml";
        const PRIVATE_KEY_FILE_PATH: &str = "./private_key";

        let private_key = PrivateKey::new(&mut OsRng);
        std::fs::write(PRIVATE_KEY_FILE_PATH, private_key.to_bytes()).unwrap();

        let config_data = Value::Table(toml! {
            api_key = "your_api_key"
            models = [{ Mamba = 3 }]
            storage_base_path = "./storage_base_path/"
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
        .await
        .unwrap();

        std::fs::remove_file(CONFIG_FILE_PATH).unwrap();
        std::fs::remove_file(PRIVATE_KEY_FILE_PATH).unwrap();
    }
}
