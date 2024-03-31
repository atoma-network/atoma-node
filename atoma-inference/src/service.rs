use candle::Error as CandleError;
use ed25519_consensus::SigningKey as PrivateKey;
use futures::StreamExt;
use std::{io, path::PathBuf, time::Instant};
use tokio::sync::mpsc::Receiver;
use tracing::{error, info};

use thiserror::Error;

use crate::{
    apis::{ApiError, ApiTrait},
    model_thread::{ModelThreadDispatcher, ModelThreadError, ModelThreadHandle},
    models::{config::ModelConfig, ModelTrait, Request, Response},
};

pub struct InferenceService<T, U>
where
    T: Request,
    U: Response,
{
    model_thread_handle: Vec<ModelThreadHandle<T, U>>,
    dispatcher: ModelThreadDispatcher<T, U>,
    start_time: Instant,
    request_receiver: Receiver<T>,
}

impl<T, U> InferenceService<T, U>
where
    T: Clone + Request,
    U: std::fmt::Debug + Response,
{
    pub fn start<M, F>(
        config_file_path: PathBuf,
        private_key_path: PathBuf,
        request_receiver: Receiver<T>,
    ) -> Result<Self, InferenceServiceError>
    where
        M: ModelTrait<FetchApi = F, Input = T::ModelInput, Output = U::ModelOutput>
            + Send
            + 'static,
        F: ApiTrait,
    {
        let private_key_bytes =
            std::fs::read(private_key_path).map_err(InferenceServiceError::PrivateKeyError)?;
        let private_key_bytes: [u8; 32] = private_key_bytes
            .try_into()
            .expect("Incorrect private key bytes length");

        let private_key = PrivateKey::from(private_key_bytes);
        let public_key = private_key.verification_key();
        let model_config = ModelConfig::from_file_path(config_file_path);
        let api_key = model_config.api_key();
        let storage_folder = model_config.storage_folder();

        let api = F::create(api_key, storage_folder)?;

        let (dispatcher, model_thread_handle) =
            ModelThreadDispatcher::start::<M, F>(api, model_config, public_key)
                .map_err(InferenceServiceError::ModelThreadError)?;
        let start_time = Instant::now();

        Ok(Self {
            dispatcher,
            model_thread_handle,
            start_time,
            request_receiver,
        })
    }

    pub async fn run(&mut self) -> Result<U, InferenceServiceError> {
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
}

impl<T, U> InferenceService<T, U>
where
    T: Request,
    U: Response,
{
    pub async fn stop(mut self) {
        info!(
            "Stopping Inference Service, running time: {:?}",
            self.start_time.elapsed()
        );

        let _ = self
            .model_thread_handle
            .drain(..)
            .map(|h| h.stop())
            .collect::<Vec<_>>();
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
    // #[error("Send error: `{0}`")]
    // SendError(SendError<_>),
    #[error("Api error: `{0}`")]
    ApiError(ApiError),
    #[error("Tokenizer error: `{0}`")]
    TokenizerError(Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("Candle error: `{0}`")]
    CandleError(CandleError),
}

impl From<ApiError> for InferenceServiceError {
    fn from(error: ApiError) -> Self {
        Self::ApiError(error)
    }
}

impl From<CandleError> for InferenceServiceError {
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

    use crate::models::ModelId;

    use super::*;

    struct MockApi {}

    impl ApiTrait for MockApi {
        fn create(_: String, _: PathBuf) -> Result<Self, ApiError>
        where
            Self: Sized,
        {
            Ok(Self {})
        }

        fn fetch(&self, _: &ModelId) -> Result<Vec<PathBuf>, ApiError> {
            Ok(vec![])
        }
    }

    impl Request for () {
        type ModelInput = ();

        fn into_model_input(self) -> Self::ModelInput {
            ()
        }

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

        fn from_model_output(_: Self::ModelOutput) -> Self {
            ()
        }
    }

    #[derive(Clone)]
    struct TestModelInstance {}

    impl ModelTrait for TestModelInstance {
        type Builder = ();
        type FetchApi = MockApi;
        type Input = ();
        type Output = ();

        fn fetch(_: &Self::FetchApi, _: ModelConfig) -> Result<(), crate::models::ModelError> {
            Ok(())
        }

        fn load(_: Vec<PathBuf>) -> Result<Self, crate::models::ModelError> {
            Ok(Self {})
        }

        fn model_id(&self) -> crate::models::ModelId {
            String::from("")
        }

        fn run(&self, _: Self::Input) -> Result<Self::Output, crate::models::ModelError> {
            Ok(())
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

        let (_, receiver) = tokio::sync::mpsc::channel::<()>(1);

        let _ = InferenceService::<(), ()>::start::<TestModelInstance, MockApi>(
            PathBuf::try_from(CONFIG_FILE_PATH).unwrap(),
            PathBuf::try_from(PRIVATE_KEY_FILE_PATH).unwrap(),
            receiver,
        )
        .unwrap();

        std::fs::remove_file(CONFIG_FILE_PATH).unwrap();
        std::fs::remove_file(PRIVATE_KEY_FILE_PATH).unwrap();
    }
}
