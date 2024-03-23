use ed25519_consensus::SigningKey as PrivateKey;
use std::{io, path::PathBuf};

use thiserror::Error;

use crate::{
    config::InferenceConfig,
    core::{ApiError, ApiTrait, InferenceCore, InferenceCoreError},
    core_thread::{CoreError, CoreThreadDispatcher, CoreThreadHandle},
    types::{InferenceRequest, InferenceResponse, ModelRequest, ModelResponse},
};

pub struct InferenceService {
    dispatcher: CoreThreadDispatcher,
    core_thread_handle: CoreThreadHandle,
}

impl InferenceService {
    pub fn start<T: ApiTrait + Send + 'static>(
        config: InferenceConfig,
        private_key_path: PathBuf,
    ) -> Result<Self, InferenceServiceError> {
        let private_key_bytes =
            std::fs::read(&private_key_path).map_err(InferenceServiceError::PrivateKeyError)?;
        let private_key_bytes: [u8; 32] = private_key_bytes
            .try_into()
            .expect("Incorrect private key bytes length");

        let private_key = PrivateKey::from(private_key_bytes);
        let inference_core = InferenceCore::<T>::new(config, private_key)?;

        let (dispatcher, core_thread_handle) = CoreThreadDispatcher::start(inference_core);

        Ok(Self {
            dispatcher,
            core_thread_handle,
        })
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
