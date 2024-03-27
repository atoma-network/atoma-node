use ed25519_consensus::{SigningKey as PrivateKey, VerificationKey as PublicKey};
use thiserror::Error;
use tracing::info;

use crate::{
    apis::{ApiError, ApiTrait},
    config::InferenceConfig,
    models::ModelType,
    types::{InferenceResponse, ModelResponse, QuantizationMethod, Temperature},
};

#[allow(dead_code)]
pub struct InferenceCore {
    pub(crate) config: InferenceConfig,
    // models: Vec<Model>,
    pub(crate) public_key: PublicKey,
    private_key: PrivateKey,
}

impl InferenceCore {
    pub fn new(
        config: InferenceConfig,
        private_key: PrivateKey,
    ) -> Result<Self, InferenceCoreError> {
        let public_key = private_key.verification_key();
        Ok(Self {
            config,
            public_key,
            private_key,
        })
    }
}

impl InferenceCore {
    #[allow(clippy::too_many_arguments)]
    pub fn inference(
        &mut self,
        prompt: String,
        model: ModelType,
        _temperature: Option<Temperature>,
        _max_tokens: usize,
        _random_seed: usize,
        _repeat_penalty: f32,
        _top_p: Option<f32>,
        _top_k: usize,
    ) -> Result<InferenceResponse, InferenceCoreError> {
        info!("Running inference on prompt: {prompt}, for model: {model}");
        let mut model_path = self.config.storage_folder().clone();
        model_path.push(model.to_string());

        todo!()
    }
}

#[derive(Debug, Error)]
pub enum InferenceCoreError {
    #[error("Failed to generate inference output: `{0}`")]
    FailedInference(Box<dyn std::error::Error + Send + Sync>),
    #[error("Failed to fetch new AI model: `{0}`")]
    FailedModelFetch(String),
    #[error("Failed to connect to web2 API: `{0}`")]
    FailedApiConnection(ApiError),
}

impl From<ApiError> for InferenceCoreError {
    fn from(error: ApiError) -> Self {
        InferenceCoreError::FailedApiConnection(error)
    }
}
