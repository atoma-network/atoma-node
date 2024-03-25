use ed25519_consensus::{SigningKey as PrivateKey, VerificationKey as PublicKey};
use thiserror::Error;

use crate::{
    apis::{ApiError, ApiTrait},
    config::InferenceConfig,
    models::ModelType,
    types::{InferenceResponse, ModelResponse, QuantizationMethod, Temperature},
};

#[allow(dead_code)]
pub struct InferenceCore<T> {
    pub(crate) config: InferenceConfig,
    // models: Vec<Model>,
    pub(crate) public_key: PublicKey,
    private_key: PrivateKey,
    pub(crate) api: T,
}

impl<T: ApiTrait> InferenceCore<T> {
    pub fn new(
        config: InferenceConfig,
        private_key: PrivateKey,
    ) -> Result<Self, InferenceCoreError> {
        let public_key = private_key.verification_key();
        let api = T::create(config.api_key(), config.storage_base_path())?;
        Ok(Self {
            config,
            public_key,
            private_key,
            api,
        })
    }
}

impl<T: ApiTrait> InferenceCore<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn inference(
        &mut self,
        _prompt: String,
        model: ModelType,
        _temperature: Option<Temperature>,
        _max_tokens: usize,
        _random_seed: usize,
        _repeat_penalty: f32,
        _top_p: Option<f32>,
        _top_k: usize,
    ) -> Result<InferenceResponse, InferenceCoreError> {
        let mut model_path = self.config.storage_base_path().clone();
        model_path.push(model.to_string());

        // let tokenizer = Tokenizer::from_file(self.config.tokenizer_file_path())
        //     .map_err(InferenceCoreError::FailedInference)?;
        // let mut tokens = tokenizer
        //     .encode(prompt.0, true)
        //     .map_err(InferenceCoreError::FailedInference)?;

        todo!()
    }

    pub async fn fetch_model(
        &mut self,
        model: ModelType,
        _quantization_method: Option<QuantizationMethod>,
    ) -> Result<ModelResponse, InferenceCoreError> {
        self.api.fetch(model).await?;
        Ok(ModelResponse {
            is_success: true,
            error: None,
        })
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
