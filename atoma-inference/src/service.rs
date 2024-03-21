use ed25519_consensus::{SigningKey as PrivateKey, VerificationKey as PublicKey};
use thiserror::Error;

use crate::{
    config::InferenceConfig,
    types::{InferenceResponse, ModelResponse, ModelType, Prompt, QuantizationMethod, Temperature},
};

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Api Error: {0}")]
    ApiError(String),
}

pub trait ApiTrait {
    fn call(&mut self) -> Result<(), ApiError>;
    fn fetch(&mut self) -> Result<(), ApiError>;
    fn connect(api_key: &str) -> Result<Self, ApiError>
    where
        Self: Sized;
}

#[allow(dead_code)]
pub struct InferenceCore<T> {
    config: InferenceConfig,
    pub(crate) public_key: PublicKey,
    private_key: PrivateKey,
    web2_api: T,
}

impl<T: ApiTrait> InferenceCore<T> {
    pub fn new(
        config: InferenceConfig,
        private_key: PrivateKey,
    ) -> Result<Self, InferenceCoreError> {
        let public_key = private_key.verification_key();
        let web2_api = T::connect(&config.api_key())?;
        Ok(Self {
            config,
            public_key,
            private_key,
            web2_api,
        })
    }
}

impl<T: ApiTrait> InferenceCore<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn inference(
        &mut self,
        _prompt: Prompt,
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

    pub fn fetch_model(
        &mut self,
        _model: ModelType,
        _quantization_method: Option<QuantizationMethod>,
    ) -> Result<ModelResponse, InferenceCoreError> {
        Ok(ModelResponse { is_success: true })
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