use ed25519_consensus::{SigningKey as PrivateKey, VerificationKey as PublicKey};
use thiserror::Error;

use crate::{
    config::InferenceConfig,
    types::{InferenceResponse, Model, ModelResponse, Prompt, QuantizationMethod, Temperature},
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
        let web2_api = T::connect(&config.api_key)?;
        Ok(Self {
            config,
            public_key,
            private_key,
            web2_api,
        })
    }
}

impl<T: ApiTrait> InferenceCore<T> {
    pub fn inference(
        &mut self,
        _prompt: Prompt,
        _model: Model,
        _temperature: Temperature,
        _max_tokens: usize,
        _top_p: f32,
        _top_k: usize,
    ) -> Result<InferenceResponse, InferenceCoreError> {
        todo!()
    }

    pub fn fetch_model(
        &mut self,
        _model: Model,
        _quantization_method: Option<QuantizationMethod>,
    ) -> Result<ModelResponse, InferenceCoreError> {
        Ok(ModelResponse { is_success: true })
    }
}

#[derive(Debug, Error)]
pub enum InferenceCoreError {
    #[error("Failed to generate inference output: `{0}`")]
    FailedInference(String),
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
