use ed25519_consensus::{SigningKey as PrivateKey, VerificationKey as PublicKey};
use thiserror::Error;

use crate::{
    config::InferenceConfig,
    types::{InferenceResponse, Model, ModelResponse, Prompt, QuantizationMethod, Temperature},
};

pub struct InferenceCore {
    config: InferenceConfig,
    pub(crate) public_key: PublicKey,
    private_key: PrivateKey,
}

impl InferenceCore {
    pub fn new(config: InferenceConfig, private_key: PrivateKey) -> Self {
        let public_key = private_key.verification_key();
        Self {
            config,
            public_key,
            private_key,
        }
    }
}

impl InferenceCore {
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
}
