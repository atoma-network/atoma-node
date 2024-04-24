use std::path::PathBuf;

use self::{config::ModelConfig, types::ModelType};
use atoma_types::ModelError;
use ed25519_consensus::VerificationKey as PublicKey;
use serde::{de::DeserializeOwned, Serialize};

pub mod candle;
pub mod config;
pub mod token_output_stream;
pub mod types;

pub type ModelId = String;

pub trait ModelTrait {
    type Input: DeserializeOwned;
    type Output: Serialize;
    type LoadData;

    fn fetch(
        api_key: String,
        cache_dir: PathBuf,
        config: ModelConfig,
    ) -> Result<Self::LoadData, ModelError>;
    fn load(load_data: Self::LoadData) -> Result<Self, ModelError>
    where
        Self: Sized;
    fn model_type(&self) -> ModelType;
    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError>;
}

pub trait Request: Send + 'static {
    type ModelInput;

    fn into_model_input(self) -> Self::ModelInput;
    fn requested_model(&self) -> ModelId;
    fn request_id(&self) -> usize; // TODO: replace with Uuid
    fn is_node_authorized(&self, public_key: &PublicKey) -> bool;
}

pub trait Response: Send + 'static {
    type ModelOutput;

    fn from_model_output(model_output: Self::ModelOutput) -> Self;
}
