use std::path::PathBuf;

use crate::{apis::ApiTrait, models::config::ModelConfig};
use ed25519_consensus::VerificationKey as PublicKey;
use thiserror::Error;

pub mod config;

pub type ModelId = String;

pub trait ModelBuilder {
    fn try_from_file(path: PathBuf) -> Result<Self, ModelError>
    where
        Self: Sized;
}

pub trait ModelTrait {
    type Builder: Send + Sync + 'static;
    type FetchApi: ApiTrait + Send + Sync + 'static;
    type Input;
    type Output;

    fn fetch(api: &Self::FetchApi, config: ModelConfig) -> Result<(), ModelError>;
    fn load(filenames: Vec<PathBuf>) -> Result<Self, ModelError>
    where
        Self: Sized;
    fn model_id(&self) -> ModelId;
    fn run(&self, input: Self::Input) -> Result<Self::Output, ModelError>;
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

#[derive(Debug, Error)]
pub enum ModelError {}
