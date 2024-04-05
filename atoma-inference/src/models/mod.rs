use std::path::PathBuf;

use ::candle::{DTypeParseError, Error as CandleError};
use ed25519_consensus::VerificationKey as PublicKey;
use thiserror::Error;

use self::{config::ModelConfig, types::ModelType};

pub mod candle;
pub mod config;
pub mod token_output_stream;
pub mod types;

pub type ModelId = String;

pub trait ModelTrait {
    type Input;
    type Output;
    type LoadData;

    fn fetch(cache_dir: PathBuf, config: ModelConfig) -> Result<Self::LoadData, ModelError>;
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

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Deserialize error: `{0}`")]
    DeserializeError(#[from] serde_json::Error),
    #[error("{0}")]
    Msg(String),
    #[error("Candle error: `{0}`")]
    CandleError(#[from] CandleError),
    #[error("Config error: `{0}`")]
    Config(String),
    #[error("Image error: `{0}`")]
    ImageError(#[from] image::ImageError),
    #[error("Io error: `{0}`")]
    IoError(#[from] std::io::Error),
    #[error("Error: `{0}`")]
    BoxedError(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("ApiError error: `{0}`")]
    ApiError(#[from] hf_hub::api::sync::ApiError),
    #[error("DTypeParseError: `{0}`")]
    DTypeParseError(#[from] DTypeParseError),
    #[error("Invalid model type: `{0}`")]
    InvalidModelType(String),
}

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err(ModelError::Msg(format!($msg).into()))
    };
    ($err:expr $(,)?) => {
        return Err(ModelError::Msg(format!($err).into()).bt())
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err(ModelError::Msg(format!($fmt, $($arg)*).into()).bt())
    };
}
