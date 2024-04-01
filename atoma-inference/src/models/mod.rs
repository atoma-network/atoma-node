use std::path::PathBuf;

use ::candle::Error as CandleError;
use ed25519_consensus::VerificationKey as PublicKey;
use thiserror::Error;

use crate::models::types::PrecisionBits;

pub mod candle;
pub mod config;
pub mod token_output_stream;
pub mod types;

pub type ModelId = String;

pub trait ModelTrait {
    type Input;
    type Output;

    fn load(filenames: Vec<PathBuf>, precision: PrecisionBits) -> Result<Self, ModelError>
    where
        Self: Sized;
    fn model_id(&self) -> ModelId;
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
    #[error("Tokenizer error: `{0}`")]
    TokenizerError(Box<dyn std::error::Error + Send + Sync>),
    #[error("IO error: `{0}`")]
    IoError(std::io::Error),
    #[error("Deserialize error: `{0}`")]
    DeserializeError(serde_json::Error),
    #[error("Candle error: `{0}`")]
    CandleError(CandleError),
    #[error("{0}")]
    Msg(String),
}

impl From<CandleError> for ModelError {
    fn from(error: CandleError) -> Self {
        Self::CandleError(error)
    }
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
