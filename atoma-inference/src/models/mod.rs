use std::path::PathBuf;

use ::candle::{DTypeParseError, Error as CandleError};
use atoma_types::{AtomaStreamingData, Digest, ModelParams};
#[cfg(feature = "nccl")]
use cudarc::{driver::DriverError, nccl::result::NcclError};
use thiserror::Error;
use tokio::sync::mpsc;
use types::{TextModelInput, TextModelOutput};

use self::{
    config::ModelConfig,
    types::{LlmOutput, ModelType},
};

pub mod candle;
pub mod config;
pub mod token_output_stream;
pub mod types;

pub type ModelId = String;

/// `ModelTrait` - An interface to host and run inference on any large language model
///
/// Such interface abstracts the fetching, loading and running of an LLM. Moreover, it
/// indirectly expects that fetching is done through some API (most likely the HuggingFace api).
pub trait ModelTrait {
    type Input: TryFrom<(Digest, ModelParams), Error = ModelError>;
    type Output: LlmOutput;
    type LoadData;

    /// Fetching the model, from an external API.
    fn fetch(
        api_key: String,
        cache_dir: PathBuf,
        config: ModelConfig,
    ) -> Result<Self::LoadData, ModelError>;
    /// Loading the model from a `LoadData`
    fn load(
        load_data: Self::LoadData,
        stream_tx: tokio::sync::mpsc::Sender<AtomaStreamingData>,
    ) -> Result<Self, ModelError>
    where
        Self: Sized;
    /// Specifies which model is being encapsulated within this type
    fn model_type(&self) -> ModelType;
    /// Responsible for running inference on a prompt request
    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError>;
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
    #[error("Invalid model input")]
    InvalidModelInput,
    #[error("Send error: `{0}`")]
    SendError(#[from] mpsc::error::SendError<AtomaStreamingData>),
    #[error("Invalid prompt params")]
    InvalidModelParams,
    #[cfg(feature = "nccl")]
    #[error("Nccl error: `{}`", 0.0)]
    NcclError(NcclError),
    #[cfg(feature = "nccl")]
    #[error("DriverError error: `{0}`")]
    DriverError(#[from] DriverError),
    #[error("Tokio error: `{0}`")]
    RecvError(#[from] tokio::sync::broadcast::error::RecvError),
    #[error("Send error: `{0}`")]
    SendErrorTextModelInput(#[from] Box<tokio::sync::broadcast::error::SendError<TextModelInput>>),
    #[error("Send error: `{0}`")]
    SendErrorTextModelOutput(#[from] tokio::sync::mpsc::error::SendError<TextModelOutput>),
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
