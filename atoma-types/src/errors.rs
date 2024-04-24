use std::io;

use candle::{DTypeParseError, Error as CandleError};
use hf_hub::api::sync::ApiError as HuggingFaceError;
use thiserror::Error;
use tokio::sync::oneshot::error::RecvError;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Api Error: `{0}`")]
    ApiError(String),
    #[error("HuggingFace API error: `{0}`")]
    HuggingFaceError(HuggingFaceError),
}

impl From<HuggingFaceError> for ApiError {
    fn from(error: HuggingFaceError) -> Self {
        Self::HuggingFaceError(error)
    }
}

#[derive(Debug, Error)]
pub enum ModelServiceError {
    #[error("Failed to run inference: `{0}`")]
    FailedInference(Box<dyn std::error::Error + Send + Sync>),
    #[error("Failed to fecth model: `{0}`")]
    FailedModelFetch(String),
    #[error("Failed to generate private key: `{0}`")]
    PrivateKeyError(io::Error),
    #[error("Core error: `{0}`")]
    ModelThreadError(ModelThreadError),
    #[error("Api error: `{0}`")]
    ApiError(ApiError),
    #[error("Candle error: `{0}`")]
    CandleError(CandleError),
    #[error("Sender error: `{0}`")]
    SendError(String),
}

impl From<ApiError> for ModelServiceError {
    fn from(error: ApiError) -> Self {
        Self::ApiError(error)
    }
}

impl From<CandleError> for ModelServiceError {
    fn from(error: CandleError) -> Self {
        Self::CandleError(error)
    }
}

#[derive(Debug, Error)]
pub enum ModelThreadError {
    #[error("Model thread shutdown: `{0}`")]
    ApiError(ApiError),
    #[error("Model thread shutdown: `{0}`")]
    ModelError(ModelError),
    #[error("Core thread shutdown: `{0}`")]
    Shutdown(RecvError),
    #[error("Serde error: `{0}`")]
    SerdeError(#[from] serde_json::Error),
}

impl From<ModelError> for ModelThreadError {
    fn from(error: ModelError) -> Self {
        Self::ModelError(error)
    }
}

impl From<ApiError> for ModelThreadError {
    fn from(error: ApiError) -> Self {
        Self::ApiError(error)
    }
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
