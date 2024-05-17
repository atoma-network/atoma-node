use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    // NcclError(NcclError),
    #[error("{0}")]
    CandleError(#[from] candle::Error),
    #[error("{0}")]
    IoError(#[from] std::io::Error),
    #[error("{0}")]
    SerdeJsonError(#[from] serde_json::Error),
    #[error("Error: `{0}`")]
    BoxedError(#[from] Box<dyn std::error::Error + Send + Sync>),
    // TokenizerError(tokenizers::Error),
    #[error("Error: `{0}`")]
    Msg(String),
    #[error("ApiError error: `{0}`")]
    ApiError(#[from] hf_hub::api::sync::ApiError),
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
