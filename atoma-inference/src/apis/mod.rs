pub mod hugging_face;
use hf_hub::api::sync::ApiError as HuggingFaceError;

use std::path::PathBuf;

use thiserror::Error;

use crate::models::ModelId;

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

pub trait ApiTrait: Send {
    fn fetch(&self, model_id: ModelId, revision: String) -> Result<Vec<PathBuf>, ApiError>;
    fn create(api_key: String, cache_dir: PathBuf) -> Result<Self, ApiError>
    where
        Self: Sized;
}
