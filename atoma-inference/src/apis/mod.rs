pub mod hugging_face;
use async_trait::async_trait;
use hf_hub::api::sync::ApiError as HuggingFaceError;

use std::path::PathBuf;

use thiserror::Error;

use crate::models::ModelType;

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

pub trait ApiTrait {
    fn call(&mut self) -> Result<(), ApiError>;
    fn fetch(&self, model: ModelType) -> Result<(), ApiError>;
    fn create(api_key: String, cache_dir: PathBuf) -> Result<Self, ApiError>
    where
        Self: Sized;
}
