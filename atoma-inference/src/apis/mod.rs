pub mod hugging_face;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Api Error: {0}")]
    ApiError(String),
}

pub trait ApiTrait {
    fn call(&mut self) -> Result<(), ApiError>;
    fn fetch(&mut self) -> Result<(), ApiError>;
    fn connect(api_key: &str) -> Result<Self, ApiError>
    where
        Self: Sized;
}
