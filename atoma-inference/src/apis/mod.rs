pub mod hugging_face;

use std::path::PathBuf;

use atoma_types::ApiError;

use crate::models::ModelId;

pub trait ApiTrait: Send {
    fn fetch(&self, model_id: ModelId, revision: String) -> Result<Vec<PathBuf>, ApiError>;
    fn create(api_key: String, cache_dir: PathBuf) -> Result<Self, ApiError>
    where
        Self: Sized;
}
