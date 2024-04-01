use std::path::PathBuf;

use async_trait::async_trait;
use hf_hub::{
    api::sync::{Api, ApiBuilder},
    Repo, RepoType,
};

use crate::models::ModelId;

use super::{ApiError, ApiTrait};

#[async_trait]
impl ApiTrait for Api {
    fn create(api_key: String, cache_dir: PathBuf) -> Result<Self, ApiError>
    where
        Self: Sized,
    {
        Ok(ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .with_cache_dir(cache_dir)
            .build()?)
    }

    fn fetch(&self, model_id: ModelId, revision: String) -> Result<Vec<PathBuf>, ApiError> {
        let repo = self.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        let mut filenames = Vec::with_capacity(3);

        filenames.push(repo.get("tokenizer.json")?);
        filenames.push(repo.get("config.json")?);
        filenames.push(repo.get("model.safetensors")?);

        Ok(filenames)
    }
}
