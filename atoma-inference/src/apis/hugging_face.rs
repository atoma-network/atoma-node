use std::path::PathBuf;

use async_trait::async_trait;
use hf_hub::{
    api::sync::{Api, ApiBuilder},
    Repo, RepoType,
};
use tracing::error;

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
        let repo = self.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            revision,
        ));

        Ok(vec![
            repo.get("config.json")?,
            if model_id.contains("mamba") {
                self.model("EleutherAI/gpt-neox-20b".to_string())
                    .get("tokenizer.json")
                    .map_err(|e| {
                        error!("Failed to fetch tokenizer file: {e}");
                        e
                    })?
            } else {
                repo.get("tokenizer.json")?
            },
            repo.get("model.safetensors")?,
        ])
    }
}
