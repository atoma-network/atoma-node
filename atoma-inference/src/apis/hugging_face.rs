use std::path::PathBuf;

use async_trait::async_trait;
use hf_hub::api::sync::{Api, ApiBuilder};

use crate::models::ModelId;

use super::{ApiError, ApiTrait};

struct FilePaths {
    file_paths: Vec<String>,
}

fn get_model_safe_tensors_from_hf(model_id: &ModelId) -> (String, FilePaths) {
    match model_id.as_str() {
        "Llama2_7b" => (
            String::from("meta-llama/Llama-2-7b-hf"),
            FilePaths {
                file_paths: vec![
                    "model-00001-of-00002.safetensors".to_string(),
                    "model-00002-of-00002.safetensors".to_string(),
                ],
            },
        ),
        "Mamba3b" => (
            String::from("state-spaces/mamba-2.8b-hf"),
            FilePaths {
                file_paths: vec![
                    "model-00001-of-00003.safetensors".to_string(),
                    "model-00002-of-00003.safetensors".to_string(),
                    "model-00003-of-00003.safetensors".to_string(),
                ],
            },
        ),
        "Mistral7b" => (
            String::from("mistralai/Mistral-7B-Instruct-v0.2"),
            FilePaths {
                file_paths: vec![
                    "model-00001-of-00003.safetensors".to_string(),
                    "model-00002-of-00003.safetensors".to_string(),
                    "model-00003-of-00003.safetensors".to_string(),
                ],
            },
        ),
        "Mixtral8x7b" => (
            String::from("mistralai/Mixtral-8x7B-Instruct-v0.1"),
            FilePaths {
                file_paths: vec![
                    "model-00001-of-00019.safetensors".to_string(),
                    "model-00002-of-00019.safetensors".to_string(),
                    "model-00003-of-00019.safetensors".to_string(),
                    "model-00004-of-00019.safetensors".to_string(),
                    "model-00005-of-00019.safetensors".to_string(),
                    "model-00006-of-00019.safetensors".to_string(),
                    "model-00007-of-00019.safetensors".to_string(),
                    "model-00008-of-00019.safetensors".to_string(),
                    "model-00009-of-00019.safetensors".to_string(),
                    "model-000010-of-00019.safetensors".to_string(),
                    "model-000011-of-00019.safetensors".to_string(),
                    "model-000012-of-00019.safetensors".to_string(),
                    "model-000013-of-00019.safetensors".to_string(),
                    "model-000014-of-00019.safetensors".to_string(),
                    "model-000015-of-00019.safetensors".to_string(),
                    "model-000016-of-00019.safetensors".to_string(),
                    "model-000017-of-00019.safetensors".to_string(),
                    "model-000018-of-00019.safetensors".to_string(),
                    "model-000019-of-00019.safetensors".to_string(),
                ],
            },
        ),
        "StableDiffusion2" => (
            String::from("stabilityai/stable-diffusion-2"),
            FilePaths {
                file_paths: vec!["768-v-ema.safetensors".to_string()],
            },
        ),
        "StableDiffusionXl" => (
            String::from("stabilityai/stable-diffusion-xl-base-1.0"),
            FilePaths {
                file_paths: vec![
                    "sd_xl_base_1.0.safetensors".to_string(),
                    "sd_xl_base_1.0_0.9vae.safetensors".to_string(),
                    "sd_xl_offset_example-lora_1.0.safetensors".to_string(),
                ],
            },
        ),
        _ => {
            panic!("Invalid model id")
        }
    }
}

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

    fn fetch(&self, model_id: &ModelId) -> Result<Vec<PathBuf>, ApiError> {
        let (model_path, files) = get_model_safe_tensors_from_hf(model_id);
        let api_repo = self.model(model_path);
        let mut path_bufs = Vec::with_capacity(files.file_paths.len());

        for file in files.file_paths {
            path_bufs.push(api_repo.get(&file)?);
        }

        Ok(path_bufs)
    }
}
