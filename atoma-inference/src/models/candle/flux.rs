#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::{path::PathBuf, str::FromStr};

use crate::{
    bail,
    models::{
        candle::convert_to_image,
        config::ModelConfig,
        types::{LlmOutput, ModelType},
        ModelError, ModelTrait,
    },
};
use atoma_types::{AtomaStreamingData, Digest, ModelParams};
use candle::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{clip, flux, t5};
use hf_hub::api::sync::ApiBuilder;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tracing::{info, trace};

use super::device;

#[derive(Debug, Clone, Deserialize, Serialize)]
/// Flux input data
pub struct FluxInput {
    /// Text input
    prompt: String,
    /// Image height
    height: Option<usize>,
    /// Image width
    width: Option<usize>,
    /// decode only
    decode_only: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Flux model variants
pub enum Model {
    Schnell,
    Dev,
}

/// Flux model structure
pub struct Flux {
    /// Device that hosts the models
    device: Device,
    /// Data type of the models
    dtype: DType,
    /// Flux model variant
    model: Model,
    /// T5 model
    t5_model: t5::T5EncoderModel,
    /// T5 tokenizer
    t5_tokenizer: Tokenizer,
    /// CLIP model
    clip_model: clip::text_model::ClipTextTransformer,
    /// CLIP tokenizer
    clip_tokenizer: Tokenizer,
    /// Biflux model
    bf_model: flux::model::Flux,
    /// Autoencoder model
    ae_model: flux::autoencoder::AutoEncoder,
}

/// Flux model load data
pub struct FluxLoadData {
    /// Device that hosts the models
    device: Device,
    /// Data type of the models
    dtype: DType,
    /// File paths to load different models
    /// configurations and tokenizers
    file_paths: Vec<PathBuf>,
    /// Flux model variant
    model: Model,
}

/// Flux model output
/// Stable diffusion output
#[derive(Serialize)]
pub struct FluxOutput {
    /// Data buffer of the image encoding
    pub image_data: Vec<u8>,
    /// Number of input tokens
    input_tokens: usize,
    /// Time to generate output
    time_to_generate: f64,
}

impl ModelTrait for Flux {
    type Input = FluxInput;
    type Output = FluxOutput;
    type LoadData = FluxLoadData;

    fn fetch(
        api_key: String,
        cache_dir: PathBuf,
        config: ModelConfig,
    ) -> Result<Self::LoadData, ModelError> {
        info!("Fetching Flux model..");
        let device = device(config.device_first_id())?;
        let dtype = DType::from_str(&config.dtype())?;

        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .with_cache_dir(cache_dir)
            .build()?;

        let model_type = ModelType::from_str(&config.model_id())?;
        let repo_id = model_type.repo().to_string();

        info!("Fetching T5 model files..");

        let bf_repo = api.repo(hf_hub::Repo::model(repo_id));
        let t5_repo = api.repo(hf_hub::Repo::with_revision(
            "google/t5-v1_1-xxl".to_string(),
            hf_hub::RepoType::Model,
            "refs/pr/2".to_string(),
        ));
        let t5_model_file = t5_repo.get("model.safetensors")?;
        let t5_config_filename = t5_repo.get("config.json")?;
        let t5_tokenizer_filename = api
            .model("lmz/mt5-tokenizers".to_string())
            .get("t5-v1_1-xxl.tokenizer.json")?;

        info!("Fetching CLIP model files..");

        let clip_repo = api.repo(hf_hub::Repo::model(
            "openai/clip-vit-large-patch14".to_string(),
        ));
        let clip_model_file = clip_repo.get("model.safetensors")?;
        let clip_tokenizer_filename = clip_repo.get("tokenizer.json")?;

        let model = match model_type {
            ModelType::FluxSchnell => Model::Schnell,
            ModelType::FluxDev => Model::Dev,
            _ => bail!("Invalid model type for Flux model"),
        };

        info!("Fetching Biflux model files..");

        let bf_model_file = match model {
            Model::Schnell => bf_repo.get("flux1-schnell.safetensors")?,
            Model::Dev => bf_repo.get("flux1-dev.safetensors")?,
        };

        info!("Fetching Autoencoder model files..");
        let ae_model_file = bf_repo.get("ae.safetensors")?;

        Ok(Self::LoadData {
            device,
            dtype,
            file_paths: vec![
                t5_config_filename,
                t5_tokenizer_filename,
                t5_model_file,
                clip_tokenizer_filename,
                clip_model_file,
                bf_model_file,
                ae_model_file,
            ],
            model,
        })
    }

    fn load(
        load_data: Self::LoadData,
        _stream_tx: tokio::sync::mpsc::Sender<AtomaStreamingData>,
    ) -> Result<Self, ModelError> {
        info!("Loading T5 model..");
        let start = std::time::Instant::now();
        let t5_config_filename = load_data.file_paths[0].clone();
        let t5_tokenizer_filename = load_data.file_paths[1].clone();
        let t5_model_filename = load_data.file_paths[2].clone();
        let clip_tokenizer_filename = load_data.file_paths[3].clone();
        let clip_model_filename = load_data.file_paths[4].clone();
        let bf_model_filename = load_data.file_paths[5].clone();
        let ae_model_filename = load_data.file_paths[6].clone();

        let t5_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[t5_model_filename],
                load_data.dtype,
                &load_data.device,
            )?
        };

        let t5_config = std::fs::read_to_string(t5_config_filename)?;
        let t5_config: t5::Config = serde_json::from_str(&t5_config)?;

        let t5_model = t5::T5EncoderModel::load(t5_vb, &t5_config)?;
        let t5_tokenizer = Tokenizer::from_file(t5_tokenizer_filename)?;

        info!(
            "Loaded T5 model in {} seconds",
            start.elapsed().as_secs_f64()
        );
        info!("Loading CLIP model..");

        let start = std::time::Instant::now();
        let clip_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[clip_model_filename],
                load_data.dtype,
                &load_data.device,
            )?
        };
        let clip_config = clip::text_model::ClipTextConfig {
            vocab_size: 49408,
            projection_dim: 768,
            activation: clip::text_model::Activation::QuickGelu,
            intermediate_size: 3072,
            embed_dim: 768,
            max_position_embeddings: 77,
            pad_with: None,
            num_hidden_layers: 12,
            num_attention_heads: 12,
        };
        let clip_model =
            clip::text_model::ClipTextTransformer::new(clip_vb.pp("text_model"), &clip_config)?;
        let clip_tokenizer = Tokenizer::from_file(clip_tokenizer_filename)?;

        info!(
            "Loaded CLIP model in {} seconds",
            start.elapsed().as_secs_f64()
        );
        info!("Loading Biflux model..");

        let start = std::time::Instant::now();
        let bf_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[bf_model_filename],
                load_data.dtype,
                &load_data.device,
            )?
        };

        let bf_config = match load_data.model {
            Model::Dev => flux::model::Config::dev(),
            Model::Schnell => flux::model::Config::schnell(),
        };
        let bf_model = flux::model::Flux::new(&bf_config, bf_vb)?;

        info!(
            "Loaded Biflux model in {} seconds",
            start.elapsed().as_secs_f64()
        );
        info!("Loading Autoencoder model..");

        let start = std::time::Instant::now();
        let ae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[ae_model_filename],
                load_data.dtype,
                &load_data.device,
            )?
        };
        let ae_config = match load_data.model {
            Model::Dev => flux::autoencoder::Config::dev(),
            Model::Schnell => flux::autoencoder::Config::schnell(),
        };
        let ae_model = flux::autoencoder::AutoEncoder::new(&ae_config, ae_vb)?;

        info!(
            "Loaded Autoencoder model in {} seconds",
            start.elapsed().as_secs_f64()
        );

        Ok(Self {
            device: load_data.device.clone(),
            dtype: load_data.dtype,
            model: load_data.model,
            t5_model,
            t5_tokenizer,
            clip_model,
            clip_tokenizer,
            bf_model,
            ae_model,
        })
    }

    fn model_type(&self) -> ModelType {
        match self.model {
            Model::Schnell => ModelType::FluxSchnell,
            Model::Dev => ModelType::FluxDev,
        }
    }

    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        info!("Running Flux model, on input prompt: {}", input.prompt);
        let start = std::time::Instant::now();

        let width = input.width.unwrap_or(1360);
        let height = input.height.unwrap_or(768);

        let mut t5_tokens = self
            .t5_tokenizer
            .encode(input.prompt.as_str(), true)?
            .get_ids()
            .to_vec();
        t5_tokens.resize(256, 0);
        let input_t5_token_ids = Tensor::new(&t5_tokens[..], &self.device)?.unsqueeze(0)?;
        let t5_embedding = self.t5_model.forward(&input_t5_token_ids)?;

        info!("Produced a T5 embedding");

        info!("Running CLIP model, on input prompt: {}", input.prompt);
        let clip_tokens = self
            .clip_tokenizer
            .encode(input.prompt.as_str(), true)?
            .get_ids()
            .to_vec();
        let input_clip_token_ids = Tensor::new(&clip_tokens[..], &self.device)?.unsqueeze(0)?;
        let clip_embedding = self.clip_model.forward(&input_clip_token_ids)?;

        info!("Produced a CLIP embedding");

        info!("Running Biflux model, on input prompt: {}", input.prompt);
        let img =
            flux::sampling::get_noise(1, height, width, &self.device)?.to_dtype(self.dtype)?;
        let state = flux::sampling::State::new(&t5_embedding, &clip_embedding, &img)?;
        let timesteps = match self.model {
            Model::Dev => flux::sampling::get_schedule(50, Some((state.img.dim(1)?, 0.5, 1.15))),
            Model::Schnell => flux::sampling::get_schedule(4, None),
        };

        trace!("{state:?}");
        trace!("{timesteps:?}");
        let img = flux::sampling::denoise(
            &self.bf_model,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            4.,
        )?;
        let img = flux::sampling::unpack(&img, height, width)?;
        trace!("latent img\n{img}");

        let img = self.ae_model.decode(&img)?;
        trace!("img\n{img}");

        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
            .to_dtype(candle::DType::U8)?
            .squeeze(0)?;
        let img = convert_to_image(&img)?;

        Ok(FluxOutput {
            image_data: img,
            input_tokens: input.prompt.len(),
            time_to_generate: start.elapsed().as_secs_f64(),
        })
    }
}

impl TryFrom<(Digest, ModelParams)> for FluxInput {
    type Error = ModelError;

    fn try_from(value: (Digest, ModelParams)) -> Result<Self, Self::Error> {
        let prompt_params = value.1;
        match prompt_params {
            ModelParams::Text2ImageModelParams(p) => {
                let height = p.height().map(|h| h as usize);
                let width = p.width().map(|w| w as usize);
                let prompt = p.get_input_text(); // TODO: for now we use the raw prompt, but likely to fetch it from an external source in the future
                let decode_only = p.decode_only();
                Ok(Self {
                    prompt,
                    height,
                    width,
                    decode_only,
                })
            }
            ModelParams::Text2TextModelParams(_) => Err(ModelError::InvalidModelParams),
        }
    }
}

impl LlmOutput for FluxOutput {
    fn num_input_tokens(&self) -> usize {
        self.input_tokens
    }
    fn num_output_tokens(&self) -> Option<usize> {
        None
    }
    fn time_to_generate(&self) -> f64 {
        self.time_to_generate
    }

    fn input_tokens(&self) -> Vec<u32> {
        unimplemented!()
    }

    fn output_tokens(&self) -> Vec<u32> {
        unimplemented!("Asking image models for the tokens is not supported")
    }

    fn text_output(&self) -> String {
        unimplemented!("Asking image models for the tokens is not supported")
    }
}
