use std::{fmt::Display, path::PathBuf};

use atoma_types::ModelType;
use candle::{DType, Device};
use ed25519_consensus::VerificationKey as PublicKey;
use serde::{Deserialize, Serialize};

use crate::models::{ModelId, Request, Response};

use super::candle::stable_diffusion::StableDiffusionInput;

pub type NodeId = PublicKey;

#[derive(Debug)]
pub struct LlmLoadData {
    pub device: Device,
    pub dtype: DType,
    pub file_paths: Vec<PathBuf>,
    pub model_type: ModelType,
    pub use_flash_attention: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TextRequest {
    pub request_id: usize,
    pub prompt: String,
    pub model: ModelId,
    pub max_tokens: usize,
    pub random_seed: usize,
    pub repeat_last_n: usize,
    pub repeat_penalty: f32,
    pub sampled_nodes: Vec<NodeId>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
}

impl Request for TextRequest {
    type ModelInput = TextModelInput;

    fn into_model_input(self) -> Self::ModelInput {
        TextModelInput::new(
            self.prompt,
            self.temperature.unwrap_or_default() as f64,
            self.random_seed as u64,
            self.repeat_penalty,
            self.repeat_last_n,
            self.max_tokens,
            self.top_k,
            self.top_p,
        )
    }

    fn request_id(&self) -> usize {
        self.request_id
    }

    fn is_node_authorized(&self, public_key: &PublicKey) -> bool {
        self.sampled_nodes.contains(public_key)
    }

    fn requested_model(&self) -> ModelId {
        self.model.clone()
    }
}

#[derive(Deserialize)]
pub struct TextModelInput {
    pub(crate) prompt: String,
    pub(crate) temperature: f64,
    pub(crate) random_seed: u64,
    pub(crate) repeat_penalty: f32,
    pub(crate) repeat_last_n: usize,
    pub(crate) max_tokens: usize,
    pub(crate) top_k: Option<usize>,
    pub(crate) top_p: Option<f64>,
}

impl TextModelInput {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        prompt: String,
        temperature: f64,
        random_seed: u64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        max_tokens: usize,
        top_k: Option<usize>,
        top_p: Option<f64>,
    ) -> Self {
        Self {
            prompt,
            temperature,
            random_seed,
            repeat_penalty,
            repeat_last_n,
            max_tokens,
            top_k,
            top_p,
        }
    }
}

#[derive(Serialize)]
pub struct TextModelOutput {
    pub text: String,
    pub time: f64,
    pub tokens_count: usize,
}

impl Display for TextModelOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Output: {}\nTime: {}\nTokens count: {}",
            self.text, self.time, self.tokens_count
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TextResponse {
    pub output: String,
    pub is_success: bool,
    pub status: String,
}

impl Response for TextResponse {
    type ModelOutput = String;

    fn from_model_output(model_output: Self::ModelOutput) -> Self {
        Self {
            output: model_output,
            is_success: true,
            status: "Successful".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StableDiffusionRequest {
    pub request_id: usize,
    pub prompt: String,
    pub uncond_prompt: String,

    pub height: Option<usize>,
    pub width: Option<usize>,

    /// The number of steps to run the diffusion for.
    pub n_steps: Option<usize>,

    /// The number of samples to generate.
    pub num_samples: i64,

    pub model: ModelId,

    pub guidance_scale: Option<f64>,

    pub img2img: Option<String>,

    /// The strength, indicates how much to transform the initial image. The
    /// value must be between 0 and 1, a value of 1 discards the initial image
    /// information.
    pub img2img_strength: f64,

    /// The seed to use when generating random samples.
    pub random_seed: Option<u64>,

    pub sampled_nodes: Vec<NodeId>,
}

impl Request for StableDiffusionRequest {
    type ModelInput = StableDiffusionInput;

    fn into_model_input(self) -> Self::ModelInput {
        Self::ModelInput {
            prompt: self.prompt,
            uncond_prompt: self.uncond_prompt,
            height: self.height,
            width: self.width,
            n_steps: self.n_steps,
            num_samples: self.num_samples,
            model: self.model,
            guidance_scale: self.guidance_scale,
            img2img: self.img2img,
            img2img_strength: self.img2img_strength,
            random_seed: self.random_seed,
        }
    }

    fn is_node_authorized(&self, public_key: &PublicKey) -> bool {
        self.sampled_nodes.contains(public_key)
    }

    fn request_id(&self) -> usize {
        self.request_id
    }

    fn requested_model(&self) -> ModelId {
        self.model.clone()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StableDiffusionResponse {
    pub output: Vec<(Vec<u8>, usize, usize)>,
    pub is_success: bool,
    pub status: String,
}

impl Response for StableDiffusionResponse {
    type ModelOutput = Vec<(Vec<u8>, usize, usize)>;

    fn from_model_output(model_output: Self::ModelOutput) -> Self {
        Self {
            output: model_output,
            is_success: true,
            status: "Successful".to_string(),
        }
    }
}
