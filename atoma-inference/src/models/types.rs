use std::{path::PathBuf, str::FromStr};

use candle::{DType, Device};
use ed25519_consensus::VerificationKey as PublicKey;
use serde::{Deserialize, Serialize};

use crate::models::{ModelId, Request, Response};

use super::{candle::stable_diffusion::StableDiffusionInput, ModelError};

pub type NodeId = PublicKey;

#[derive(Debug)]
pub struct LlmLoadData {
    pub device: Device,
    pub dtype: DType,
    pub file_paths: Vec<PathBuf>,
    pub model_type: ModelType,
    pub use_flash_attention: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelType {
    Falcon7b,
    Falcon40b,
    Falcon180b,
    LlamaV1,
    LlamaV2,
    LlamaSolar10_7B,
    LlamaTinyLlama1_1BChat,
    Mamba130m,
    Mamba370m,
    Mamba790m,
    Mamba1_4b,
    Mamba2_8b,
    Mistral7b,
    Mixtral8x7b,
    StableDiffusionV1_5,
    StableDiffusionV2_1,
    StableDiffusionXl,
    StableDiffusionTurbo,
}

impl FromStr for ModelType {
    type Err = ModelError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "falcon_7b" => Ok(Self::Falcon7b),
            "falcon_40b" => Ok(Self::Falcon40b),
            "falcon_180b" => Ok(Self::Falcon180b),
            "llama_v1" => Ok(Self::LlamaV1),
            "llama_v2" => Ok(Self::LlamaV2),
            "llama_solar_10_7b" => Ok(Self::LlamaSolar10_7B),
            "llama_tiny_llama_1_1b_chat" => Ok(Self::LlamaTinyLlama1_1BChat),
            "mamba_130m" => Ok(Self::Mamba130m),
            "mamba_370m" => Ok(Self::Mamba370m),
            "mamba_790m" => Ok(Self::Mamba790m),
            "mamba_1-4b" => Ok(Self::Mamba1_4b),
            "mamba_2-8b" => Ok(Self::Mamba2_8b),
            "mistral_7b" => Ok(Self::Mistral7b),
            "mixtral_8x7b" => Ok(Self::Mixtral8x7b),
            "stable_diffusion_v1-5" => Ok(Self::StableDiffusionV1_5),
            "stable_diffusion_v2-1" => Ok(Self::StableDiffusionV2_1),
            "stable_diffusion_xl" => Ok(Self::StableDiffusionXl),
            "stable_diffusion_turbo" => Ok(Self::StableDiffusionTurbo),
            _ => Err(ModelError::InvalidModelType(
                "Invalid string model type descryption".to_string(),
            )),
        }
    }
}

impl ModelType {
    pub fn repo(&self) -> &'static str {
        match self {
            Self::Falcon7b => "tiiuae/falcon-7b",
            Self::Falcon40b => "tiiuae/falcon-40b",
            Self::Falcon180b => "tiiuae/falcon-180b",
            Self::LlamaV1 => "Narsil/amall-7b",
            Self::LlamaV2 => "meta-llama/Llama-2-7b-hf",
            Self::LlamaSolar10_7B => "upstage/SOLAR-10.7B-v1.0",
            Self::LlamaTinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Self::Mamba130m => "state-spaces/mamba-130m",
            Self::Mamba370m => "state-spaces/mamba-370m",
            Self::Mamba790m => "state-spaces/mamba-790m",
            Self::Mamba1_4b => "state-spaces/mamba-1.4b",
            Self::Mamba2_8b => "state-spaces/mamba-2.8b",
            Self::Mistral7b => "TODO",
            Self::Mixtral8x7b => "TODO",
            Self::StableDiffusionV1_5 => "runwayml/stable-diffusion-v1-5",
            Self::StableDiffusionV2_1 => "stabilityai/stable-diffusion-2-1",
            Self::StableDiffusionXl => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::StableDiffusionTurbo => "stabilityai/sdxl-turbo",
        }
    }

    pub fn default_revision(&self) -> &'static str {
        match self {
            Self::Falcon7b => "refs/pr/43",
            Self::Falcon40b => "refs/pr/43",
            Self::Falcon180b => "refs/pr/43",
            Self::LlamaV1 => "main",
            Self::LlamaV2 => "main",
            Self::LlamaSolar10_7B => "main",
            Self::LlamaTinyLlama1_1BChat => "main",
            Self::Mamba130m => "refs/pr/1",
            Self::Mamba370m => "refs/pr/1",
            Self::Mamba790m => "refs/pr/1",
            Self::Mamba1_4b => "refs/pr/1",
            Self::Mamba2_8b => "refs/pr/4",
            Self::Mistral7b => "TODO",
            Self::Mixtral8x7b => "TODO",
            Self::StableDiffusionV1_5 => "",
            Self::StableDiffusionV2_1 => "",
            Self::StableDiffusionTurbo => "",
            Self::StableDiffusionXl => "",
        }
    }
}

impl ToString for ModelType {
    fn to_string(&self) -> String {
        match self {
            Self::Falcon7b => "falcon_7b".to_string(),
            Self::Falcon40b => "falcon_40b".to_string(),
            Self::Falcon180b => "falcon_180b".to_string(),
            Self::LlamaV1 => "llama_v1".to_string(),
            Self::LlamaV2 => "llama_v2".to_string(),
            Self::LlamaSolar10_7B => "llama_solar_10_7b".to_string(),
            Self::LlamaTinyLlama1_1BChat => "llama_tiny_llama_1_1b_chat".to_string(),
            Self::Mamba130m => "mamba_130m".to_string(),
            Self::Mamba370m => "mamba_370m".to_string(),
            Self::Mamba790m => "mamba_790m".to_string(),
            Self::Mamba1_4b => "mamba_1-4b".to_string(),
            Self::Mamba2_8b => "mamba_2-8b".to_string(),
            Self::Mistral7b => "mistral_7b".to_string(),
            Self::Mixtral8x7b => "mixtral_8x7b".to_string(),
            Self::StableDiffusionV1_5 => "stable_diffusion_v1-5".to_string(),
            Self::StableDiffusionV2_1 => "stable_diffusion_v2-1".to_string(),
            Self::StableDiffusionXl => "stable_diffusion_xl".to_string(),
            Self::StableDiffusionTurbo => "stable_diffusion_turbo".to_string(),
        }
    }
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
    pub _top_k: usize,
    pub top_p: Option<f32>,
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
            self._top_k,
            self.top_p.unwrap_or_default() as f64,
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
    pub(crate) _top_k: usize,
    pub(crate) top_p: f64,
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
        _top_k: usize,
        top_p: f64,
    ) -> Self {
        Self {
            prompt,
            temperature,
            random_seed,
            repeat_penalty,
            repeat_last_n,
            max_tokens,
            _top_k,
            top_p,
        }
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

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub enum PrecisionBits {
    BF16,
    F16,
    F32,
    F64,
    I64,
    U8,
    U32,
}

impl PrecisionBits {
    #[allow(dead_code)]
    pub(crate) fn into_dtype(self) -> DType {
        match self {
            Self::BF16 => DType::BF16,
            Self::F16 => DType::F16,
            Self::F32 => DType::F32,
            Self::F64 => DType::F64,
            Self::I64 => DType::I64,
            Self::U8 => DType::U8,
            Self::U32 => DType::U32,
        }
    }
}
