use std::{fmt::Display, path::PathBuf, str::FromStr};

use candle::{DType, Device};
use serde::{Deserialize, Serialize};

use crate::models::{ModelId, Request, Response};

use super::{candle::stable_diffusion::StableDiffusionInput, ModelError};

#[derive(Debug)]
pub struct LlmLoadData {
    pub device: Device,
    pub dtype: DType,
    pub file_paths: Vec<PathBuf>,
    pub model_type: ModelType,
    pub use_flash_attention: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ModelType {
    Falcon7b,
    Falcon40b,
    Falcon180b,
    LlamaV1,
    LlamaV2,
    LlamaSolar10_7B,
    LlamaTinyLlama1_1BChat,
    Llama3_8b,
    Llama3Instruct8b,
    Llama3_70b,
    Mamba130m,
    Mamba370m,
    Mamba790m,
    Mamba1_4b,
    Mamba2_8b,
    Mistral7bV01,
    Mistral7bV02,
    Mistral7bInstructV01,
    Mistral7bInstructV02,
    Mixtral8x7b,
    Phi3Mini,
    StableDiffusionV1_5,
    StableDiffusionV2_1,
    StableDiffusionXl,
    StableDiffusionTurbo,
    // Quantized models
    QuantizedLlamaV2_7b,
    QuantizedLlamaV2_13b,
    QuantizedLlamaV2_70b,
    QuantizedLlamaV2_7bChat,
    QuantizedLlamaV2_13bChat,
    QuantizedLlamaV2_70bChat,
    QuantizedLlama7b,
    QuantizedLlama13b,
    QuantizedLlama34b,
    QuantizedLeo7b,
    QuantizedLeo13b,
    QuantizedMistral7b,
    QuantizedMistral7bInstruct,
    QuantizedMistral7bInstructV02,
    QuantizedZephyr7bAlpha,
    QuantizedZephyr7bBeta,
    QuantizedOpenChat35,
    QuantizedStarling7bAlpha,
    QuantizedMixtral,
    QuantizedMixtralInstruct,
    QuantizedL8b,
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
            "llama3_8b" => Ok(Self::Llama3_8b),
            "llama3_instruct_8b" => Ok(Self::Llama3Instruct8b),
            "llama3_70b" => Ok(Self::Llama3_70b),
            "mamba_130m" => Ok(Self::Mamba130m),
            "mamba_370m" => Ok(Self::Mamba370m),
            "mamba_790m" => Ok(Self::Mamba790m),
            "mamba_1-4b" => Ok(Self::Mamba1_4b),
            "mamba_2-8b" => Ok(Self::Mamba2_8b),
            "mistral_7bv01" => Ok(Self::Mistral7bV01),
            "mistral_7bv02" => Ok(Self::Mistral7bV02),
            "mistral_7b-instruct-v01" => Ok(Self::Mistral7bInstructV01),
            "mistral_7b-instruct-v02" => Ok(Self::Mistral7bInstructV02),
            "mixtral_8x7b" => Ok(Self::Mixtral8x7b),
            "phi_3-mini" => Ok(Self::Phi3Mini),
            "stable_diffusion_v1-5" => Ok(Self::StableDiffusionV1_5),
            "stable_diffusion_v2-1" => Ok(Self::StableDiffusionV2_1),
            "stable_diffusion_xl" => Ok(Self::StableDiffusionXl),
            "stable_diffusion_turbo" => Ok(Self::StableDiffusionTurbo),
            "quantized_7b" => Ok(Self::QuantizedLlamaV2_7b),
            "quantized_13b" => Ok(Self::QuantizedLlamaV2_13b),
            "quantized_70b" => Ok(Self::QuantizedLlamaV2_70b),
            "quantized_7b-chat" => Ok(Self::QuantizedLlamaV2_7bChat),
            "quantized_13b-chat" => Ok(Self::QuantizedLlamaV2_13bChat),
            "quantized_70b-chat" => Ok(Self::QuantizedLlamaV2_70bChat),
            "quantized_7b-code" => Ok(Self::QuantizedLlama7b),
            "quantized_13b-code" => Ok(Self::QuantizedLlama13b),
            "quantized_32b-code" => Ok(Self::QuantizedLlama34b),
            "quantized_7b-leo" => Ok(Self::QuantizedLeo7b),
            "quantized_13b-leo" => Ok(Self::QuantizedLeo13b),
            "quantized_7b-mistral" => Ok(Self::QuantizedMistral7b),
            "quantized_7b-mistral-instruct" => Ok(Self::QuantizedMistral7bInstruct),
            "quantized_7b-mistral-instruct-v0.2" => Ok(Self::QuantizedMistral7bInstructV02),
            "quantized_7b-zephyr-a" => Ok(Self::QuantizedZephyr7bAlpha),
            "quantized_7b-zephyr-b" => Ok(Self::QuantizedZephyr7bBeta),
            "quantized_7b-open-chat-3.5" => Ok(Self::QuantizedOpenChat35),
            "quantized_7b-starling-a" => Ok(Self::QuantizedStarling7bAlpha),
            "quantized_mixtral" => Ok(Self::QuantizedMixtral),
            "quantized_mixtral-instruct" => Ok(Self::QuantizedMixtralInstruct),
            "quantized_llama3-8b" => Ok(Self::QuantizedL8b),
            _ => Err(ModelError::InvalidModelType(
                "Invalid string model type description".to_string(),
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
            Self::Llama3_8b => "meta-llama/Meta-Llama-3-8B",
            Self::Llama3Instruct8b => "meta-llama/Meta-Llama-3-8B-Instruct",
            Self::Llama3_70b => "meta-llama/Meta-Llama-3-70B",
            Self::Mamba130m => "state-spaces/mamba-130m",
            Self::Mamba370m => "state-spaces/mamba-370m",
            Self::Mamba790m => "state-spaces/mamba-790m",
            Self::Mamba1_4b => "state-spaces/mamba-1.4b",
            Self::Mamba2_8b => "state-spaces/mamba-2.8b",
            Self::Mistral7bV01 => "mistralai/Mistral-7B-v0.1",
            Self::Mistral7bV02 => "mistralai/Mistral-7B-v0.2",
            Self::Mistral7bInstructV01 => "mistralai/Mistral-7B-Instruct-v0.1",
            Self::Mistral7bInstructV02 => "mistralai/Mistral-7B-Instruct-v0.2",
            Self::Mixtral8x7b => "mistralai/Mixtral-8x7B-v0.1",
            Self::Phi3Mini => "microsoft/Phi-3-mini-4k-instruct",
            Self::StableDiffusionV1_5 => "runwayml/stable-diffusion-v1-5",
            Self::StableDiffusionV2_1 => "stabilityai/stable-diffusion-2-1",
            Self::StableDiffusionXl => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::StableDiffusionTurbo => "stabilityai/sdxl-turbo",
            Self::QuantizedLlamaV2_7b => "TheBloke/Llama-2-7B-GGML",
            Self::QuantizedLlamaV2_13b => "TheBloke/Llama-2-13B-GGML",
            Self::QuantizedLlamaV2_70b => "TheBloke/Llama-2-70B-GGML",
            Self::QuantizedLlamaV2_7bChat => "TheBloke/Llama-2-7B-Chat-GGML",
            Self::QuantizedLlamaV2_13bChat => "TheBloke/Llama-2-13B-Chat-GGML",
            Self::QuantizedLlamaV2_70bChat => "TheBloke/Llama-2-70B-Chat-GGML",
            Self::QuantizedLlama7b => "TheBloke/CodeLlama-7B-GGUF",
            Self::QuantizedLlama13b => "TheBloke/CodeLlama-13B-GGUF",
            Self::QuantizedLlama34b => "TheBloke/CodeLlama-34B-GGUF",
            Self::QuantizedLeo7b => "TheBloke/leo-hessianai-7B-GGUF",
            Self::QuantizedLeo13b => "TheBloke/leo-hessianai-13B-GGUF",
            Self::QuantizedMistral7b => "TheBloke/Mistral-7B-v0.1-GGUF",
            Self::QuantizedMistral7bInstruct => "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
            Self::QuantizedMistral7bInstructV02 => "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            Self::QuantizedZephyr7bAlpha => "TheBloke/zephyr-7B-alpha-GGUF",
            Self::QuantizedZephyr7bBeta => "TheBloke/zephyr-7B-beta-GGUF",
            Self::QuantizedOpenChat35 => "TheBloke/openchat_3.5-GGUF",
            Self::QuantizedStarling7bAlpha => "TheBloke/Starling-LM-7B-alpha-GGUF",
            Self::QuantizedMixtral => "TheBloke/Mixtral-8x7B-v0.1-GGUF",
            Self::QuantizedMixtralInstruct => "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            Self::QuantizedL8b => "QuantFactory/Meta-Llama-3-8B-GGUF",
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
            Self::Llama3_8b => "main",
            Self::Llama3Instruct8b => "main",
            Self::Llama3_70b => "main",
            Self::Phi3Mini => "main",
            Self::Mamba130m => "refs/pr/1",
            Self::Mamba370m => "refs/pr/1",
            Self::Mamba790m => "refs/pr/1",
            Self::Mamba1_4b => "refs/pr/1",
            Self::Mamba2_8b => "refs/pr/4",
            Self::Mistral7bV01 => "main",
            Self::Mistral7bV02 => "main",
            Self::Mistral7bInstructV01 => "main",
            Self::Mistral7bInstructV02 => "main",
            Self::Mixtral8x7b => "main",
            Self::QuantizedL8b
            | Self::QuantizedLeo13b
            | Self::QuantizedLeo7b
            | Self::QuantizedLlama13b
            | Self::QuantizedLlama34b
            | Self::QuantizedLlama7b
            | Self::QuantizedLlamaV2_13b
            | Self::QuantizedLlamaV2_13bChat
            | Self::QuantizedLlamaV2_70b
            | Self::QuantizedLlamaV2_70bChat
            | Self::QuantizedLlamaV2_7b
            | Self::QuantizedLlamaV2_7bChat
            | Self::QuantizedMistral7b
            | Self::QuantizedMistral7bInstruct
            | Self::QuantizedMistral7bInstructV02
            | Self::QuantizedMixtral
            | Self::QuantizedMixtralInstruct
            | Self::QuantizedOpenChat35
            | Self::QuantizedStarling7bAlpha
            | Self::QuantizedZephyr7bAlpha
            | Self::QuantizedZephyr7bBeta
            | Self::StableDiffusionTurbo
            | Self::StableDiffusionV1_5
            | Self::StableDiffusionV2_1
            | Self::StableDiffusionXl => "",
        }
    }
}

impl Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Falcon7b => write!(f, "falcon_7b"),
            Self::Falcon40b => write!(f, "falcon_40b"),
            Self::Falcon180b => write!(f, "falcon_180b"),
            Self::LlamaV1 => write!(f, "llama_v1"),
            Self::LlamaV2 => write!(f, "llama_v2"),
            Self::LlamaSolar10_7B => write!(f, "llama_solar_10_7b"),
            Self::LlamaTinyLlama1_1BChat => write!(f, "llama_tiny_llama_1_1b_chat"),
            Self::Llama3_8b => write!(f, "llama3_8b"),
            Self::Llama3Instruct8b => write!(f, "llama3_instruct_8b"),
            Self::Llama3_70b => write!(f, "llama3_70b"),
            Self::Mamba130m => write!(f, "mamba_130m"),
            Self::Mamba370m => write!(f, "mamba_370m"),
            Self::Mamba790m => write!(f, "mamba_790m"),
            Self::Mamba1_4b => write!(f, "mamba_1-4b"),
            Self::Mamba2_8b => write!(f, "mamba_2-8b"),
            Self::Mistral7bV01 => write!(f, "mistral_7bv01"),
            Self::Mistral7bV02 => write!(f, "mistral_7bv02"),
            Self::Mistral7bInstructV01 => write!(f, "mistral_7b-instruct-v01"),
            Self::Mistral7bInstructV02 => write!(f, "mistral_7b-instruct-v02"),
            Self::Mixtral8x7b => write!(f, "mixtral_8x7b"),
            Self::Phi3Mini => write!(f, "phi_3-mini"),
            Self::StableDiffusionV1_5 => write!(f, "stable_diffusion_v1-5"),
            Self::StableDiffusionV2_1 => write!(f, "stable_diffusion_v2-1"),
            Self::StableDiffusionXl => write!(f, "stable_diffusion_xl"),
            Self::StableDiffusionTurbo => write!(f, "stable_diffusion_turbo"),
            Self::QuantizedLlamaV2_7b => write!(f, "quantized_7b"),
            Self::QuantizedLlamaV2_13b => write!(f, "quantized_13b"),
            Self::QuantizedLlamaV2_70b => write!(f, "quantized_70b"),
            Self::QuantizedLlamaV2_7bChat => write!(f, "quantized_7b-chat"),
            Self::QuantizedLlamaV2_13bChat => write!(f, "quantized_13b-chat"),
            Self::QuantizedLlamaV2_70bChat => write!(f, "quantized_70b-chat"),
            Self::QuantizedLlama7b => write!(f, "quantized_7b-code"),
            Self::QuantizedLlama13b => write!(f, "quantized_13b-code"),
            Self::QuantizedLlama34b => write!(f, "quantized_32b-code"),
            Self::QuantizedLeo7b => write!(f, "quantized_7b-leo"),
            Self::QuantizedLeo13b => write!(f, "quantized_13b-leo"),
            Self::QuantizedMistral7b => write!(f, "quantized_7b-mistral"),
            Self::QuantizedMistral7bInstruct => write!(f, "quantized_7b-mistral-instruct"),
            Self::QuantizedMistral7bInstructV02 => write!(f, "quantized_7b-mistral-instruct-v0.2"),
            Self::QuantizedZephyr7bAlpha => write!(f, "quantized_7b-zephyr-a"),
            Self::QuantizedZephyr7bBeta => write!(f, "quantized_7b-zephyr-b"),
            Self::QuantizedOpenChat35 => write!(f, "quantized_7b-open-chat-3.5"),
            Self::QuantizedStarling7bAlpha => write!(f, "quantized_7b-starling-a"),
            Self::QuantizedMixtral => write!(f, "quantized_mixtral"),
            Self::QuantizedMixtralInstruct => write!(f, "quantized_mixtral-instruct"),
            Self::QuantizedL8b => write!(f, "quantized_llama3-8b"),
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
    pub sampled_nodes: Vec<Vec<u8>>,
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

    fn requested_model(&self) -> ModelId {
        self.model.clone()
    }
}

#[derive(Debug, Deserialize)]
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

    pub sampled_nodes: Vec<Vec<u8>>,
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
