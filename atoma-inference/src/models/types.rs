use std::{fmt::Display, path::PathBuf, str::FromStr};

use atoma_types::{Digest, PromptParams};
use candle::{DType, Device};
use serde::{Deserialize, Serialize};

use super::ModelError;

#[derive(Debug)]
/// `LlmLoadData` - encapsulates data necessary to load a
/// large language model in memory.
pub struct LlmLoadData {
    /// The `Device` in which the model is hosted.
    /// Likely to be a GPU card.
    pub device: Device,
    /// The `DType`, representing the decimal
    /// precision in which the model is supposed to run
    pub dtype: DType,
    /// Vector of all the downloaded model weights file paths
    pub file_paths: Vec<PathBuf>,
    /// The model type, to identify the model (e.g. Llama3-8b)
    pub model_type: ModelType,
    /// Parameter specifying if inference should run on highly
    /// attention multiplication, through the flash attention algorithm
    /// See https://arxiv.org/abs/2205.14135 and https://arxiv.org/abs/2307.08691
    /// for more details, on the algorithm
    pub use_flash_attention: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// A well specified (unique) model identifier
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
    QwenW0_5b,
    QwenW1_8b,
    QwenW4b,
    QwenW7b,
    QwenW14b,
    QwenW72b,
    QwenMoeA27b,
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
            "qwen_w0.5b" => Ok(Self::QwenW0_5b),
            "qwen_w1.8b" => Ok(Self::QwenW1_8b),
            "qwen_w4b" => Ok(Self::QwenW4b),
            "qwen_w7b" => Ok(Self::QwenW7b),
            "qwen_w14b" => Ok(Self::QwenW14b),
            "qwen_w72b" => Ok(Self::QwenW72b),
            "qwen_moe_a2.7b" => Ok(Self::QwenMoeA27b),
            _ => Err(ModelError::InvalidModelType(
                "Invalid string model type description".to_string(),
            )),
        }
    }
}

impl ModelType {
    /// Extracts the HuggingFace API repository
    /// holding the model with given `ModelType`
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
            Self::QwenW0_5b => "Qwen/Qwen1.5-0.5B",
            Self::QwenW1_8b => "Qwen/Qwen1.5-1.8B",
            Self::QwenW4b => "Qwen/Qwen1.5-4B",
            Self::QwenW7b => "Qwen/Qwen1.5-7B",
            Self::QwenW14b => "Qwen/Qwen1.5-14B",
            Self::QwenW72b => "Qwen/Qwen1.5-72B",
            Self::QwenMoeA27b => "Qwen/Qwen1.5-MoE-A2.7B",
        }
    }

    /// Outputs the HF revision for each model
    pub fn default_revision(&self) -> &'static str {
        match self {
            Self::Falcon7b => "refs/pr/43",
            Self::Falcon40b => "refs/pr/43",
            Self::Falcon180b => "refs/pr/43",
            Self::LlamaV1
            | Self::LlamaV2
            | Self::LlamaSolar10_7B
            | Self::LlamaTinyLlama1_1BChat
            | Self::Llama3_8b
            | Self::Llama3Instruct8b
            | Self::Llama3_70b
            | Self::Mistral7bV01
            | Self::Mistral7bV02
            | Self::Mistral7bInstructV01
            | Self::Mistral7bInstructV02
            | Self::Mixtral8x7b
            | Self::Phi3Mini
            | Self::QwenW0_5b
            | Self::QwenW1_8b
            | Self::QwenW4b
            | Self::QwenW7b
            | Self::QwenW14b
            | Self::QwenW72b
            | Self::QwenMoeA27b => "main",
            Self::Mamba130m => "refs/pr/1",
            Self::Mamba370m => "refs/pr/1",
            Self::Mamba790m => "refs/pr/1",
            Self::Mamba1_4b => "refs/pr/1",
            Self::Mamba2_8b => "refs/pr/4",
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
            Self::QwenW0_5b => write!(f, "qwen_w0.5b"),
            Self::QwenW1_8b => write!(f, "qwen_w1.8b"),
            Self::QwenW4b => write!(f, "qwen_w4b"),
            Self::QwenW7b => write!(f, "qwen_w7b"),
            Self::QwenW14b => write!(f, "qwen_w14b"),
            Self::QwenW72b => write!(f, "qwen_w72b"),
            Self::QwenMoeA27b => write!(f, "qwen_moe_a2.7b"),
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

#[derive(Clone, Debug, Deserialize)]
/// `TextModelInput` - used to specify all prompt
/// parameters used for running LLM inference on a given
/// text prompt. It should only be consumed by text to
/// text models, and not multi-modality models
pub struct TextModelInput {
    /// The request id's. This is likely to come from
    /// the Atoma's smart contract (on some blockchain).
    /// In that case, is a unique identifier in 32-byte
    /// format
    pub(crate) request_id: Digest,
    /// The actual prompt text (to be used as input
    /// on the model's inference run)
    pub(crate) prompt: String,
    /// Temperature - controls the expressivity of the output
    /// generated by the model
    pub(crate) temperature: f64,
    /// Random seed, used in sampling next tokens, out of a set
    /// of most likely tokens (specified by the model's generated
    /// logits tensor)
    pub(crate) random_seed: u64,
    /// Repeat penalty, contains a float point (>= 1.0) which is
    /// used to penalize repetition of previous works/tokens, in
    /// LLM output generation
    pub(crate) repeat_penalty: f32,
    /// Repeat last n - only allows the model to repeat the previous
    /// last n tokens generated already
    pub(crate) repeat_last_n: usize,
    /// Number of maximum tokens (think of words) that should be generated by
    /// the model. Therefore, the output should have at most `max_tokens`
    /// of token length (it can have less if the model hits `eos` token)
    pub(crate) max_tokens: usize,
    /// Top k words to be selected
    pub(crate) top_k: Option<usize>,
    /// Top logits higher than `p`
    pub(crate) top_p: Option<f64>,
    /// If the current request is part of a chat application, or not
    pub(crate) chat: bool,
    /// Pre-prompt tokens, specified in conversational applications (such
    /// as chats). In which, what was previously generated in the conversation
    /// is kept for future predictions
    pub(crate) pre_prompt_tokens: Vec<u32>,
    /// Boolean specifying if the output of the inference run should be
    /// streamed to the end user (most applicable to chat use cases)
    pub(crate) should_stream_output: bool,
}

impl TextModelInput {
    #[allow(clippy::too_many_arguments)]
    /// Constructor
    pub fn new(
        request_id: String,
        prompt: String,
        temperature: f64,
        random_seed: u64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        max_tokens: usize,
        top_k: Option<usize>,
        top_p: Option<f64>,
        chat: bool,
        pre_prompt_tokens: Vec<u32>,
        should_stream_output: bool,
    ) -> Self {
        Self {
            request_id,
            prompt,
            temperature,
            random_seed,
            repeat_penalty,
            repeat_last_n,
            max_tokens,
            top_k,
            top_p,
            chat,
            pre_prompt_tokens,
            should_stream_output,
        }
    }
}

impl TryFrom<(Digest, PromptParams)> for TextModelInput {
    type Error = ModelError;

    fn try_from((request_id, value): (Digest, PromptParams)) -> Result<Self, Self::Error> {
        match value {
            PromptParams::Text2TextPromptParams(p) => Ok(Self {
                request_id,
                prompt: p.prompt(),
                temperature: p.temperature(),
                random_seed: p.random_seed(),
                repeat_penalty: p.repeat_penalty(),
                repeat_last_n: p.repeat_last_n().try_into().unwrap(),
                max_tokens: p.max_tokens().try_into().unwrap(),
                top_k: p.top_k().map(|t| t.try_into().unwrap()),
                top_p: p.top_p(),
                chat: p.is_chat(),
                pre_prompt_tokens: p.pre_prompt_tokens(),
                should_stream_output: p.should_stream_output(),
            }),
            PromptParams::Text2ImagePromptParams(_) => Err(ModelError::InvalidModelInput),
        }
    }
}

#[derive(Serialize, Debug)]
/// `TextModelOutput` - Encapsulates the actual AI generated output, for a given
/// request. It contains additional metadata about the generation that is relevant
/// to keep track. To be used in the context of text to text models and not
/// multi modality ones
pub struct TextModelOutput {
    /// Number of input tokens for the request
    pub input_tokens: usize,
    /// The actual AI generated text output
    pub text: String,
    /// The token identifiers, corresponding to the
    /// generated text
    pub tokens: Vec<u32>,
    /// The duration, in seconds, of the entire inference
    /// run
    pub time: f64,
    /// Number of the output tokens, that were generated
    /// by the model inference
    pub tokens_count: usize,
}

impl Display for TextModelOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Output: {}\nInput tokens: {}\nTime: {}\nTokens count: {}",
            self.text, self.input_tokens, self.time, self.tokens_count
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// `StableDiffusionRequest` - to be used as input to
/// stable diffusion models, which are used to generate
/// images from input text
pub struct StableDiffusionInput {
    /// The actual text prompt for image generation
    pub prompt: String,
    /// Unconditional prompt, to give more context
    /// on the image details to generate
    pub uncond_prompt: String,
    /// The height of the final image to generate
    pub height: Option<usize>,
    /// The width of the final image to generate
    pub width: Option<usize>,
    /// The number of steps to run the diffusion for.
    pub n_steps: Option<usize>,
    /// The number of samples to generate.
    pub num_samples: i64,
    /// Model identifier
    pub model: ModelType,
    /// Guidance scale, optional
    pub guidance_scale: Option<f64>,
    /// Image to image, to be used if one aims to
    /// transform the initial generated image in a given
    /// specific way
    pub img2img: Option<String>,
    /// The strength, indicates how much to transform the initial image. The
    /// value must be between 0 and 1, a value of 1 discards the initial image
    /// information.
    pub img2img_strength: f64,
    /// The seed to use when generating random samples.
    pub random_seed: Option<u32>,
}

impl TryFrom<(Digest, PromptParams)> for StableDiffusionInput {
    type Error = ModelError;

    fn try_from((_, value): (Digest, PromptParams)) -> Result<Self, Self::Error> {
        match value {
            PromptParams::Text2ImagePromptParams(p) => Ok(Self {
                prompt: p.prompt(),
                uncond_prompt: p.uncond_prompt(),
                height: p.height().map(|t| t.try_into().unwrap()),
                width: p.width().map(|t| t.try_into().unwrap()),
                n_steps: p.n_steps().map(|t| t.try_into().unwrap()),
                num_samples: p.num_samples() as i64,
                model: ModelType::from_str(&p.model())?,
                guidance_scale: p.guidance_scale(),
                img2img: p.img2img(),
                img2img_strength: p.img2img_strength(),
                random_seed: p.random_seed(),
            }),
            _ => Err(ModelError::InvalidModelInput),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
/// `StableDiffusionResponse` - encapsulates the output
/// of a Stable diffusion model
pub struct StableDiffusionResponse {
    /// The actual image byte buffer, it should be formatted
    /// with height x width
    pub output: Vec<(Vec<u8>, usize, usize)>,
    /// It's true if the inference was performed correctly
    pub is_success: bool,
}
