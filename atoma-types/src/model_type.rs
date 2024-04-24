use std::{fmt::Display, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::ModelError;

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
            _ => "",
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

impl ModelType {
    pub fn unet_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::StableDiffusionV1_5
            | Self::StableDiffusionV2_1
            | Self::StableDiffusionXl
            | Self::StableDiffusionTurbo => {
                if use_f16 {
                    "unet/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "unet/diffusion_pytorch_model.safetensors"
                }
            }
            _ => panic!("Invalid stable diffusion model type"),
        }
    }

    pub fn vae_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::StableDiffusionV1_5
            | Self::StableDiffusionV2_1
            | Self::StableDiffusionXl
            | Self::StableDiffusionTurbo => {
                if use_f16 {
                    "vae/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "vae/diffusion_pytorch_model.safetensors"
                }
            }
            _ => panic!("Invalid stable diffusion model type"),
        }
    }

    pub fn clip_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::StableDiffusionV1_5
            | Self::StableDiffusionV2_1
            | Self::StableDiffusionXl
            | Self::StableDiffusionTurbo => {
                if use_f16 {
                    "text_encoder/model.fp16.safetensors"
                } else {
                    "text_encoder/model.safetensors"
                }
            }
            _ => panic!("Invalid stable diffusion model type"),
        }
    }

    pub fn clip2_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::StableDiffusionV1_5
            | Self::StableDiffusionV2_1
            | Self::StableDiffusionXl
            | Self::StableDiffusionTurbo => {
                if use_f16 {
                    "text_encoder_2/model.fp16.safetensors"
                } else {
                    "text_encoder_2/model.safetensors"
                }
            }
            _ => panic!("Invalid stable diffusion model type"),
        }
    }
}
