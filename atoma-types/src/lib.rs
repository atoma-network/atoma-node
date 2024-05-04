use anyhow::{anyhow, Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub type Digest = [u8; 32];
pub type SmallId = u64;

/// Represents a request object containing information about a request
/// Prompt event, emitted on a given blockchain, see https://github.com/atoma-network/atoma-contracts/blob/main/sui/packages/atoma/sources/gate.move#L45.
/// It includes information about a ticket ID, sampled nodes, and request parameters.
///
/// Fields:
/// id: Vec<u8> - The ticket ID associated with the request (or event).
/// sampled_nodes: Vec<SmallId> - A vector of sampled nodes, each represented by a SmallId structure.
/// body: serde_json::Value - JSON value containing request parameters.
#[derive(Clone, Debug, Deserialize)]
pub struct Request {
    #[serde(rename(deserialize = "ticket_id"))]
    id: Vec<u8>,
    #[serde(rename(deserialize = "nodes"))]
    sampled_nodes: Vec<SmallId>,
    params: PromptParams,
}

impl Request {
    pub fn new(id: Vec<u8>, sampled_nodes: Vec<SmallId>, params: PromptParams) -> Self {
        Self {
            id,
            sampled_nodes,
            params,
        }
    }

    pub fn id(&self) -> Vec<u8> {
        self.id.clone()
    }

    pub fn model(&self) -> String {
        self.params.model()
    }

    pub fn sampled_nodes(&self) -> Vec<SmallId> {
        self.sampled_nodes.clone()
    }

    pub fn params(&self) -> PromptParams {
        self.params.clone()
    }
}

impl TryFrom<Value> for Request {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let id = hex::decode(
            value["ticket_id"]
                .as_str()
                .ok_or(anyhow!("Failed to decode hex string for request ticket_id"))?
                .replace("0x", ""),
        )?;
        let sampled_nodes = value["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| utils::parse_u64(&v["inner"]))
            .collect::<Result<Vec<_>>>()?;
        let body = PromptParams::try_from(value["params"].clone())?;
        Ok(Request::new(id, sampled_nodes, body))
    }
}

/// Enum encapsulating possible modal prompt params. Including both
/// - Text to text prompt parameters;
/// - Text to image prompt parameters.
#[derive(Clone, Debug, Deserialize)]
pub enum PromptParams {
    Text2TextPromptParams(Text2TextPromptParams),
    Text2ImagePromptParams(Text2ImagePromptParams),
}

impl PromptParams {
    pub fn model(&self) -> String {
        match self {
            Self::Text2ImagePromptParams(p) => p.model(),
            Self::Text2TextPromptParams(p) => p.model(),
        }
    }

    /// Extracts a `Text2TextPromptParams` from a `PromptParams` enum, or None
    /// if `PromptParams` does not correspond to `PromptParams::Text2TextPromptParams`
    pub fn into_text2text_prompt_params(self) -> Option<Text2TextPromptParams> {
        match self {
            Self::Text2TextPromptParams(p) => Some(p),
            Self::Text2ImagePromptParams(_) => None,
        }
    }

    // Extracts a `Text2ImagePromptParams` from a `PromptParams` enum, or None
    /// if `PromptParams` does not correspond to `PromptParams::Text2ImagePromptParams`
    pub fn into_text2image_prompt_params(self) -> Option<Text2ImagePromptParams> {
        match self {
            Self::Text2ImagePromptParams(p) => Some(p),
            Self::Text2TextPromptParams(_) => None,
        }
    }
}

impl TryFrom<Value> for PromptParams {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if value["temperature"].is_null() {
            Ok(Self::Text2ImagePromptParams(
                Text2ImagePromptParams::try_from(value)?,
            ))
        } else {
            Ok(Self::Text2TextPromptParams(
                Text2TextPromptParams::try_from(value)?,
            ))
        }
    }
}

/// Text to text prompt parameters. It includes:
/// - prompt: Prompt to be passed to model as input;
/// - model: Name of the model;
/// - temperature: parameter to control creativity of model
/// - random_seed: seed parameter for sampling
/// - repeat penalty: parameter to penalize token repetition (it should be >= 1.0)
/// - repeat last n: parameter to penalize last `n` token repetition
/// - top_k: parameter controlling `k` top tokens for sampling
/// - top_p: parameter controlling probabilities for top tokens
#[derive(Clone, Debug, Deserialize)]
pub struct Text2TextPromptParams {
    prompt: String,
    model: String,
    temperature: f64,
    random_seed: u64,
    repeat_penalty: f32,
    repeat_last_n: u64,
    max_tokens: u64,
    top_k: Option<u64>,
    top_p: Option<f64>,
}

impl Text2TextPromptParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        prompt: String,
        model: String,
        temperature: f64,
        random_seed: u64,
        repeat_penalty: f32,
        repeat_last_n: u64,
        max_tokens: u64,
        top_k: Option<u64>,
        top_p: Option<f64>,
    ) -> Self {
        Self {
            prompt,
            model,
            temperature,
            random_seed,
            repeat_penalty,
            repeat_last_n,
            max_tokens,
            top_k,
            top_p,
        }
    }

    pub fn prompt(&self) -> String {
        self.prompt.clone()
    }

    pub fn model(&self) -> String {
        self.model.clone()
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    pub fn random_seed(&self) -> u64 {
        self.random_seed
    }

    pub fn repeat_penalty(&self) -> f32 {
        self.repeat_penalty
    }

    pub fn repeat_last_n(&self) -> u64 {
        self.repeat_last_n
    }

    pub fn max_tokens(&self) -> u64 {
        self.max_tokens
    }

    pub fn top_k(&self) -> Option<u64> {
        self.top_k
    }

    pub fn top_p(&self) -> Option<f64> {
        self.top_p
    }
}

impl TryFrom<Value> for Text2TextPromptParams {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Ok(Self {
            prompt: utils::parse_str(&value["prompt"])?,
            model: utils::parse_str(&value["model"])?,
            temperature: utils::parse_f32_from_le_bytes(&value["temperature"])? as f64,
            random_seed: utils::parse_u64(&value["random_seed"])?,
            repeat_penalty: utils::parse_f32_from_le_bytes(&value["repeat_penalty"])?,
            repeat_last_n: utils::parse_u64(&value["repeat_last_n"])?,
            max_tokens: utils::parse_u64(&value["max_tokens"])?,
            top_k: Some(utils::parse_u64(&value["top_k"])?),
            top_p: Some(utils::parse_f32_from_le_bytes(&value["top_p"])? as f64),
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
/// Text to image prompt parameters. It includes:
/// - prompt: prompt to be passed to model as input;
/// - model: name of the model;
/// - uncond_prompt: unconditional prompt;
/// - height: output image height;
/// - width: output image width;
/// - n_steps: The number of steps to run the diffusion for;
/// - num_samples: number of samples to generate;
/// - guidance_scale:
/// - img2img: generate new AI images from an input image and text prompt.
///   The output image will follow the color and composition of the input image;
/// - img2img_strength: the strength, indicates how much to transform the initial image. The
///   value must be between 0 and 1, a value of 1 discards the initial image
///   information;
/// - random_seed: the seed to use when generating random samples.
pub struct Text2ImagePromptParams {
    prompt: String,
    model: String,
    uncond_prompt: String,
    height: Option<u64>,
    width: Option<u64>,
    n_steps: Option<u64>,
    num_samples: u64,
    guidance_scale: Option<f64>,
    img2img: Option<String>,
    img2img_strength: f64,
    random_seed: Option<u64>,
}

impl Text2ImagePromptParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        prompt: String,
        model: String,
        uncond_prompt: String,
        height: Option<u64>,
        width: Option<u64>,
        n_steps: Option<u64>,
        num_samples: u64,
        guidance_scale: Option<f64>,
        img2img: Option<String>,
        img2img_strength: f64,
        random_seed: Option<u64>,
    ) -> Self {
        Self {
            prompt,
            model,
            uncond_prompt,
            height,
            width,
            n_steps,
            num_samples,
            guidance_scale,
            img2img,
            img2img_strength,
            random_seed,
        }
    }

    pub fn prompt(&self) -> String {
        self.prompt.clone()
    }

    pub fn model(&self) -> String {
        self.model.clone()
    }

    pub fn uncond_prompt(&self) -> String {
        self.uncond_prompt.clone()
    }

    pub fn height(&self) -> Option<u64> {
        self.height
    }

    pub fn width(&self) -> Option<u64> {
        self.width
    }

    pub fn n_steps(&self) -> Option<u64> {
        self.n_steps
    }

    pub fn num_samples(&self) -> u64 {
        self.num_samples
    }

    pub fn guidance_scale(&self) -> Option<f64> {
        self.guidance_scale
    }

    pub fn img2img(&self) -> Option<String> {
        self.img2img.clone()
    }

    pub fn img2img_strength(&self) -> f64 {
        self.img2img_strength
    }

    pub fn random_seed(&self) -> Option<u64> {
        self.random_seed
    }
}

impl TryFrom<Value> for Text2ImagePromptParams {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Ok(Self {
            prompt: utils::parse_str(&value["prompt"])?,
            model: utils::parse_str(&value["model"])?,
            uncond_prompt: utils::parse_str(&value["uncond_prompt"])?,
            random_seed: Some(utils::parse_u64(&value["random_seed"])?),
            height: Some(utils::parse_u64(&value["height"])?),
            width: Some(utils::parse_u64(&value["width"])?),
            n_steps: Some(utils::parse_u64(&value["n_steps"])?),
            num_samples: utils::parse_u64(&value["num_samples"])?,
            guidance_scale: Some(utils::parse_f32_from_le_bytes(&value["guidance_scale"])? as f64),
            img2img: Some(utils::parse_str(&value["img2img"])?),
            img2img_strength: utils::parse_f32_from_le_bytes(&value["img2img2_strength"])? as f64,
        })
    }
}

/// Represents a response object containing information about a response, including an ID, sampled nodes, and the response data.
///
/// Fields:
/// id: Vec<u8> - The ticket id associated with the request, that lead to the generation of this response.
/// sampled_nodes: Vec<SmallId> - A vector of sampled nodes, each represented by a SmallId structure.
/// response: serde_json::Value - JSON value containing the response data.
#[derive(Debug, Deserialize, Serialize)]
pub struct Response {
    id: Vec<u8>,
    sampled_nodes: Vec<SmallId>,
    response: Value,
}

impl Response {
    pub fn new(id: Vec<u8>, sampled_nodes: Vec<SmallId>, response: Value) -> Self {
        Self {
            id,
            sampled_nodes,
            response,
        }
    }

    pub fn id(&self) -> Vec<u8> {
        self.id.clone()
    }

    pub fn sampled_nodes(&self) -> Vec<SmallId> {
        self.sampled_nodes.clone()
    }

    pub fn response(&self) -> Value {
        self.response.clone()
    }
}

mod utils {
    use super::*;

    /// Parses an appropriate JSON value, from a number (represented as a `u32`) to a `f32` type, by
    /// representing the extracted u32 value into little endian byte representation, and then applying `f32::from_le_bytes`.  
    /// See https://github.com/atoma-network/atoma-contracts/blob/main/sui/packages/atoma/sources/gate.move#L26
    pub(crate) fn parse_f32_from_le_bytes(value: &Value) -> Result<f32> {
        let u32_value: u32 = value
            .as_u64()
            .ok_or(anyhow!(
                "Failed to extract `f32` little endian bytes representation"
            ))?
            .try_into()?;
        let f32_le_bytes = u32_value.to_le_bytes();
        Ok(f32::from_le_bytes(f32_le_bytes))
    }

    /// Parses an appropriate JSON value, representing a `u64` number, from a Sui
    /// `Text2TextPromptEvent` `u64` fields.
    pub(crate) fn parse_u64(value: &Value) -> Result<u64> {
        value
            .as_str()
            .ok_or(anyhow!("Failed to extract `u64` number"))?
            .parse::<u64>()
            .map_err(|e| anyhow!("Failed to parse `u64` from string, with error: {e}"))
    }

    /// Parses an appropriate JSON value, representing a `String` value, from a Sui
    /// `Text2TextPromptEvent` `String` fields.
    pub(crate) fn parse_str(value: &Value) -> Result<String> {
        Ok(value
            .as_str()
            .ok_or(anyhow!("Failed to extract `String` from JSON value"))?
            .to_string())
    }
}
