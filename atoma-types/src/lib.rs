use anyhow::{anyhow, Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const NON_SAMPLED_NODE_ERR: &str = "Node has not been selected";
pub type Digest = String;
pub type RequestId = Vec<u8>;
pub type SmallId = u64;

/// Represents a request object containing information about a request
/// Prompt event, emitted on a given blockchain, see https://github.com/atoma-network/atoma-contracts/blob/main/sui/packages/atoma/sources/gate.move#L45.
/// It includes information about a ticket ID, sampled nodes, and request parameters.
///
/// Fields:
///     `id`: RequestId - The ticket ID associated with the request (or event).
///     `sampled_node_index`: usize - Current node id in the list of sampled nodes.
///         This value should not be optional, as a request is only processed if a node has been selected, to begin with.
///     `num_sampled_nodes`: usize - The total number of sampled nodes to process this request.
///     `params`: ModelParams - Contains the relevant data to run the requested AI inference, including
///         which model to be used, the prompt input, temperature, etc.
///     `output_destination`: Vec<u8> - The output destination, serialized into a byte buffer.
#[derive(Clone, Debug, Deserialize)]
pub struct Request {
    /// The request id, which corresponds to the ticket id,
    /// emitted from the Atoma smart contract.
    id: RequestId,
    /// Current node index in the quorum of all sampled nodes, selected
    /// to process the current request.
    sampled_node_index: usize,
    /// Number of total sampled nodes that form the quorum to run
    /// the current inference request
    num_sampled_nodes: usize,
    /// Prompt parameters
    params: ModelParams,
    /// The output destination, in a byte representation. The node
    /// should be able to deserialize it into actual metadata
    /// specifying how to store the actual output
    output_destination: Vec<u8>,
}

impl Request {
    /// Constructor
    pub fn new(
        id: RequestId,
        sampled_node_index: usize,
        num_sampled_nodes: usize,
        params: ModelParams,
        output_destination: Vec<u8>,
    ) -> Self {
        Self {
            id,
            sampled_node_index,
            num_sampled_nodes,
            params,
            output_destination,
        }
    }

    /// Getter for `id`
    pub fn id(&self) -> RequestId {
        self.id.clone()
    }

    /// Getter for `model`
    pub fn model(&self) -> String {
        self.params.model()
    }

    /// Getter for `sampled_node_index`
    pub fn sampled_node_index(&self) -> usize {
        self.sampled_node_index
    }

    /// Getter for `num_sampled_nodes`
    pub fn num_sampled_nodes(&self) -> usize {
        self.num_sampled_nodes
    }

    /// Getter for `params`
    pub fn params(&self) -> ModelParams {
        self.params.clone()
    }

    /// Getter for `output_destination`
    pub fn output_destination(&self) -> Vec<u8> {
        self.output_destination.clone()
    }

    /// Once we get the prompt, we can set it as raw for the inference
    pub fn set_raw_prompt(&mut self, prompt: String) {
        self.params.set_raw_prompt(prompt);
    }

    /// Once we get an image,
    pub fn set_image_path(&mut self, path: String) {
        self.params.set_image_path(path);
    }

    /// Set preprompts for the request
    pub fn set_preprompt_tokens(&mut self, pre_prompt_tokens: Vec<u32>) {
        self.params.set_preprompt_tokens(pre_prompt_tokens);
    }

    /// Image to image request
    pub fn set_raw_image(&mut self, image_bytes: Vec<u8>) {
        self.params.set_raw_image(image_bytes);
    }
}

impl TryFrom<(u64, Value)> for Request {
    type Error = Error;

    fn try_from((node_id, value): (u64, Value)) -> Result<Self, Self::Error> {
        let id = hex::decode(
            value["ticket_id"]
                .as_str()
                .ok_or(anyhow!("Failed to decode hex string for request ticket_id"))?
                .replace("0x", ""),
        )?;
        let sampled_nodes = value["nodes"]
            .as_array()
            .ok_or(anyhow!("Request is malformed, missing `nodes` field"))?
            .iter()
            .map(|v| utils::parse_u64(&v["inner"]))
            .collect::<Result<Vec<_>>>()?;
        let sampled_node_index = sampled_nodes
            .iter()
            .position(|n| n == &node_id)
            .ok_or(anyhow!(NON_SAMPLED_NODE_ERR))?;
        let num_sampled_nodes = sampled_nodes.len();
        let params = ModelParams::try_from(value["params"].clone())?;
        let output_destination = value["output_destination"]
            .as_array()
            .ok_or(anyhow!(
                "Request malformed, missing `output_destination` field"
            ))?
            .iter()
            .map(|x| {
                x.as_u64()
                    .map(|b| b as u8)
                    .ok_or(anyhow!("Invalid byte value for output destination"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Request::new(
            id,
            sampled_node_index,
            num_sampled_nodes,
            params,
            output_destination,
        ))
    }
}

impl TryFrom<(&str, usize, Value)> for Request {
    type Error = Error;

    fn try_from(
        (ticket_id, sampled_node_index, value): (&str, usize, Value),
    ) -> Result<Self, Self::Error> {
        let id = hex::decode(ticket_id.replace("0x", ""))?;
        let num_sampled_nodes = value
            .get("sampled_nodes")
            .ok_or(anyhow!("missing `sampled_nodes` field",))?
            .as_array()
            .ok_or(anyhow!("invalid `sampled_nodes` field",))?
            .len();

        let prompt_params_value = value
            .get("params")
            .ok_or(anyhow!("invalid `params` field",))?;
        let prompt_params = ModelParams::try_from(prompt_params_value.clone())?;

        let output_destination = value["output_destination"]
            .as_array()
            .ok_or(anyhow!(
                "Request malformed, missing `output_destination` field"
            ))?
            .iter()
            .map(|x| {
                x.as_u64()
                    .map(|b| b as u8)
                    .ok_or(anyhow!("Invalid byte value for output destination"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Request::new(
            id,
            sampled_node_index,
            num_sampled_nodes,
            prompt_params,
            output_destination,
        ))
    }
}

/// Enum encapsulating possible modal prompt params. Including both
/// - Text to text prompt parameters;
/// - Text to image prompt parameters.
#[derive(Clone, Debug, Deserialize)]
pub enum ModelParams {
    Text2TextModelParams(Text2TextModelParams),
    Text2ImageModelParams(Text2ImageModelParams),
}

impl ModelParams {
    /// Gets the model associated to the current instance of `Self`
    pub fn model(&self) -> String {
        match self {
            Self::Text2ImageModelParams(p) => p.model(),
            Self::Text2TextModelParams(p) => p.model(),
        }
    }

    /// Extracts a `Text2TextModelParams` from a `ModelParams` enum, or None
    /// if `ModelParams` does not correspond to `ModelParams::Text2TextModelParams`
    pub fn into_text2text_prompt_params(self) -> Option<Text2TextModelParams> {
        match self {
            Self::Text2TextModelParams(p) => Some(p),
            Self::Text2ImageModelParams(_) => None,
        }
    }

    // Extracts a `Text2ImageModelParams` from a `ModelParams` enum, or None
    /// if `ModelParams` does not correspond to `ModelParams::Text2ImageModelParams`
    pub fn into_text2image_prompt_params(self) -> Option<Text2ImageModelParams> {
        match self {
            Self::Text2ImageModelParams(p) => Some(p),
            Self::Text2TextModelParams(_) => None,
        }
    }

    pub fn prompt(&self) -> InputSource {
        match self {
            Self::Text2TextModelParams(p) => p.prompt(),
            Self::Text2ImageModelParams(p) => p.prompt(),
        }
    }

    pub fn get_input_text(&self) -> String {
        match self.prompt() {
            InputSource::Firebase { .. } => {
                unreachable!("Firebase request id found when raw prompt was expected")
            }
            InputSource::Ipfs { .. } => {
                unreachable!("IPFS request id found when raw prompt was expected")
            }
            InputSource::Raw { prompt } => prompt,
        }
    }

    /// Once we get the prompt, we can set it as raw for the inference
    pub fn set_raw_prompt(&mut self, prompt: String) {
        match self {
            Self::Text2TextModelParams(p) => p.set_raw_prompt(prompt),
            Self::Text2ImageModelParams(p) => p.set_raw_prompt(prompt),
        }
    }

    /// Set the image to image value for `Text2ImageModelParams` variant
    pub fn set_raw_image(&mut self, image_bytes: Vec<u8>) {
        match self {
            Self::Text2ImageModelParams(p) => p.set_raw_image(image_bytes),
            Self::Text2TextModelParams(_) => {
                unreachable!("Setting raw image for Text2TextModelParams is not allowed")
            }
        }
    }

    /// Sets the image path for `Text2ImageModelParams` variant
    pub fn set_image_path(&mut self, path: String) {
        match self {
            Self::Text2ImageModelParams(p) => p.set_image_path(path),
            Self::Text2TextModelParams(_) => {
                unreachable!("Setting image path for Text2TextModelParams is not allowed")
            }
        }
    }

    pub fn set_preprompt_tokens(&mut self, pre_prompt_tokens: Vec<u32>) {
        match self {
            Self::Text2TextModelParams(p) => p.set_preprompt_tokens(pre_prompt_tokens),
            Self::Text2ImageModelParams(_) => unimplemented!("Preprompt tokens are not supported"),
        }
    }
}

impl TryFrom<Value> for ModelParams {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if value["temperature"].is_null() {
            Ok(Self::Text2ImageModelParams(
                Text2ImageModelParams::try_from(value)?,
            ))
        } else {
            Ok(Self::Text2TextModelParams(Text2TextModelParams::try_from(
                value,
            )?))
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
/// - should_stream_output: boolean value used for streaming the response back to the User, on some UI
#[derive(Clone, Debug, Deserialize)]
pub struct Text2TextModelParams {
    prompt: InputSource,
    /// The specified model of the request
    model: String,
    /// The temperature
    temperature: f64,
    /// The random seed, used in sampling new tokens
    random_seed: u64,
    /// The repeat penalty
    repeat_penalty: f32,
    /// The repeat last n
    repeat_last_n: u64,
    /// Number of max tokens
    max_tokens: u64,
    /// Optional top k value
    top_k: Option<u64>,
    /// Optional top p value
    top_p: Option<f64>,
    /// Boolean that controls if the request is to be used in chat mode or not
    chat: bool,
    /// Pre prompt tokens, to be used solely in case `chat == true`
    pre_prompt_tokens: Vec<u32>,
    /// Should stream output, if true the node should stream the output
    /// back to the used (on the UI). Used only when `chat == true`
    should_stream_output: bool,
}

impl Text2TextModelParams {
    #[allow(clippy::too_many_arguments)]
    /// Constructor
    pub fn new(
        prompt: InputSource,
        model: String,
        temperature: f64,
        random_seed: u64,
        repeat_penalty: f32,
        repeat_last_n: u64,
        max_tokens: u64,
        top_k: Option<u64>,
        top_p: Option<f64>,
        chat: bool,
        pre_prompt_tokens: Vec<u32>,
        should_stream_output: bool,
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
            chat,
            pre_prompt_tokens,
            should_stream_output,
        }
    }

    /// Getter for `prompt`
    pub fn prompt(&self) -> InputSource {
        self.prompt.clone()
    }

    /// Getter for `model`
    pub fn model(&self) -> String {
        self.model.clone()
    }

    /// Getter for `temperature`
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Getter for `random_seed`
    pub fn random_seed(&self) -> u64 {
        self.random_seed
    }

    /// Getter for `repeat_penalty`
    pub fn repeat_penalty(&self) -> f32 {
        self.repeat_penalty
    }

    /// Getter for `repeat_last_n`
    pub fn repeat_last_n(&self) -> u64 {
        self.repeat_last_n
    }

    /// Getter for `max_tokens`
    pub fn max_tokens(&self) -> u64 {
        self.max_tokens
    }

    /// Getter for `top_k`
    pub fn top_k(&self) -> Option<u64> {
        self.top_k
    }

    /// Getter for `top_p`
    pub fn top_p(&self) -> Option<f64> {
        self.top_p
    }

    /// Getter for `should_stream_output`
    pub fn should_stream_output(&self) -> bool {
        self.should_stream_output
    }

    /// Getter for `chat`
    pub fn is_chat(&self) -> bool {
        self.chat
    }

    /// Getter for `pre_prompt_tokens`
    pub fn pre_prompt_tokens(&self) -> Vec<u32> {
        self.pre_prompt_tokens.clone()
    }

    pub fn set_raw_prompt(&mut self, prompt: String) {
        self.prompt = InputSource::Raw { prompt };
    }

    pub fn set_preprompt_tokens(&mut self, pre_prompt_tokens: Vec<u32>) {
        self.pre_prompt_tokens = pre_prompt_tokens;
    }

    pub fn get_input_text(&self) -> String {
        match &self.prompt {
            InputSource::Firebase { .. } => {
                unreachable!("Firebase request id found when raw prompt was expected")
            }
            InputSource::Ipfs { .. } => {
                unreachable!("IPFS request cid found when raw prompt was expected")
            }
            InputSource::Raw { prompt } => prompt.clone(),
        }
    }
}

impl TryFrom<Value> for Text2TextModelParams {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Ok(Self {
            prompt: Deserialize::deserialize(&mut rmp_serde::Deserializer::new(
                serde_json::from_value::<Vec<u8>>(value["prompt"].clone())?.as_slice(),
            ))?,
            model: utils::parse_str(&value["model"])?,
            temperature: utils::parse_f32_from_le_bytes(&value["temperature"])? as f64,
            random_seed: utils::parse_u64(&value["random_seed"])?,
            repeat_penalty: utils::parse_f32_from_le_bytes(&value["repeat_penalty"])?,
            repeat_last_n: utils::parse_u64(&value["repeat_last_n"])?,
            max_tokens: utils::parse_u64(&value["max_tokens"])?,
            top_k: Some(utils::parse_u64(&value["top_k"])?),
            top_p: Some(utils::parse_f32_from_le_bytes(&value["top_p"])? as f64),
            should_stream_output: utils::parse_bool(&value["should_stream_output"])?,
            chat: utils::parse_bool(&value["prepend_output_with_input"])?,
            pre_prompt_tokens: value["pre_prompt_tokens"]
                .as_array()
                .ok_or_else(|| anyhow!("Expected an array for pre_prompt_tokens"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or_else(|| anyhow!("Expected a u64 for pre_prompt_tokens"))
                        .map(|v| v as u32)
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

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
#[derive(Clone, Debug, Deserialize)]
pub struct Text2ImageModelParams {
    /// Prompt, in String format
    prompt: InputSource,
    /// Model to run the inference
    model: String,
    /// Unconditional prompt, used in stable diffusion models
    uncond_prompt: Option<String>,
    /// Height of the final generated image
    height: Option<u64>,
    /// Width of the final generated image
    width: Option<u64>,
    /// Number of n steps (optional)
    n_steps: Option<u64>,
    /// Number of samples, in the image generation
    num_samples: u64,
    /// Guidance scale (optional)
    guidance_scale: Option<f64>,
    /// Image to image (optional)
    img2img: Option<Vec<u8>>,
    /// Image path (optional)
    img_path: Option<String>,
    /// Image to image strength
    img2img_strength: f64,
    /// The random seed for inference sampling
    random_seed: Option<u32>,
    /// Only decode the image (applicable to Flux models)
    decode_only: Option<String>,
}

impl Text2ImageModelParams {
    #[allow(clippy::too_many_arguments)]
    /// Constructor
    pub fn new(
        prompt: InputSource,
        model: String,
        uncond_prompt: Option<String>,
        height: Option<u64>,
        width: Option<u64>,
        n_steps: Option<u64>,
        num_samples: u64,
        guidance_scale: Option<f64>,
        img2img: Option<Vec<u8>>,
        img2img_strength: f64,
        img_path: Option<String>,
        random_seed: Option<u32>,
        decode_only: Option<String>,
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
            img_path,
            random_seed,
            decode_only,
        }
    }

    /// Getter for `prompt`
    pub fn prompt(&self) -> InputSource {
        self.prompt.clone()
    }

    /// Getter for `model`
    pub fn model(&self) -> String {
        self.model.clone()
    }

    /// Getter for `uncond_prompt`
    pub fn uncond_prompt(&self) -> Option<String> {
        self.uncond_prompt.clone()
    }

    /// Getter for `height`
    pub fn height(&self) -> Option<u64> {
        self.height
    }

    /// Getter for `width`
    pub fn width(&self) -> Option<u64> {
        self.width
    }

    /// Getter for `n_steps`
    pub fn n_steps(&self) -> Option<u64> {
        self.n_steps
    }

    /// Getter for `num_samples`
    pub fn num_samples(&self) -> u64 {
        self.num_samples
    }

    /// Getter for `guidance_scale`
    pub fn guidance_scale(&self) -> Option<f64> {
        self.guidance_scale
    }

    /// Getter for `img2img`
    pub fn img2img(&self) -> Option<Vec<u8>> {
        self.img2img.clone()
    }

    /// Getter for `img2img_strength`
    pub fn img2img_strength(&self) -> f64 {
        self.img2img_strength
    }

    /// Getter for `random_seed`
    pub fn random_seed(&self) -> Option<u32> {
        self.random_seed
    }

    /// Once we get the prompt, we can set it as raw for the inference
    pub fn set_raw_prompt(&mut self, prompt: String) {
        self.prompt = InputSource::Raw { prompt };
    }

    /// Returns the input text for the prompt, if this prompt is a raw prompt
    pub fn get_input_text(&self) -> String {
        match &self.prompt {
            InputSource::Firebase { .. } => {
                unreachable!("Firebase request id found when raw prompt was expected")
            }
            InputSource::Ipfs { .. } => {
                unreachable!("IPFS request cid found when raw prompt was expected")
            }
            InputSource::Raw { prompt } => prompt.clone(),
        }
    }

    /// Getter for `decode_only`
    pub fn decode_only(&self) -> Option<String> {
        self.decode_only.clone()
    }

    /// Sets the image path for `Text2ImageModelParams` variant
    pub fn set_image_path(&mut self, path: String) {
        self.img_path = Some(path);
    }

    /// Sets the image to image value for `Text2ImageModelParams` variant
    pub fn set_raw_image(&mut self, image_bytes: Vec<u8>) {
        self.img2img = Some(image_bytes);
    }
}

impl TryFrom<Value> for Text2ImageModelParams {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Ok(Self {
            prompt: Deserialize::deserialize(&mut rmp_serde::Deserializer::new(
                serde_json::from_value::<Vec<u8>>(value["prompt"].clone())?.as_slice(),
            ))?,
            model: utils::parse_str(&value["model"])?,
            uncond_prompt: utils::parse_optional_str(&value["uncond_prompt"]),
            random_seed: Some(utils::parse_u32(&value["random_seed"])?),
            height: Some(utils::parse_u64(&value["height"])?),
            width: Some(utils::parse_u64(&value["width"])?),
            n_steps: Some(utils::parse_u64(&value["n_steps"])?),
            num_samples: utils::parse_u64(&value["num_samples"])?,
            guidance_scale: Some(utils::parse_f32_from_le_bytes(&value["guidance_scale"])? as f64),
            img2img: utils::parse_optional_bytes(&value["img2img"]),
            img2img_strength: utils::parse_f32_from_le_bytes(&value["img2img_strength"])? as f64,
            img_path: utils::parse_optional_str(&value["img_path"]),
            decode_only: utils::parse_optional_str(&value["decode_only"]),
        })
    }
}

/// Represents a response object containing information about a response, including an ID, sampled nodes, and the response data.
///
/// Fields:
///     `id`: RequestId - The ticket id associated with the request, that lead to the generation of this response.
///     `sampled_node_index`: usize - The node's index position in the request's original vector of sampled nodes.
///         This value should not be optional, as a node only processes a request if it was sampled to begin with.
///     `num_sampled_nodes`: usize - The total number of sampled nodes, in the original request.
///     `response`: serde_json::Value - JSON value containing the response data.
///     `output_destination`: Vec<u8> - Output destination metadata, in byte format.
#[derive(Debug, Deserialize, Serialize)]
pub struct Response {
    /// The request id, which corresponds to the Atoma's smart contract
    /// emitted ticket id.
    id: RequestId,
    /// The current node index in the quorum of sampled nodes to run
    /// the request, to which this response was generated from.
    sampled_node_index: usize,
    /// Number of sampled nodes in the quorum of sampled nodes.
    num_sampled_nodes: usize,
    /// The actual response output, in JSON format.
    response: Value,
    /// The output destination metadata, serialized as a byte buffer
    output_destination: Vec<u8>,
    /// Output type, e.g. `Text`, `Image`, etc
    output_type: OutputType,
}

impl Response {
    /// Constructor
    pub fn new(
        id: RequestId,
        sampled_node_index: usize,
        num_sampled_nodes: usize,
        response: Value,
        output_destination: Vec<u8>,
        output_type: OutputType,
    ) -> Self {
        Self {
            id,
            sampled_node_index,
            num_sampled_nodes,
            response,
            output_destination,
            output_type,
        }
    }

    /// Getter for `id`
    pub fn id(&self) -> RequestId {
        self.id.clone()
    }

    /// Getter for `sampled_node_index`
    pub fn sampled_node_index(&self) -> usize {
        self.sampled_node_index
    }

    /// Getter for `num_sampled_nodes`
    pub fn num_sampled_nodes(&self) -> usize {
        self.num_sampled_nodes
    }

    /// Getter for `response`
    pub fn response(&self) -> Value {
        self.response.clone()
    }

    /// Getter for `output_destination`
    pub fn output_destination(&self) -> &[u8] {
        self.output_destination.as_slice()
    }

    /// Getter for `output_type`
    pub fn output_type(&self) -> OutputType {
        self.output_type
    }
}

impl Response {
    /// Extracts the number of `input_tokens` from the request's prompt
    pub fn input_tokens(&self) -> u64 {
        self.response["input_tokens"].as_u64().unwrap_or(0)
    }

    /// Extracts the number of `output_tokens` from the generated output
    pub fn tokens_count(&self) -> u64 {
        self.response["tokens_count"].as_u64().unwrap_or(0)
    }

    /// The duration, in seconds, taken to generate the current output
    pub fn time_to_generate(&self) -> f64 {
        self.response["time"].as_f64().unwrap_or(0.0)
    }

    /// Tokens generated
    pub fn tokens(&self) -> Vec<u32> {
        self.response["tokens"]
            .as_array()
            .map(|v| v.iter().map(|t| t.as_u64().unwrap_or(0) as u32).collect())
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum InputFormat {
    Image,
    Text,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ModelInput {
    Chat((String, Vec<u32>)),
    ImageBytes(Vec<u8>),
    ImageFile(String),
    Text(String),
}

/// `InputSource` - Enum describing available input sources
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum InputSource {
    Firebase { request_id: String },
    Ipfs { cid: String, format: InputFormat },
    Raw { prompt: String }, // This means that the prompt is stored in the request
}

/// `OutputDestination` - enum encapsulating the output's destination
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum OutputDestination {
    Firebase { request_id: String },
    Ipfs,
    Gateway { gateway_user_id: String },
}

impl OutputDestination {
    /// Getter for `user_id`
    pub fn request_id(&self) -> String {
        match self {
            Self::Firebase { request_id } => request_id.clone(),
            Self::Ipfs => unimplemented!("IPFS output destination not implemented"),
            Self::Gateway { .. } => unimplemented!("Gateway user id not implemented"),
        }
    }
}

/// `OutputType` - enum encapsulating the output type (e.g. `Text`, `Image`, etc)
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum OutputType {
    Text,
    Image,
}

/// `OutputType` - enum encapsulating the output type (e.g. `Text`, `Image`, etc)
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum InputType {
    Text,
}

impl std::fmt::Display for OutputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputType::Text => write!(f, "Text"),
            OutputType::Image => write!(f, "Image"),
        }
    }
}

/// `AtomaOutputMetadata` - metadata associated with the output generated by a node,
///     for a given request
#[derive(Debug, Deserialize, Serialize)]
pub struct AtomaOutputMetadata {
    /// Node's own public key address, in hex format
    pub node_public_key: String,
    /// The ticket id associated with the request, in hex format
    pub ticket_id: String,
    /// Number of prompt input tokens
    pub num_input_tokens: u64,
    /// Number of generated output tokens
    pub num_output_tokens: u64,
    /// Time taken to generate the output, in seconds
    pub time_to_generate: f64,
    /// Byte representation of the output commitment, submitted on-chain
    pub commitment_root_hash: Vec<u8>,
    /// Number of sampled nodes to process the request
    pub num_sampled_nodes: usize,
    /// The index of the current node in the sampled quorum to process the output,
    /// it should be comprised between `[0, num_sampled_nodes)`
    pub index_of_node: usize,
    /// The leaf hash, submitted by the current node
    /// while providing the root_hash commitment on-chain
    pub leaf_hash: Vec<u8>,
    /// The transaction in base58 format (used in the Sui blockchain)
    pub transaction_base_58: String,
    /// The output destination
    pub output_destination: OutputDestination,
    /// The output type (e.g. `Text`, `Image`)
    pub output_type: OutputType,
    /// Tokens generated
    pub tokens: Vec<u32>,
}

mod utils {
    use super::*;

    /// Parses an appropriate JSON value, from a number (represented as a `u32`) to a `f32` type, by
    /// representing the extracted u32 value into little endian byte representation, and then applying `f32::from_le_bytes`.  
    /// See https://github.com/atoma-network/atoma-contracts/blob/main/sui/packages/atoma/sources/gate.move#L26
    pub(crate) fn parse_f32_from_le_bytes(value: &Value) -> Result<f32> {
        let u32_value: u32 = value
            .as_u64()
            .ok_or_else(|| anyhow!("Expected a u64 for f32 conversion, found none"))?
            .try_into()?;
        let f32_le_bytes = u32_value.to_le_bytes();
        Ok(f32::from_le_bytes(f32_le_bytes))
    }

    /// Parses an appropriate JSON value, representing a `u32` number, from a Sui
    /// `Text2ImagePromptEvent` or `Text2TextPromptEvent` `u32` fields.
    pub(crate) fn parse_u32(value: &Value) -> Result<u32> {
        value
            .as_u64()
            .ok_or_else(|| anyhow!("Expected a u64 for u32 parsing, found none"))
            .and_then(|v| {
                v.try_into()
                    .map_err(|_| anyhow!("u64 to u32 conversion failed"))
            })
    }

    /// Parses an appropriate JSON value, representing a `u64` number, from a Sui
    /// `Text2ImagePromptEvent` or `Text2TextPromptEvent` `u64` fields.
    pub(crate) fn parse_u64(value: &Value) -> Result<u64> {
        value
            .as_str()
            .ok_or_else(|| anyhow!("Expected a string for u64 parsing, found none"))
            .and_then(|s| {
                s.parse::<u64>()
                    .map_err(|e| anyhow!("Failed to parse u64: {}", e))
            })
    }

    /// Parses an appropriate JSON value, representing a `String` value, from a Sui
    /// `Text2TextPromptEvent` `String` fields.
    pub(crate) fn parse_str(value: &Value) -> Result<String> {
        value
            .as_str()
            .ok_or_else(|| anyhow!("Expected a string, found none"))
            .map(|s| s.to_string())
    }

    /// Util method that parses a suitable JSON value, to extracts
    /// an underlying string
    pub(crate) fn parse_optional_str(value: &Value) -> Option<String> {
        value.as_str().map(|s| s.to_string())
    }

    /// Util method that extract a boolean value from a JSON
    /// representing a boolean.
    pub(crate) fn parse_bool(value: &Value) -> Result<bool> {
        value
            .as_bool()
            .ok_or_else(|| anyhow!("Expected a bool, found none"))
    }

    /// Util method that extract an (optional) slice of bytes
    /// from a JSON value.
    pub(crate) fn parse_optional_bytes(value: &Value) -> Option<Vec<u8>> {
        value.as_array().and_then(|arr| {
            arr.iter()
                .map(|v| v.as_u64().map(|n| n as u8))
                .collect::<Option<Vec<u8>>>()
        })
    }
}

pub struct AtomaStreamingData {
    output_source_id: String,
    data: String,
}

impl AtomaStreamingData {
    pub fn new(output_source_id: String, data: String) -> Self {
        Self {
            output_source_id,
            data,
        }
    }

    pub fn data(&self) -> &String {
        &self.data
    }

    pub fn output_source_id(&self) -> &String {
        &self.output_source_id
    }
}
