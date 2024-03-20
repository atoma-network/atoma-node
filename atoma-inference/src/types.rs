use ed25519_consensus::VerificationKey;

#[derive(Clone, Debug)]
pub struct Prompt(pub(crate) String);

#[derive(Clone, Debug)]
pub enum Model {
    Llama2(usize),
    Mamba(usize),
    Mixtral8x7b,
    Mistral(usize),
    StableDiffusionV1,
    StableDiffusionV2,
    StableDiffusionV3,
}

impl Model {
    pub fn to_string(&self) -> String {
        match self {
            Self::Llama2(size) => format!("Llama2({})", size),
            Self::Mamba(size) => format!("Mamba({})", size),
            Self::Mixtral8x7b => String::from("Mixtral8x7b"),
            Self::Mistral(size) => format!("Mistral({})", size),
            Self::StableDiffusionV1 => String::from("StableDiffusionV1"),
            Self::StableDiffusionV2 => String::from("StableDiffusionV2"),
            Self::StableDiffusionV3 => String::from("StableDiffusionV3"),
        }
    }
}

pub type NodeId = VerificationKey;
pub type Temperature = f32;

#[derive(Clone, Debug)]
pub struct InferenceRequest {
    pub(crate) prompt: Prompt,
    pub(crate) model: Model,
    pub(crate) max_tokens: usize,
    pub(crate) sampled_nodes: Vec<NodeId>,
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
}

#[derive(Clone, Debug)]
pub struct InferenceResponse {
    // TODO: possibly a Merkle root hash
    pub(crate) response_hash: [u8; 32],
    pub(crate) node_id: NodeId,
    pub(crate) signature: Vec<u8>,
    pub(crate) response: String,
}

#[derive(Clone, Debug)]
pub struct ModelRequest {
    pub(crate) model: Model,
    pub(crate) quantization_method: Option<QuantizationMethod>,
}

pub struct ModelResponse {
    pub(crate) is_success: bool,
}

#[derive(Clone, Debug)]
pub enum QuantizationMethod {
    Ggml(QuantizationBits),
    Gptq(QuantizationBits),
}

#[derive(Clone, Debug)]
pub enum QuantizationBits {
    Q1,
    Q2,
    Q4,
    Q5,
    Q8,
}
