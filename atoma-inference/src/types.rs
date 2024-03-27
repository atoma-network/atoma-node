use crate::models::ModelType;
use ed25519_consensus::VerificationKey;

pub type NodeId = VerificationKey;
pub type Temperature = f32;

#[derive(Clone, Debug)]
pub struct InferenceRequest {
    pub prompt: String,
    pub model: ModelType,
    pub max_tokens: usize,
    pub random_seed: usize,
    pub repeat_last_n: usize,
    pub repeat_penalty: f32,
    pub sampled_nodes: Vec<NodeId>,
    pub temperature: Option<f32>,
    pub top_k: usize,
    pub top_p: Option<f32>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct InferenceResponse {
    // TODO: possibly a Merkle root hash
    // pub(crate) response_hash: [u8; 32],
    // pub(crate) node_id: NodeId,
    // pub(crate) signature: Vec<u8>,
    pub(crate) response: String,
}

#[derive(Clone, Debug)]
pub struct ModelRequest {
    pub(crate) model: ModelType,
    pub(crate) quantization_method: Option<QuantizationMethod>,
}

#[allow(dead_code)]
pub struct ModelResponse {
    pub(crate) is_success: bool,
    pub(crate) error: Option<String>,
}

#[derive(Clone, Debug)]
pub enum QuantizationMethod {
    Ggml(PrecisionBits),
    Gptq(PrecisionBits),
}

#[derive(Clone, Debug)]
pub enum PrecisionBits {
    Q1,
    Q2,
    Q4,
    Q5,
    Q8,
    F16,
    F32,
}
