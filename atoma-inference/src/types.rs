use crate::models::ModelType;
use ed25519_consensus::VerificationKey;

pub type NodeId = VerificationKey;
pub type Temperature = f32;

#[derive(Clone, Debug)]
pub struct Prompt(pub(crate) String);

#[derive(Clone, Debug)]
pub struct InferenceRequest {
    pub(crate) prompt: Prompt,
    pub(crate) model: ModelType,
    pub(crate) max_tokens: usize,
    pub(crate) random_seed: usize,
    pub(crate) repeat_penalty: f32,
    pub(crate) sampled_nodes: Vec<NodeId>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_k: usize,
    pub(crate) top_p: Option<f32>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct InferenceResponse {
    // TODO: possibly a Merkle root hash
    pub(crate) response_hash: [u8; 32],
    pub(crate) node_id: NodeId,
    pub(crate) signature: Vec<u8>,
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
