use crate::models::ModelType;
use candle::DType;
use ed25519_consensus::VerificationKey;
use serde::Deserialize;

pub type NodeId = VerificationKey;
pub type Temperature = f32;

#[derive(Clone, Debug)]
pub struct InferenceRequest {
    pub request_id: u128,
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
pub enum QuantizationMethod {
    Ggml(PrecisionBits),
    Gptq(PrecisionBits),
}

#[derive(Copy, Clone, Debug, Deserialize)]
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
