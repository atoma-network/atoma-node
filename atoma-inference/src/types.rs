use ed25519_consensus::VerificationKey;

#[derive(Clone, Debug)]
pub struct Prompt(String);

#[derive(Clone, Debug)]
pub enum Model {
    Llama2(usize),
    Mixtral8x7b,
    Mistral(usize),
    StableDiffusionV1,
    StableDiffusionV2,
    StableDiffusionV3,
}

#[derive(Clone, Debug)]
pub struct NodeId(VerificationKey);

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

#[derive(Clone, Debug)]
pub struct ModelResponse { 
    pub(crate) is_success: bool,
}

#[derive(Clone, Debug)]
pub enum QuantizationMethod {
    Ggml,
    Gptq(),
}

#[derive(Clone, Debug)]
pub enum QuantizationBits {
    Q1,
    Q2,
    Q4,
    Q5,
    Q8,
}
