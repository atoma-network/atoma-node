use ed25519_consensus::VerificationKey as PublicKey;
use serde::{Deserialize, Serialize};

use crate::models::{ModelId, Request, Response};

pub type NodeId = PublicKey;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TextRequest {
    pub request_id: usize,
    pub prompt: String,
    pub model: ModelId,
    pub max_tokens: usize,
    pub random_seed: usize,
    pub repeat_last_n: usize,
    pub repeat_penalty: f32,
    pub sampled_nodes: Vec<NodeId>,
    pub temperature: Option<f32>,
    pub top_k: usize,
    pub top_p: Option<f32>,
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
            self.top_p.unwrap_or_default() as f64,
        )
    }

    fn request_id(&self) -> usize {
        self.request_id
    }

    fn is_node_authorized(&self, public_key: &PublicKey) -> bool {
        self.sampled_nodes.contains(&public_key)
    }

    fn requested_model(&self) -> ModelId {
        self.model.clone()
    }
}

pub struct TextModelInput {
    pub(crate) prompt: String,
    pub(crate) temperature: f64,
    pub(crate) random_seed: u64,
    pub(crate) repeat_penalty: f32,
    pub(crate) repeat_last_n: usize,
    pub(crate) max_tokens: usize,
    pub(crate) _top_k: usize,
    pub(crate) top_p: f64,
}

impl TextModelInput {
    pub fn new(
        prompt: String,
        temperature: f64,
        random_seed: u64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        max_tokens: usize,
        _top_k: usize,
        top_p: f64,
    ) -> Self {
        Self {
            prompt,
            temperature,
            random_seed,
            repeat_penalty,
            repeat_last_n,
            max_tokens,
            _top_k,
            top_p,
        }
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
