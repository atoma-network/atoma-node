use serde::Deserialize;
use serde_json::Value;
mod errors;
mod model_type;
pub use errors::*;
pub use model_type::*;

#[derive(Clone, Debug, Deserialize)]
pub struct Request {
    id: usize,
    sampled_nodes: Vec<u64>,
    model: String,
    body: Value,
}

impl Request {
    pub fn new(id: usize, sampled_nodes: Vec<u64>, model: String, body: Value) -> Self {
        Self {
            id,
            sampled_nodes,
            model,
            body,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn sampled_nodes(&self) -> Vec<u64> {
        self.sampled_nodes.clone()
    }

    pub fn model(&self) -> String {
        self.model.clone()
    }

    pub fn body(&self) -> Value {
        self.body.clone()
    }
}

#[derive(Debug)]
pub struct Response {
    id: usize,
    sampled_nodes: Vec<u64>,
    response: Value,
}

impl Response {
    pub fn new(id: usize, sampled_nodes: Vec<u64>, response: Value) -> Self {
        Self {
            id,
            sampled_nodes,
            response,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn sampled_nodes(&self) -> Vec<u64> {
        self.sampled_nodes.clone()
    }

    pub fn response(&self) -> Value {
        self.response.clone()
    }
}
