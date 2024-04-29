use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
pub struct SmallId {
    inner: u64,
}

impl SmallId {
    pub fn new(inner: u64) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> u64 {
        self.inner
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Request {
    id: Vec<u8>,
    #[serde(rename = "nodes")]
    sampled_nodes: Vec<SmallId>,
    #[serde(rename = "params")]
    body: Value,
}

impl Request {
    pub fn new(id: Vec<u8>, sampled_nodes: Vec<SmallId>, body: Value) -> Self {
        Self {
            id,
            sampled_nodes,
            body,
        }
    }

    pub fn id(&self) -> Vec<u8> {
        self.id.clone()
    }

    pub fn model(&self) -> String {
        self.body["model"].as_str().unwrap().to_string()
    }

    pub fn sampled_nodes(&self) -> Vec<SmallId> {
        self.sampled_nodes.clone()
    }

    pub fn body(&self) -> Value {
        self.body.clone()
    }
}

#[derive(Debug)]
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
