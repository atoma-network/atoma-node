use anyhow::{Error, Result};
use serde::Deserialize;
use serde_json::{Value, json};

pub type SmallId = u64;

#[derive(Clone, Debug, Deserialize)]
pub struct Request {
    #[serde(rename(deserialize = "ticket_id"))]
    id: Vec<u8>,
    #[serde(rename(deserialize = "nodes"))]
    sampled_nodes: Vec<SmallId>,
    #[serde(rename(deserialize = "params"))]
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

impl TryFrom<Value> for Request { 
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let id = hex::decode(value["ticket_id"].as_str().unwrap().replace("0x", ""))?;
    let sampled_nodes = value["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| parse_u64(&v["inner"]))
        .collect::<Vec<_>>();
    let body = parse_body(value["params"].clone())?;
    Ok(Request::new(id, sampled_nodes, body))
    }
}

fn parse_body(json: Value) -> Result<Value> {
    let output = json!({
            "max_tokens": parse_u64(&json["max_tokens"]),
            "model": json["model"],
            "prompt": json["prompt"],
            "random_seed": parse_u64(&json["random_seed"]),
            "repeat_last_n": parse_u64(&json["repeat_last_n"]),
            "repeat_penalty": parse_f32(&json["repeat_penalty"]),
            "temperature": parse_f32(&json["temperature"]),
            "top_k": parse_u64(&json["top_k"]),
            "top_p": parse_f32(&json["top_p"]),
    });
    Ok(output)
}

fn parse_f32(value: &Value) -> f32 {
    f32::from_be_bytes((value.as_u64().unwrap() as u32).to_be_bytes())
}

fn parse_u64(value: &Value) -> u64 {
    value.as_str().unwrap().parse::<u64>().unwrap()
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
