use anyhow::{anyhow, Error, Result};
use serde::Deserialize;
use serde_json::{json, Value};

pub type SmallId = u64;

/// Represents a request object containing information about a request
/// Prompt event, emitted on a given blockchain, see https://github.com/atoma-network/atoma-contracts/blob/main/sui/packages/atoma/sources/gate.move#L45.
/// It includes information about a ticket ID, sampled nodes, and request parameters.
///
/// Fields:
/// id: Vec<u8> - The ticket ID associated with the request (or event).
/// sampled_nodes: Vec<SmallId> - A vector of sampled nodes, each represented by a SmallId structure.
/// body: serde_json::Value - JSON value containing request parameters.
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
        let id = hex::decode(
            value["ticket_id"]
                .as_str()
                .ok_or(anyhow!("Failed to decode hex string for request ticket_id"))?
                .replace("0x", ""),
        )?;
        let sampled_nodes = value["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| parse_u64(&v["inner"]))
            .collect::<Result<Vec<_>>>()?;
        let body = parse_body(value["params"].clone())?;
        Ok(Request::new(id, sampled_nodes, body))
    }
}

/// Parses the body of a JSON value. This JSON value is supposed to be obtained
/// from a Sui `Text2TextPromptEvent`,
/// see https://github.com/atoma-network/atoma-contracts/blob/main/sui/packages/atoma/sources/gate.move#L28
fn parse_body(json: Value) -> Result<Value> {
    let output = json!({
            "max_tokens": parse_u64(&json["max_tokens"])?,
            "model": json["model"],
            "prompt": json["prompt"],
            "random_seed": parse_u64(&json["random_seed"])?,
            "repeat_last_n": parse_u64(&json["repeat_last_n"])?,
            "repeat_penalty": parse_f32_from_le_bytes(&json["repeat_penalty"])?,
            "temperature": parse_f32_from_le_bytes(&json["temperature"])?,
            "top_k": parse_u64(&json["top_k"])?,
            "top_p": parse_f32_from_le_bytes(&json["top_p"])?,
    });
    Ok(output)
}

/// Parses an appropriate JSON value, from a number (represented as a `u32`) to a `f32` type, by
/// representing the extracted u32 value into little endian byte representation, and then applying `f32::from_le_bytes`.  
/// See https://github.com/atoma-network/atoma-contracts/blob/main/sui/packages/atoma/sources/gate.move#L26
fn parse_f32_from_le_bytes(value: &Value) -> Result<f32> {
    let u32_value: u32 = value
        .as_u64()
        .ok_or(anyhow!(
            "Failed to extract `f32` little endian bytes representation"
        ))?
        .try_into()?;
    let f32_le_bytes = u32_value.to_le_bytes();
    Ok(f32::from_le_bytes(f32_le_bytes))
}

/// Parses an appropriate JSON value, representing a `u64` number, from a Sui
/// `Text2TextPromptEvent` `u64` fields.
fn parse_u64(value: &Value) -> Result<u64> {
    value
        .as_str()
        .ok_or(anyhow!("Failed to extract `u64` number"))?
        .parse::<u64>()
        .map_err(|e| anyhow!("Failed to parse `u64` from string, with error: {e}"))
}

/// Represents a response object containing information about a response, including an ID, sampled nodes, and the response data.
///
/// Fields:
/// id: Vec<u8> - The ticket id associated with the request, that lead to the generation of this response.
/// sampled_nodes: Vec<SmallId> - A vector of sampled nodes, each represented by a SmallId structure.
/// response: serde_json::Value - JSON value containing the response data.
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
