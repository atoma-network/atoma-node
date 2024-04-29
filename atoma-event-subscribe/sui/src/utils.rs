use atoma_types::{Request, SmallId};
use serde_json::{json, Value};

use crate::subscriber::SuiSubscriberError;

pub(crate) fn try_from_json(json: Value) -> Result<Request, SuiSubscriberError> {
    let id = hex::decode(json["ticket_id"].as_str().unwrap().replace("0x", ""))?;
    let sampled_nodes = json["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| SmallId::new(v["inner"].as_str().unwrap().parse::<u64>().unwrap()))
        .collect::<Vec<_>>();
    let body = parse_body(json["params"].clone())?;
    Ok(Request::new(id, sampled_nodes, body))
}

fn parse_body(json: Value) -> Result<Value, SuiSubscriberError> {
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
