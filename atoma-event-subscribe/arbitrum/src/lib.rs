use serde::Deserialize;

pub mod subscriber;

#[derive(Debug, Deserialize)]
pub struct AtomaEvent {
    pub model: String
}
