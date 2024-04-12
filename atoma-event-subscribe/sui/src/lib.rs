use serde::Deserialize;

pub mod subscriber;

#[derive(Debug, Deserialize)]
pub struct TextPromptParams {
    pub model: String,
    pub prompt: String,
    pub max_tokens: String,
    /// Represents a floating point number between 0 and 1, big endian.
    pub temperature: u32,
}
