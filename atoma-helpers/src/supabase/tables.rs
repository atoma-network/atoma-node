use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct PromptsRow {
    pub prompt: String,
    pub previous_tx: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TokensRow {
    pub tokens: Vec<u32>,
}
