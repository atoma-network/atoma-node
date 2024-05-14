use tokio::sync::mpsc;

use atoma_types::Digest;

use crate::{bail, models::ModelError};

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
    stream_tx: mpsc::Sender<(Digest, String)>,
}

impl TokenOutputStream {
    pub fn new(
        tokenizer: tokenizers::Tokenizer,
        stream_tx: mpsc::Sender<(Digest, String)>,
    ) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
            stream_tx,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, ModelError> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => bail!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(
        &mut self,
        token: u32,
        request_id: Option<Digest>,
    ) -> Result<Option<String>, ModelError> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            let output = text.1.to_string();
            if let Some(digest) = request_id {
                self.stream_tx
                    .blocking_send((digest, output.clone()))
                    .map_err(ModelError::SendError)?;
            }
            Ok(Some(output))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self, request_id: Option<Digest>) -> Result<Option<String>, ModelError> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            let output = text.1.to_string();
            if let Some(digest) = request_id {
                self.stream_tx
                    .blocking_send((digest, output.clone()))
                    .map_err(ModelError::SendError)?;
            }
            Ok(Some(output))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> Result<String, ModelError> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn get_num_generated_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}
