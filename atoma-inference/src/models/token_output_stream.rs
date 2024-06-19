use tokio::sync::mpsc;

use atoma_types::Digest;

use crate::{bail, models::ModelError};

const END_STREAM: &str = "[ATOMA_END_STREAM]";

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    /// The actual model's `Tokenizer` (responsible for mapping each token/word to
    /// an `u32` identifier, i.e., tokenization)
    tokenizer: tokenizers::Tokenizer,
    /// Vector of tokens
    tokens: Vec<u32>,
    /// Previous considered index
    prev_index: usize,
    /// The current index, in the stream,
    current_index: usize,
    /// A `mpsc` channel responsible to stream each newly generated token
    /// to some external streaming service (currently Atoma's stream service)
    stream_tx: mpsc::Sender<(Digest, String)>,
}

impl TokenOutputStream {
    /// Constructor
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

    /// Outputs the underlying `Tokenizer`, consuming `self`
    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    /// Tries to decode a slice of tokens (in `u32` format) to an actual `String`
    fn decode(&self, tokens: &[u32]) -> Result<String, ModelError> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => bail!("cannot decode: {err}"),
        }
    }

    /// Predicts the next `String` predicted by the LLM, through an inference run and
    /// sampling. More details can be found in
    /// https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    ///
    /// It further sends the next predicted token to a streaming service (currently, the Atoma streaming service),
    /// if streaming is requested
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

    /// Tries to decode the rest of the `String`, in
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

    /// Decodes all the available tokens
    pub fn decode_all(&self) -> Result<String, ModelError> {
        self.decode(&self.tokens)
    }

    /// Get token id, from the `Tokenizer` vocabulary
    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    /// Outputs a reference to the underlying `Tokenizer`
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Get number of generated tokens, so far
    pub fn get_num_generated_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Clears the current `self.tokens` being held
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }

    /// Ends the stream, through a special value, encapsulated in `END_STREAM`
    pub fn end_stream(&self, tx_digest: Digest) -> Result<(), ModelError> {
        self.stream_tx
            .blocking_send((tx_digest, END_STREAM.to_string()))
            .map_err(ModelError::SendError)
    }
}
