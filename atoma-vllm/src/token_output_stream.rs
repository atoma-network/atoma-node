// /// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
// /// streaming way rather than having to wait for the full decoding.
// pub struct TokenOutputStream {
//     /// The actual model's `Tokenizer` (responsible for mapping each token/word to
//     /// an `u32` identifier, i.e., tokenization).
//     /// Optional for testing
//     tokenizer: Option<tokenizers::Tokenizer>,
//     /// Vector of tokens
//     tokens: Vec<u32>,
//     /// Previous considered index
//     prev_index: usize,
//     /// The current index, in the stream,
//     current_index: usize,
// }

// impl TokenOutputStream {
//     /// Constructor
//     pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
//         Self {
//             tokenizer: Some(tokenizer),
//             tokens: Vec::new(),
//             prev_index: 0,
//             current_index: 0,
//         }
//     }

//     /// Creates a new empty instance of `Self` from a `Tokenizer`,
//     /// only for testing purposes
//     pub fn empty() -> Self {
//         Self {
//             tokenizer: None,
//             tokens: Vec::new(),
//             prev_index: 0,
//             current_index: 0,
//         }
//     }

//     /// Outputs the underlying `Tokenizer`, consuming `self`
//     pub fn into_inner(self) -> tokenizers::Tokenizer {
//         self.tokenizer.unwrap()
//     }

//     /// Tries to decode a slice of tokens (in `u32` format) to an actual `String`
//     fn decode(&self, tokens: &[u32]) -> Result<String, TokenOutputStreamError> {
//         match self.tokenizer.unwrap().decode(tokens, true) {
//             Ok(str) => Ok(str),
//             Err(err) => TokenOutputStreamError::DecodeError(String::from("cannot decode: {err}")),
//         }
//     }

//     /// Predicts the next `String` predicted by the LLM, through an inference run and
//     /// sampling. More details can be found in
//     /// https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
//     ///
//     /// It further sends the next predicted token to a streaming service (currently, the Atoma streaming service),
//     /// if streaming is requested
//     pub fn next_token(&mut self, token: u32) -> Result<Option<String>, TokenOutputStreamError> {
//         let prev_text = if self.tokens.is_empty() {
//             String::new()
//         } else {
//             let tokens = &self.tokens[self.prev_index..self.current_index];
//             self.decode(tokens)?
//         };
//         self.tokens.push(token);
//         let text = self.decode(&self.tokens[self.prev_index..])?;
//         if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
//             let text = text.split_at(prev_text.len());
//             self.prev_index = self.current_index;
//             self.current_index = self.tokens.len();
//             let output = text.1.to_string();
//             Ok(Some(output))
//         } else {
//             Ok(None)
//         }
//     }

//     /// Tries to decode the rest of the `String`, in
//     pub fn decode_rest(&self) -> Result<Option<String>, TokenOutputStreamError> {
//         let prev_text = if self.tokens.is_empty() {
//             String::new()
//         } else {
//             let tokens = &self.tokens[self.prev_index..self.current_index];
//             self.decode(tokens)?
//         };
//         let text = self.decode(&self.tokens[self.prev_index..])?;
//         if text.len() > prev_text.len() {
//             let text = text.split_at(prev_text.len());
//             let output = text.1.to_string();
//             Ok(Some(output))
//         } else {
//             Ok(None)
//         }
//     }

//     /// Decodes all the available tokens
//     pub fn decode_all(&self) -> Result<String, TokenOutputStreamError> {
//         self.decode(&self.tokens)
//     }

//     /// Get token id, from the `Tokenizer` vocabulary
//     pub fn get_token(&self, token_s: &str) -> Option<u32> {
//         self.tokenizer
//             .unwrap()
//             .get_vocab(true)
//             .get(token_s)
//             .copied()
//     }

//     /// Outputs a reference to the underlying `Tokenizer`
//     pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
//         &self.tokenizer.unwrap()
//     }

//     /// Get number of generated tokens, so far
//     pub fn get_num_generated_tokens(&self) -> usize {
//         self.tokens.len()
//     }

//     /// Clears the current `self.tokens` being held
//     pub fn clear(&mut self) {
//         self.tokens.clear();
//         self.prev_index = 0;
//         self.current_index = 0;
//     }
// }

// #[derive(Debug, Error)]
// pub enum TokenOutputStreamError {
//     #[error("Send error: `{0}`")]
//     DecodeError(String),
// }
