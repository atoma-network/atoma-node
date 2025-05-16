use serde_json::Value;
use tokenizers::Tokenizer;

use crate::error::AtomaServiceError;

/// A struct that contains the estimated number of compute units needed for a
/// request and the maximum number of compute units that can be used for the request.
#[derive(Debug, Clone)]
pub struct TokensEstimate {
    /// The number of compute units needed for the input tokens
    pub num_input_tokens: u64,
    /// The number of compute units needed for the output tokens
    pub max_output_tokens: u64,
    /// The maximum total number of compute units that can be used for the request
    pub max_total_tokens: u64,
}

/// A trait for parsing and handling AI model requests across different endpoints (chat, embeddings, images).
///
/// This trait provides a common interface for processing various types of AI model requests
/// and estimating their computational costs.
pub trait RequestModel {
    /// Constructs a new request model instance by parsing the provided JSON request.
    ///
    /// # Arguments
    /// * `request` - The JSON payload containing the request parameters
    ///
    /// # Returns
    /// * `Ok(Self)` - Successfully parsed request model
    /// * `Err(AtomaProxyError)` - If the request is invalid or malformed
    ///
    /// # Errors
    /// Returns `AtomaServiceError` if the request is invalid or malformed
    fn new(request: &Value) -> Result<Self, AtomaServiceError>
    where
        Self: Sized;

    /// Calculates the estimated computational resources required for this request.
    ///
    /// # Arguments
    /// * `tokenizer` - The tokenizer to use for the request
    ///
    /// # Returns
    /// * `Ok(u64)` - The estimated compute units needed
    /// * `Err(AtomaProxyError)` - If the estimation fails or parameters are invalid
    ///
    /// # Errors
    /// Returns `AtomaServiceError` if the estimation fails or parameters are invalid
    ///
    /// # Warning
    /// This method assumes that the tokenizer has been correctly retrieved from the `ProxyState` for
    /// the associated model, as obtained by calling `get_model` on `Self`.
    fn get_tokens_estimate(
        &self,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<TokensEstimate, AtomaServiceError>;
}
