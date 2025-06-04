use std::{str::FromStr, time::Instant};

use crate::{
    error::AtomaServiceError,
    handlers::{
        chat_completions::CHAT_COMPLETIONS_PATH,
        completions::COMPLETIONS_PATH,
        embeddings::EMBEDDINGS_PATH,
        image_generations::IMAGE_GENERATIONS_PATH,
        metrics::{
            CONFIDENTIAL_COMPUTE_MIDDLEWARE_SUCCESSFUL_TIME,
            SIGNATURE_VERIFICATION_MIDDLEWARE_SUCCESSFUL_TIME,
            VERIFY_PERMISSIONS_MIDDLEWARE_SUCCESSFUL_TIME,
        },
        request_model::TokensEstimate,
    },
    server::AppState,
    types::ConfidentialComputeRequest,
};
use atoma_confidential::types::{ConfidentialComputeDecryptionRequest, DH_PUBLIC_KEY_SIZE};
use atoma_p2p::constants::ONE_MILLION;
use atoma_state::types::{AtomaAtomaStateManagerEvent, StackAvailability};
use atoma_utils::{
    constants::{NONCE_SIZE, PAYLOAD_HASH_SIZE, SALT_SIZE},
    hashing::blake2b_hash,
    verify_signature,
};
use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{request::Parts, HeaderValue, Request},
    middleware::Next,
    response::Response,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use opentelemetry::KeyValue;
use serde_json::Value;
use sui_sdk::types::{
    base_types::SuiAddress,
    crypto::{PublicKey, Signature, SuiSignature},
    digests::TransactionDigest,
};
use tokio::sync::oneshot;
use tracing::instrument;

/// Body size limit for signature verification (contains the body size of the request)
const MAX_BODY_SIZE: usize = 1024 * 1024 * 1024; // 1GB

/// The key for the model in the request body
const MODEL: &str = "model";

/// Metadata for confidential compute decryption requests
pub struct DecryptionMetadata {
    /// The plaintext body
    pub plaintext: Vec<u8>,

    /// The client's Diffie-Hellman public key
    pub client_dh_public_key: [u8; DH_PUBLIC_KEY_SIZE],

    /// The salt
    pub salt: [u8; SALT_SIZE],
}

/// Metadata for confidential compute encryption requests
#[derive(Clone, Debug)]
pub struct EncryptionMetadata {
    /// The client's proxy X25519 public key
    pub client_x25519_public_key: [u8; DH_PUBLIC_KEY_SIZE],
    /// The salt
    pub salt: [u8; SALT_SIZE],
}

/// Metadata extracted from the request
#[derive(Clone, Debug, Default)]
pub struct RequestMetadata {
    /// The stack small ID
    pub stack_small_id: Option<i64>,
    /// The number of input tokens
    pub num_input_tokens: i64,
    /// The estimated total number of tokens
    pub estimated_output_tokens: i64,
    /// The price per one million tokens
    pub price_per_one_million_tokens: i64,
    /// User id in the proxy db
    pub user_id: Option<i64>,
    /// User address that sent the request
    pub user_address: String,
    /// The payload hash
    pub payload_hash: [u8; 32],
    /// The type of request
    pub request_type: RequestType,
    /// The client's Diffie-Hellman public key and salt
    pub client_encryption_metadata: Option<EncryptionMetadata>,
    /// endpoint path
    pub endpoint_path: String,
}

/// The type of request
///
/// This enum is used to determine the type of request based on the path of the request.
#[derive(Clone, Debug, Default, Eq, PartialEq, Copy)]
pub enum RequestType {
    #[default]
    ChatCompletions,
    Completions,
    Embeddings,
    ImageGenerations,
    NonInference,
}

impl RequestMetadata {
    /// Create a new `RequestMetadata` with the given stack info
    #[must_use]
    pub const fn with_stack_info(mut self, stack_small_id: i64) -> Self {
        self.stack_small_id = Some(stack_small_id);
        self
    }

    #[must_use]
    pub const fn with_tokens_information(
        mut self,
        num_input_tokens: i64,
        estimated_output_tokens: i64,
    ) -> Self {
        self.num_input_tokens = num_input_tokens;
        self.estimated_output_tokens = estimated_output_tokens;
        self
    }

    /// Create a new `RequestMetadata` with the given price per one million tokens
    ///
    /// # Arguments
    /// * `price_per_one_million_tokens` - The price per one million tokens
    ///
    /// # Returns
    /// Returns self with the updated price per one million tokens for method chaining
    ///
    /// # Example
    ///
    /// ```
    /// use atoma_service::middleware::RequestMetadata;
    ///
    /// let metadata = RequestMetadata::default()
    ///     .with_price_per_one_million_tokens(100000);
    /// ```
    #[must_use]
    pub const fn with_price_per_one_million_tokens(
        mut self,
        price_per_one_million_tokens: i64,
    ) -> Self {
        self.price_per_one_million_tokens = price_per_one_million_tokens;
        self
    }

    /// Create a new `RequestMetadata` with the given user address
    ///
    /// # Arguments
    /// * `user_address` - The user address that sent the request
    ///
    /// # Returns
    /// Returns self with the updated user address for method chaining
    ///
    /// # Example
    /// ```
    /// use atoma_service::middleware::RequestMetadata;
    ///
    /// let metadata = RequestMetadata::default()
    ///    .with_user_address("0x1234567890abcdef".to_string());
    /// ```
    #[must_use]
    pub fn with_user_address(mut self, user_address: String) -> Self {
        self.user_address = user_address;
        self
    }

    /// Create a new `RequestMetadata` with the given user id
    ///
    /// # Arguments
    /// * `user_id` - The user id in the proxy db
    ///
    /// # Returns
    /// Returns self with the updated user id for method chaining
    ///
    /// # Example
    /// ```
    /// use atoma_service::middleware::RequestMetadata;
    ///
    /// let metadata = RequestMetadata::default()
    ///     .with_user_id(Some(1234567890));
    /// ```
    #[must_use]
    pub const fn with_user_id(mut self, user_id: Option<i64>) -> Self {
        self.user_id = user_id;
        self
    }

    /// Create a new `RequestMetadata` with the given payload hash
    #[must_use]
    pub const fn with_payload_hash(mut self, payload_hash: [u8; PAYLOAD_HASH_SIZE]) -> Self {
        self.payload_hash = payload_hash;
        self
    }

    /// Sets the request type for this metadata instance
    ///
    /// # Arguments
    /// * `request_type` - The type of request (ChatCompletions, Embeddings, ImageGenerations, or NonInference)
    ///
    /// # Returns
    /// Returns self with the updated request type for method chaining
    ///
    /// # Example
    /// ```
    /// use atoma_service::middleware::{RequestMetadata, RequestType};
    ///
    /// let metadata = RequestMetadata::default()
    ///     .with_request_type(RequestType::ChatCompletions);
    /// ```
    #[must_use]
    pub const fn with_request_type(mut self, request_type: RequestType) -> Self {
        self.request_type = request_type;
        self
    }

    /// Sets the client's encryption metadata for this metadata instance
    ///
    /// * `client_dh_public_key` - The client's Diffie-Hellman public key
    /// * `salt` - The salt
    ///
    /// # Returns
    /// Returns self with the updated client's encryption metadata for method chaining
    ///
    /// # Example
    /// ```rust,ignore
    /// use atoma_service::middleware::RequestMetadata;
    ///
    /// let metadata = RequestMetadata::default()
    ///     .with_client_encryption_metadata(client_dh_public_key, salt);
    /// ```
    #[must_use]
    pub const fn with_client_encryption_metadata(
        mut self,
        client_x25519_public_key: [u8; DH_PUBLIC_KEY_SIZE],
        salt: [u8; SALT_SIZE],
    ) -> Self {
        self.client_encryption_metadata = Some(EncryptionMetadata {
            client_x25519_public_key,
            salt,
        });
        self
    }

    /// Sets the endpoint path for this metadata instance
    ///
    /// * `endpoint_path` - The path of the endpoint
    ///
    /// # Returns
    /// Returns self with the updated endpoint path for method chaining
    ///
    /// # Example
    /// ```rust,ignore
    /// use atoma_service::middleware::RequestMetadata;
    ///
    /// let metadata = RequestMetadata::default().with_endpoint_path(CHAT_COMPLETIONS_PATH.to_string());
    /// ```
    #[must_use]
    pub fn with_endpoint_path(mut self, endpoint_path: String) -> Self {
        self.endpoint_path = endpoint_path;
        self
    }
}

/// Middleware for verifying the signature of incoming requests.
///
/// This middleware is designed to authenticate and verify the integrity of incoming requests
/// for an OpenAI API-compatible chat completions endpoint. It performs the following steps:
///
/// 1. Extracts the signature, public key, and signature scheme from the request headers.
/// 2. Decodes the signature and public key from base64.
/// 3. Reads and hashes the request body, which contains data such as the prompt, parameters,
///    and other information necessary for chat completions.
/// 4. Verifies the signature using the specified signature scheme, ensuring the request
///    has not been tampered with and comes from a trusted source.
///
/// # Request Body
/// The body of the request is expected to contain JSON data compatible with the OpenAI API
/// chat completions endpoint, including:
/// - `prompt`: The input text or conversation history.
/// - `parameters`: Various parameters for controlling the chat completion (e.g., temperature,
///   max_tokens, stop sequences).
/// - Other fields as required by the OpenAI API specification.
///
/// # Headers
/// The middleware expects the following custom headers:
/// - `X-Signature`: The signature of the request body, base64 encoded.
///
/// # Extensions
/// This middleware adds or updates a `RequestMetadata` extension to the request containing:
/// - `payload_hash`: The 32-byte Blake2b hash of the request body
///
/// # Errors
/// Returns a `BAD_REQUEST` status code if:
/// - Required headers are missing or cannot be parsed.
/// - The signature or public key cannot be decoded.
/// - The request body exceeds the maximum size limit.
///
/// Returns an `UNAUTHORIZED` status code if:
/// - The signature verification fails.
///
/// # Security Note
/// This middleware is crucial for ensuring that only authorized clients can access the
/// chat completions endpoint, protecting against unauthorized use and potential abuse.
#[instrument(
    level = "info",
    skip_all,
    fields(
        endpoint = %req.uri().path(),
    ),
    err
)]
pub async fn signature_verification_middleware(
    req: Request<Body>,
    next: Next,
) -> Result<Response, AtomaServiceError> {
    let instant = Instant::now();
    let (mut req_parts, req_body) = req.into_parts();
    let endpoint = req_parts.uri.path().to_string();

    let base64_signature = req_parts
        .headers
        .get(atoma_utils::constants::SIGNATURE)
        .ok_or_else(|| AtomaServiceError::MissingHeader {
            header: atoma_utils::constants::SIGNATURE.to_string(),
            endpoint: endpoint.clone(),
        })?
        .to_str()
        .map_err(|e| AtomaServiceError::InvalidHeader {
            message: format!("Failed to convert signature to string, with error: {e}"),
            endpoint: endpoint.clone(),
        })?;
    let body_bytes = axum::body::to_bytes(req_body, MAX_BODY_SIZE)
        .await
        .map_err(|e| AtomaServiceError::InvalidBody {
            message: format!("Failed to convert body to bytes, with error: {e}"),
            endpoint: endpoint.clone(),
        })?;
    let body_json: Value =
        serde_json::from_slice(&body_bytes).map_err(|e| AtomaServiceError::InvalidBody {
            message: format!("Failed to parse body as JSON, with error: {e}"),
            endpoint: endpoint.clone(),
        })?;
    let body_blake2b_hash = blake2b_hash(body_json.to_string().as_bytes());
    let body_blake2b_hash_bytes: [u8; 32] = body_blake2b_hash
        .as_slice()
        .try_into()
        .expect("Invalid Blake2b hash length");
    verify_signature(base64_signature, &body_blake2b_hash_bytes).map_err(|e| {
        AtomaServiceError::AuthError {
            auth_error: format!("Failed to verify signature, with error: {e}"),
            endpoint: endpoint.clone(),
        }
    })?;
    let request_metadata = req_parts
        .extensions
        .get::<RequestMetadata>()
        .cloned()
        .unwrap_or_default()
        .with_payload_hash(body_blake2b_hash_bytes);
    req_parts.extensions.insert(request_metadata);
    let req = Request::from_parts(req_parts, Body::from(body_bytes));
    SIGNATURE_VERIFICATION_MIDDLEWARE_SUCCESSFUL_TIME.record(
        instant.elapsed().as_secs_f64(),
        &[KeyValue::new("endpoint", endpoint)],
    );
    Ok(next.run(req).await)
}

/// Generates a fiat request for the given parameters.
///
/// This function is used to create a request for the fiat payment system.
/// It checks if the user is allowed to pay by fiat and locks the amount
/// for the request.
///
/// # Arguments
/// * `stack_small_id` - The ID of the stack being used for this request.
/// * `endpoint` - The endpoint of the request.
/// * `state` - The application state.
/// * `sui_address` - The Sui address of the user.
/// * `max_total_tokens` - The maximum total tokens for the request.
/// * `num_input_tokens` - The number of input tokens.
/// * `req_parts` - The request parts.
/// * `request_type` - The type of request.
/// * `body_bytes` - The body bytes of the request.
/// * `instant` - The instant when the request was created.
///
/// # Returns
/// * `Ok(Request<Body>)` - The generated request.
/// * `Err(AtomaServiceError)` - An error occurred while generating the request.
#[instrument(
    level = "info",
    skip_all,
    fields(
        endpoint = %endpoint,
    ),
    err
)]
#[allow(clippy::too_many_arguments)]
async fn generate_request_from_stack(
    stack_small_id: HeaderValue,
    endpoint: String,
    state: State<AppState>,
    sui_address: SuiAddress,
    max_total_tokens: i64,
    num_input_tokens: i64,
    mut req_parts: Parts,
    request_type: RequestType,
    body_bytes: Bytes,
    instant: Instant,
) -> Result<Request<Body>, AtomaServiceError> {
    let stack_small_id = stack_small_id
        .to_str()
        .map_err(|e| AtomaServiceError::InvalidHeader {
            message: format!("Stack small ID cannot be converted to a string, with error: {e}"),
            endpoint: endpoint.clone(),
        })?
        .parse::<i64>()
        .map_err(|e| AtomaServiceError::InvalidHeader {
            message: format!("Stack small ID is not a valid integer, with error: {e}"),
            endpoint: endpoint.clone(),
        })?;
    let (result_sender, result_receiver) = oneshot::channel();
    state
        .state_manager_sender
        .send(
            AtomaAtomaStateManagerEvent::GetAvailableStackWithComputeUnits {
                stack_small_id,
                sui_address: sui_address.to_string(),
                total_num_tokens: max_total_tokens,
                result_sender,
            },
        )
        .map_err(|err| AtomaServiceError::InternalError {
            message: format!("Failed to get available stacks: {}", err),
            endpoint: endpoint.clone(),
        })?;
    let available_stack = result_receiver
        .await
        .map_err(|e| AtomaServiceError::AuthError {
            auth_error: format!(
                "Failed to get available stack with enough tokens, with error: {e}"
            ),
            endpoint: endpoint.clone(),
        })?
        .map_err(|err| AtomaServiceError::AuthError {
            auth_error: format!(
                "Failed to get available stack with enough tokens, with error: {err}"
            ),
            endpoint: endpoint.clone(),
        })?;

    match available_stack {
        StackAvailability::Available => {
            // NOTE: If we are within this branch logic, it means that there is a stack with the same
            // stack_small_id and the client has enough compute units to use it.
        }
        StackAvailability::DoesNotExist => {
            // NOTE: If we are within this branch logic, it means that no available stack was found,
            // which implies that no compute units were locked, so far. For this reason, we query the
            // Sui blockchain to check if a new stack was created for the client, already.
            let tx_digest_str = req_parts
                .headers
                .get(atoma_utils::constants::TX_DIGEST)
                .ok_or_else(|| AtomaServiceError::InvalidHeader {
                    message: "Stack not found, tx digest header expected but not found".to_string(),
                    endpoint: endpoint.clone(),
                })?
                .to_str()
                .map_err(|e| AtomaServiceError::InvalidHeader {
                    message: format!("Tx digest cannot be converted to a string, with error: {e}"),
                    endpoint: endpoint.clone(),
                })?;
            let tx_digest = TransactionDigest::from_str(tx_digest_str).unwrap();
            utils::request_blockchain_for_stack(
                &state,
                tx_digest,
                max_total_tokens,
                stack_small_id,
                endpoint.clone(),
            )
            .await?;
            // NOTE: We do not need to check that the stack small id matches the one in the request,
            // or that the number of compute units within the stack is enough for processing the request,
            // as the Sui subscriber service should handle this verification.
        }
        StackAvailability::Locked => {
            // NOTE: If we are within this branch logic, it means that there is a stack with the same
            // stack_small_id, but it is locked, so the user needs to buy a new stack, and we provide
            // a specific status code to flag this scenario to the client.
            return Err(AtomaServiceError::LockedStackError {
                message: format!(
                    "Stack with stack_small_id={stack_small_id} is locked, please buy a new stack."
                ),
                endpoint: endpoint.clone(),
            });
        }
        StackAvailability::Unavailable => {
            // NOTE: If we are within this branch logic, it means that there is a stack with the same
            // stack_small_id, but it is unavailable, so the client either buys a new stack or awaits
            // the stack to be available again.
            return Err(AtomaServiceError::UnavailableStackError {
                message: format!("Stack with stack_small_id={stack_small_id} is unavailable, please buy a new stack or await it to be available again."),
                endpoint: endpoint.clone(),
            });
        }
    }
    let request_metadata = req_parts
        .extensions
        .get::<RequestMetadata>()
        .cloned()
        .unwrap_or_default()
        .with_stack_info(stack_small_id)
        .with_tokens_information(num_input_tokens, max_total_tokens)
        .with_request_type(request_type)
        .with_endpoint_path(req_parts.uri.path().to_string());
    req_parts.extensions.insert(request_metadata);
    let req = Request::from_parts(req_parts, Body::from(body_bytes));
    {
        let mut entry = state
            .concurrent_requests_per_stack
            .entry(stack_small_id)
            .or_insert(0);
        *entry += 1;
    }
    VERIFY_PERMISSIONS_MIDDLEWARE_SUCCESSFUL_TIME.record(
        instant.elapsed().as_secs_f64(),
        &[KeyValue::new("endpoint", endpoint)],
    );
    Ok(req)
}

/// Generates a fiat request for the given parameters.
///
/// This function is used to create a request for the fiat payment system.
/// It checks if the user is allowed to pay by fiat and locks the amount
/// for the request.
///
/// # Arguments
/// * `endpoint` - The endpoint of the request.
/// * `state` - The application state.
/// * `sui_address` - The Sui address of the user.
/// * `max_total_tokens` - The maximum total tokens for the request.
/// * `num_input_tokens` - The number of input tokens.
/// * `req_parts` - The request parts.
/// * `request_type` - The type of request.
/// * `body_bytes` - The body bytes of the request.
/// * `model` - The model for the request.
/// * `instant` - The instant when the request was created.
///
/// # Returns
/// * `Ok(Request<Body>)` - The generated request.
/// * `Err(AtomaServiceError)` - An error occurred while generating the request.
#[instrument(
    level = "info",
    skip_all,
    fields(
        endpoint = %endpoint,
    ),
    err
)]
#[allow(clippy::too_many_arguments)]
async fn generate_fiat_request(
    endpoint: String,
    state: State<AppState>,
    sui_address: SuiAddress,
    max_output_tokens: i64,
    num_input_tokens: i64,
    mut req_parts: Parts,
    request_type: RequestType,
    body_bytes: Bytes,
    model: &str,
    instant: Instant,
) -> Result<Request<Body>, AtomaServiceError> {
    if !state
        .whitelist_sui_addresses_for_fiat
        .contains(&sui_address.to_string())
    {
        // The stack was not found and the address is not enabled for fiat.
        return Err(AtomaServiceError::MissingHeader {
            header: atoma_utils::constants::STACK_SMALL_ID.to_string(),
            endpoint: endpoint.clone(),
        });
    }

    let (result_sender, result_receiver) = oneshot::channel();

    state
        .state_manager_sender
        .send(AtomaAtomaStateManagerEvent::GetModelPricing {
            model: model.to_string(),
            result_sender,
        })
        .map_err(|err| AtomaServiceError::InternalError {
            message: format!("Failed to get model pricing: {}", err),
            endpoint: endpoint.clone(),
        })?;

    let price_per_one_million_tokens = result_receiver
        .await
        .map_err(|_| AtomaServiceError::InternalError {
            message: "Failed to get model pricing".to_string(),
            endpoint: endpoint.clone(),
        })?
        .map_err(|_| AtomaServiceError::ModelError {
            model_error: format!("Failed to get model pricing for model {model}"),
            endpoint: endpoint.clone(),
        })?
        .ok_or_else(|| AtomaServiceError::ModelError {
            model_error: format!("No pricing found for model {model}"),
            endpoint: endpoint.clone(),
        })?;

    let user_id = req_parts
        .headers
        .get(atoma_utils::constants::USER_ID)
        .ok_or_else(|| AtomaServiceError::MissingHeader {
            header: atoma_utils::constants::USER_ID.to_string(),
            endpoint: endpoint.clone(),
        })?;
    let user_id = user_id
        .to_str()
        .map_err(|e| AtomaServiceError::InvalidHeader {
            message: format!("User ID cannot be converted to a string, with error: {e}"),
            endpoint: endpoint.clone(),
        })?
        .parse::<i64>()
        .map_err(|e| AtomaServiceError::InvalidHeader {
            message: format!("User ID is not a valid integer, with error: {e}"),
            endpoint: endpoint.clone(),
        })?;

    state
        .state_manager_sender
        .send(AtomaAtomaStateManagerEvent::LockFiatAmount {
            user_id,
            user_address: sui_address.to_string(),
            estimated_input_amount: ((num_input_tokens as u128
                * price_per_one_million_tokens as u128)
                / ONE_MILLION) as i64,
            estimated_output_amount: ((max_output_tokens as u128
                * price_per_one_million_tokens as u128)
                / ONE_MILLION) as i64,
        })
        .map_err(|err| AtomaServiceError::InternalError {
            message: format!("Failed to lock fiat amount: {}", err),
            endpoint: endpoint.clone(),
        })?;

    let user_id = req_parts
        .headers
        .get(atoma_utils::constants::USER_ID)
        .ok_or_else(|| AtomaServiceError::MissingHeader {
            header: atoma_utils::constants::USER_ID.to_string(),
            endpoint: endpoint.clone(),
        })?;
    let user_id = user_id
        .to_str()
        .map_err(|e| AtomaServiceError::InvalidHeader {
            message: format!("User ID cannot be converted to a string, with error: {e}"),
            endpoint: endpoint.clone(),
        })?
        .parse::<i64>()
        .map_err(|e| AtomaServiceError::InvalidHeader {
            message: format!("User ID is not a valid integer, with error: {e}"),
            endpoint: endpoint.clone(),
        })?;
    let request_metadata = req_parts
        .extensions
        .get::<RequestMetadata>()
        .cloned()
        .unwrap_or_default()
        .with_user_address(sui_address.to_string())
        .with_price_per_one_million_tokens(price_per_one_million_tokens)
        .with_tokens_information(num_input_tokens, max_output_tokens)
        .with_request_type(request_type)
        .with_user_id(Some(user_id))
        .with_endpoint_path(req_parts.uri.path().to_string());
    req_parts.extensions.insert(request_metadata);
    let req = Request::from_parts(req_parts, Body::from(body_bytes));
    VERIFY_PERMISSIONS_MIDDLEWARE_SUCCESSFUL_TIME.record(
        instant.elapsed().as_secs_f64(),
        &[KeyValue::new("endpoint", endpoint)],
    );

    Ok(req)
}

/// Middleware for verifying stack permissions and compute units usage.
///
/// This middleware performs several checks to ensure that the incoming request
/// is authorized to use the specified model and has sufficient compute units available.
///
/// # Steps:
/// 1. Extracts and validates the public key and stack ID from request headers.
/// 2. Parses the request body to extract the model and messages.
/// 3. Verifies that the requested model is supported.
/// 4. Calculates the total number of compute units required for the request.
/// 5. Checks if the user has an available stack with sufficient compute units.
///
/// # Headers
/// The middleware expects the following custom headers:
/// - `X-Stack-Small-Id`: The ID of the stack being used for this request.
///
/// # Request Body
/// The body should be a JSON object containing:
/// - `model`: The name of the AI model to be used.
/// - `messages`: An array of message objects, each containing a "content" field.
/// - `max_tokens`: The maximum number of tokens for the AI's response.
///
/// # Extensions
/// This middleware adds a `RequestMetadata` extension to the request containing:
/// - `stack_small_id`: The ID of the stack being used
/// - `estimated_total_tokens`: The total number of tokens calculated for the request
///
/// This metadata can be accessed by downstream handlers using `req.extensions()`.
///
/// # Errors
/// Returns a `BAD_REQUEST` status code if:
/// - Required headers are missing or invalid.
/// - The request body is invalid or missing required fields.
/// - The requested model is not supported.
///
/// Returns an `UNAUTHORIZED` status code if:
/// - There's no available stack with sufficient compute units.
/// - Fetching available stacks fails.
///
/// # Security Note
/// This middleware is crucial for ensuring that users only consume resources they're
/// authorized to use and have sufficient compute units for their requests.
#[instrument(
    level = "info",
    skip_all,
    fields(
        endpoint = %req.uri().path(),
    ),
    err
)]
pub async fn verify_permissions(
    state: State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Result<Response, AtomaServiceError> {
    let instant = Instant::now();
    let (req_parts, req_body) = req.into_parts();
    let endpoint = req_parts.uri.path().to_string();

    // Get request path to determine type
    let request_type = match req_parts.uri.path() {
        CHAT_COMPLETIONS_PATH => RequestType::ChatCompletions,
        EMBEDDINGS_PATH => RequestType::Embeddings,
        IMAGE_GENERATIONS_PATH => RequestType::ImageGenerations,
        COMPLETIONS_PATH => RequestType::Completions,
        _ => RequestType::NonInference,
    };

    let base64_signature = req_parts
        .headers
        .get(atoma_utils::constants::SIGNATURE)
        .ok_or_else(|| AtomaServiceError::MissingHeader {
            header: atoma_utils::constants::SIGNATURE.to_string(),
            endpoint: endpoint.clone(),
        })?
        .to_str()
        .map_err(|e| AtomaServiceError::InvalidHeader {
            message: format!("Failed to convert signature to string, with error: {e}"),
            endpoint: endpoint.clone(),
        })?;
    let signature =
        Signature::from_str(base64_signature).map_err(|e| AtomaServiceError::InvalidHeader {
            message: format!("Failed to parse signature, with error: {e}"),
            endpoint: endpoint.clone(),
        })?;
    let public_key_bytes = signature.public_key_bytes();
    let public_key =
        PublicKey::try_from_bytes(signature.scheme(), public_key_bytes).map_err(|e| {
            AtomaServiceError::InvalidHeader {
                message: format!("Failed to extract public key from bytes, with error: {e}"),
                endpoint: endpoint.clone(),
            }
        })?;
    let sui_address = SuiAddress::from(&public_key);

    let body_bytes = axum::body::to_bytes(req_body, MAX_BODY_SIZE)
        .await
        .map_err(|e| AtomaServiceError::InvalidBody {
            message: format!("Failed to convert body to bytes, with error: {e}"),
            endpoint: endpoint.clone(),
        })?;
    let body_json: Value =
        serde_json::from_slice(&body_bytes).map_err(|e| AtomaServiceError::InvalidBody {
            message: format!("Failed to parse body as JSON, with error: {e}"),
            endpoint: endpoint.clone(),
        })?;
    let model = body_json
        .get(MODEL)
        .ok_or_else(|| AtomaServiceError::InvalidBody {
            message: "Model not found in body".to_string(),
            endpoint: endpoint.clone(),
        })?
        .as_str()
        .ok_or_else(|| AtomaServiceError::InvalidBody {
            message: "Model is not a string".to_string(),
            endpoint: endpoint.clone(),
        })?;
    utils::check_if_too_many_requests(&state, model, &endpoint).await?;
    if !state.models.contains(&model.to_string()) {
        return Err(AtomaServiceError::InvalidBody {
            message: format!("Model not supported, supported models: {:?}", state.models),
            endpoint: endpoint.clone(),
        });
    }

    let TokensEstimate {
        num_input_tokens,
        max_output_tokens,
        max_total_tokens,
    } = utils::calculate_tokens(&body_json, request_type, &state, model, &endpoint)?;

    let max_total_tokens = max_total_tokens as i64;
    let max_output_tokens = max_output_tokens as i64;
    let num_input_tokens = num_input_tokens as i64;

    let stack_small_id = req_parts
        .headers
        .get(atoma_utils::constants::STACK_SMALL_ID)
        .cloned();

    let req = if let Some(stack_small_id) = stack_small_id {
        generate_request_from_stack(
            stack_small_id,
            endpoint,
            state,
            sui_address,
            max_total_tokens,
            num_input_tokens,
            req_parts,
            request_type,
            body_bytes,
            instant,
        )
        .await?
    } else {
        generate_fiat_request(
            endpoint,
            state,
            sui_address,
            max_output_tokens,
            num_input_tokens,
            req_parts,
            request_type,
            body_bytes,
            model,
            instant,
        )
        .await?
    };
    Ok(next.run(req).await)
}

/// Middleware for handling confidential compute requests by decrypting encrypted payloads.
///
/// This middleware intercepts requests containing encrypted data and processes them through
/// a confidential compute pipeline before passing them to the next handler. It performs
/// decryption using a combination of:
/// - A salt for key derivation
/// - A nonce for encryption uniqueness
/// - A Diffie-Hellman public key for secure key exchange
///
/// # Headers Required
/// The middleware expects the following headers:
/// - `X-Salt`: Base string containing the salt used in key derivation
/// - `X-Nonce`: Base string containing the nonce used in encryption
/// - `X-Node-X25519-PublicKey`: Base64-encoded public key (32 bytes) for key exchange
///
/// # Request Flow
/// 1. Extracts and validates required headers
/// 2. Decodes the Diffie-Hellman public key from base64
/// 3. Reads the encrypted request body
/// 4. Sends decryption request to confidential compute service
/// 5. Receives decrypted plaintext
/// 6. Reconstructs request with decrypted body
///
/// # Returns
/// * `Ok(Response)` - The response from the next handler after successful decryption
/// * `Err(AtomaServiceError)` - One of:
///   - `AtomaServiceError::InvalidHeader` if headers are missing or invalid
///   - `AtomaServiceError::InternalError` if decryption service communication fails
///
/// # Example Headers
/// ```text
/// X-Salt: randomsaltvalue
/// X-Nonce: uniquenoncevalue
/// X-Node-X25519-PublicKey: base64encodedkey...
/// ```
///
/// # Security Notes
/// - The salt and nonce should be unique for each request
/// - The Diffie-Hellman public key must be exactly 32 bytes when decoded
/// - All cryptographic operations are performed in a separate confidential compute service
#[instrument(
    level = "info", skip_all,
    fields(
        endpoint = %req.uri().path(),
    ),
    err
)]
pub async fn confidential_compute_middleware(
    state: State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Result<Response, AtomaServiceError> {
    let instant = Instant::now();
    let (mut req_parts, req_body) = req.into_parts();

    let endpoint = req_parts.uri.path().to_string();
    let body_bytes = axum::body::to_bytes(req_body, MAX_BODY_SIZE)
        .await
        .map_err(|e| AtomaServiceError::InvalidBody {
            message: format!("Failed to convert body to bytes, with error: {e}"),
            endpoint: endpoint.clone(),
        })?;
    let confidential_compute_request =
        serde_json::from_slice::<ConfidentialComputeRequest>(&body_bytes).map_err(|e| {
            AtomaServiceError::InvalidBody {
                message: format!(
                    "Failed to parse body as ConfidentialComputeRequest, with error: {e}"
                ),
                endpoint: endpoint.clone(),
            }
        })?;

    let plaintext_body_hash_bytes = STANDARD
        .decode(&confidential_compute_request.plaintext_body_hash)
        .map_err(|e| AtomaServiceError::InvalidHeader {
            message: format!(
                "Plaintext body hash cannot be converted to a string, with error: {e}"
            ),
            endpoint: endpoint.clone(),
        })?;
    let plaintext_body_hash_bytes: [u8; PAYLOAD_HASH_SIZE] = plaintext_body_hash_bytes.try_into().map_err(|e| {
        AtomaServiceError::InvalidHeader {
            message: format!(
                "Failed to convert plaintext body hash bytes to {PAYLOAD_HASH_SIZE}-byte array, incorrect length, with error: {:?}",
                e
            ),
            endpoint: endpoint.to_string(),
        }
    })?;

    utils::verify_plaintext_body_hash(&plaintext_body_hash_bytes, &req_parts.headers, &endpoint)?;

    match utils::decrypt_confidential_compute_request(
        &state,
        &confidential_compute_request,
        &endpoint,
    )
    .await
    {
        Ok(DecryptionMetadata {
            plaintext,
            client_dh_public_key: client_dh_public_key_bytes,
            salt: salt_bytes,
        }) => {
            utils::check_plaintext_body_hash(plaintext_body_hash_bytes, &plaintext, &endpoint)?;
            let body = Body::from(plaintext);
            let request_metadata = req_parts
                .extensions
                .get::<RequestMetadata>()
                .cloned()
                .unwrap_or_default();
            req_parts.extensions.insert(
                request_metadata
                    .with_client_encryption_metadata(client_dh_public_key_bytes, salt_bytes)
                    .with_payload_hash(plaintext_body_hash_bytes),
            );
            let stack_small_id = confidential_compute_request.stack_small_id;
            req_parts.headers.insert(
                atoma_utils::constants::STACK_SMALL_ID,
                HeaderValue::from_str(&stack_small_id.to_string()).map_err(|e| {
                    AtomaServiceError::InvalidHeader {
                        message: format!(
                            "Failed to convert stack small ID to a string, with error: {e}"
                        ),
                        endpoint: endpoint.clone(),
                    }
                })?,
            );
            let req = Request::from_parts(req_parts, body);
            CONFIDENTIAL_COMPUTE_MIDDLEWARE_SUCCESSFUL_TIME.record(
                instant.elapsed().as_secs_f64(),
                &[KeyValue::new("endpoint", endpoint)],
            );
            Ok(next.run(req).await)
        }
        Err(e) => Err(AtomaServiceError::InternalError {
            message: format!("Failed to decrypt confidential compute response, with error: {e}"),
            endpoint: endpoint.clone(),
        }),
    }
}

pub mod utils {
    use hyper::HeaderMap;

    use crate::handlers::{
        chat_completions::RequestModelChatCompletions,
        completions::RequestModelCompletions,
        embeddings::RequestModelEmbeddings,
        image_generations::RequestModelImageGenerations,
        inference_service_metrics::get_all_metrics,
        request_model::{RequestModel, TokensEstimate},
    };

    use super::{
        blake2b_hash, instrument, oneshot, verify_signature, AppState, AtomaServiceError,
        ConfidentialComputeDecryptionRequest, ConfidentialComputeRequest, DecryptionMetadata,
        Engine, RequestType, TransactionDigest, Value, DH_PUBLIC_KEY_SIZE, NONCE_SIZE,
        PAYLOAD_HASH_SIZE, SALT_SIZE, STANDARD,
    };

    /// Default max completion tokens for chat completions
    const DEFAULT_MAX_TOKENS_CHAT_COMPLETIONS: i64 = 8192;

    /// The key for the max tokens in the request body (currently deprecated, as per OpenAI API spec)
    const MAX_TOKENS: &str = "max_tokens";

    /// The key for max completion tokens in the request body
    const MAX_COMPLETION_TOKENS: &str = "max_completion_tokens";

    /// The key for the messages in the request body
    const MESSAGES: &str = "messages";

    /// Requests and verifies stack information from the blockchain for a given transaction.
    ///
    /// This function communicates with a blockchain service to verify the existence and validity
    /// of a stack associated with a transaction. It checks if the stack has sufficient compute
    /// units available for the requested operation.
    ///
    /// # Arguments
    ///
    /// * `state` - Reference to the application state containing the stack retrieval channel
    /// * `tx_digest` - Transaction digest to query on the blockchain
    /// * `estimated_compute_units` - Number of compute units required for the operation
    /// * `stack_small_id` - Identifier for the stack being verified
    /// * `endpoint` - API endpoint path (used for error context)
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the stack exists and has sufficient compute units
    /// * `Err(AtomaServiceError)` - If verification fails, with variants:
    ///   - `InternalError` - Channel communication failures
    ///   - `AuthError` - Insufficient compute units or invalid stack
    ///
    /// # Channel Communication
    ///
    /// Uses a oneshot channel pattern for async communication:
    /// 1. Creates a oneshot channel for receiving the verification result
    /// 2. Sends request data through the state's stack retrieval channel
    /// 3. Awaits response containing stack and compute unit information
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sui_sdk::types::digests::TransactionDigest;
    ///
    /// async fn verify_stack(state: &AppState, digest: TransactionDigest) {
    ///     let result = request_blockchain_for_stack(
    ///         state,
    ///         digest,
    ///         1000, // estimated compute units
    ///         42,   // stack_small_id
    ///         "/v1/completions".to_string()
    ///     ).await;
    ///
    ///     match result {
    ///         Ok(()) => println!("Stack verified successfully"),
    ///         Err(e) => eprintln!("Stack verification failed: {}", e),
    ///     }
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Will return an error if:
    /// - Channel communication fails when sending/receiving verification request
    /// - No stack is found for the given transaction
    /// - Stack exists but has insufficient compute units
    /// - Stack small ID doesn't match the expected value
    #[instrument(level = "trace", skip_all, err)]
    pub async fn request_blockchain_for_stack(
        state: &AppState,
        tx_digest: TransactionDigest,
        estimated_compute_units: i64,
        stack_small_id: i64,
        endpoint: String,
    ) -> Result<(), AtomaServiceError> {
        let (result_sender, result_receiver) = oneshot::channel();
        state
            .stack_retrieve_sender
            .send((
                tx_digest,
                estimated_compute_units,
                stack_small_id,
                result_sender,
            ))
            .map_err(|_| AtomaServiceError::InternalError {
                message: "Failed to send compute units request".to_string(),
                endpoint: endpoint.clone(),
            })?;
        let result = result_receiver
            .await
            .map_err(|_| AtomaServiceError::InternalError {
                message: "Failed to receive compute units".to_string(),
                endpoint: endpoint.clone(),
            })?;
        if let (Some(_), Some(_)) = result {
            Ok(())
        } else {
            Err(AtomaServiceError::AuthError {
                auth_error: format!(
                    "Not enough compute units found for transaction with digest {tx_digest}"
                ),
                endpoint,
            })
        }
    }

    /// Calculates the total number of compute units required for a request based on its type and content.
    ///
    /// # Arguments
    /// * `body_json` - The parsed JSON body of the request containing model-specific parameters
    /// * `request_type` - The type of request (ChatCompletions, Embeddings, ImageGenerations, or NonInference)
    /// * `state` - Application state containing model configurations and tokenizers
    /// * `model` - The name of the AI model being used
    /// * `endpoint` - The endpoint that the request was made to
    ///
    /// # Returns
    /// * `Ok(i64)` - The total number of compute units required
    /// * `Err(AtomaServiceError)` - If there's an error calculating the units, returns an appropriate HTTP status code
    ///
    /// # Errors
    /// - `InvalidBody` - If the request body is invalid
    /// - `InvalidHeader` - If the request headers are invalid
    /// - `InternalError` - If there's an error calculating the units
    ///
    /// # Compute Unit Calculation
    /// The calculation varies by request type:
    /// - ChatCompletions: Based on input tokens + max output tokens
    /// - Embeddings: Based on input text length
    /// - ImageGenerations: Based on image dimensions and quantity
    /// - NonInference: Returns 0 (no compute units required)
    ///
    /// This function delegates to specific calculators based on the request type:
    /// - `RequestModelChatCompletions`
    /// - `RequestModelEmbeddings`
    /// - `RequestModelImageGenerations`
    pub fn calculate_tokens(
        body_json: &Value,
        request_type: RequestType,
        state: &AppState,
        model: &str,
        endpoint: &str,
    ) -> Result<TokensEstimate, AtomaServiceError> {
        match request_type {
            RequestType::ChatCompletions => {
                let request_model = RequestModelChatCompletions::new(body_json)?;
                let tokenizer_index =
                    state
                        .models
                        .iter()
                        .position(|m| m == model)
                        .ok_or_else(|| AtomaServiceError::InvalidBody {
                            message: "Model not supported".to_string(),
                            endpoint: endpoint.to_string(),
                        })?;
                request_model.get_tokens_estimate(Some(&state.tokenizers[tokenizer_index]))
            }
            RequestType::Completions => {
                let request_model = RequestModelCompletions::new(body_json)?;
                let tokenizer_index =
                    state
                        .models
                        .iter()
                        .position(|m| m == model)
                        .ok_or_else(|| AtomaServiceError::InvalidBody {
                            message: "Model not supported".to_string(),
                            endpoint: endpoint.to_string(),
                        })?;
                request_model.get_tokens_estimate(Some(&state.tokenizers[tokenizer_index]))
            }
            RequestType::Embeddings => {
                let request_model = RequestModelEmbeddings::new(body_json)?;
                let tokenizer_index =
                    state
                        .models
                        .iter()
                        .position(|m| m == model)
                        .ok_or_else(|| AtomaServiceError::InvalidBody {
                            message: "Model not supported".to_string(),
                            endpoint: endpoint.to_string(),
                        })?;
                request_model.get_tokens_estimate(Some(&state.tokenizers[tokenizer_index]))
            }
            RequestType::ImageGenerations => {
                let request_model = RequestModelImageGenerations::new(body_json)?;
                request_model.get_tokens_estimate(None)
            }
            RequestType::NonInference => Ok(TokensEstimate {
                num_input_tokens: 0,
                max_output_tokens: 0,
                max_total_tokens: 0,
            }),
        }
    }

    /// Calculates the total number of compute units required for a chat completion request.
    ///
    /// This function analyzes the request body to determine the total computational cost by:
    /// 1. Counting tokens in all input messages
    /// 2. Adding safety margins for message formatting
    /// 3. Including the requested maximum output tokens
    ///
    /// # Arguments
    /// * `body_json` - The parsed JSON body of the request containing:
    ///   - `messages`: Array of message objects with "content" fields
    ///   - `max_tokens`: Maximum number of tokens for the model's response
    /// * `state` - Application state containing model configurations and tokenizers
    /// * `model` - The name of the AI model being used
    /// * `endpoint` - The endpoint that the request was made to
    /// # Returns
    /// * `Ok(i64)` - The total number of compute units required
    /// * `Err(AtomaServiceError)` - AtomaServiceError::InvalidBody if:
    ///   - The model is not supported
    ///   - Required fields are missing
    ///   - Message content is invalid
    ///   - Tokenization fails
    ///
    /// # Token Calculation
    /// For each message:
    /// - Base tokens: Number of tokens in the message content
    /// - +2 tokens: Safety margin for message delimiters
    /// - +1 token: Safety margin for role name
    ///     Plus the requested max_tokens for the response
    ///
    /// # Example JSON Structure
    /// ```json
    /// {
    ///     "messages": [
    ///         {"role": "user", "content": "Hello, how are you?"},
    ///         {"role": "assistant", "content": "I'm doing well, thank you!"}
    ///     ],
    ///     "max_tokens": 100
    /// }
    /// ```
    #[instrument(level = "trace", skip_all, err)]
    pub fn calculate_chat_completion_compute_units(
        body_json: &Value,
        state: &AppState,
        model: &str,
        endpoint: String,
    ) -> Result<i64, AtomaServiceError> {
        let tokenizer_index = state
            .models
            .iter()
            .position(|m| m == model)
            .ok_or_else(|| AtomaServiceError::InvalidBody {
                message: "Model not supported".to_string(),
                endpoint: endpoint.clone(),
            })?;

        let messages = body_json
            .get(MESSAGES)
            .ok_or_else(|| AtomaServiceError::InvalidBody {
                message: "Messages not found in body".to_string(),
                endpoint: endpoint.clone(),
            })?
            .as_array()
            .ok_or_else(|| AtomaServiceError::InvalidBody {
                message: "Messages is not an array".to_string(),
                endpoint: endpoint.clone(),
            })?;

        let mut total_num_compute_units = 0;
        for message in messages {
            let content = message
                .get("content")
                .ok_or_else(|| AtomaServiceError::InvalidBody {
                    message: "Message content not found".to_string(),
                    endpoint: endpoint.clone(),
                })?;
            let content_str = content
                .as_str()
                .ok_or_else(|| AtomaServiceError::InvalidBody {
                    message: "Message content is not a string".to_string(),
                    endpoint: endpoint.clone(),
                })?;
            let num_tokens = state.tokenizers[tokenizer_index]
                .encode(content_str, true)
                .map_err(|_| AtomaServiceError::InvalidBody {
                    message: "Failed to encode message content".to_string(),
                    endpoint: endpoint.clone(),
                })?
                .get_ids()
                .len() as i64;
            total_num_compute_units += num_tokens;
            // add 2 tokens as a safety margin, for start and end message delimiters
            total_num_compute_units += 2;
            // add 1 token as a safety margin, for the role name of the message
            total_num_compute_units += 1;
        }

        total_num_compute_units += body_json
            .get(MAX_COMPLETION_TOKENS)
            .or_else(|| body_json.get(MAX_TOKENS))
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(DEFAULT_MAX_TOKENS_CHAT_COMPLETIONS);

        Ok(total_num_compute_units)
    }

    /// Verifies a plaintext body hash against a provided signature.
    ///
    /// This function performs signature verification for confidential compute requests by:
    /// 1. Retrieving and validating the signature from request headers
    /// 2. Verifying the signature against the provided plaintext body hash
    ///
    /// # Arguments
    /// * `plaintext_body_hash` - The 32-byte plaintext body hash to verify
    /// * `headers` - HTTP headers containing the signature for verification
    /// * `endpoint` - The API endpoint path being accessed (used for error context)
    ///
    /// # Returns
    /// * `Ok(())` - If signature verification succeeds
    /// * `Err(AtomaServiceError)` - If any step fails, with variants:
    ///   - `InvalidHeader` - If the signature is malformed
    ///   - `MissingHeader` - If the signature header is missing
    ///
    /// # Errors
    /// This function will return an error if:
    /// - The signature header is missing
    /// - The signature cannot be parsed as a string
    /// - The signature verification fails
    ///
    /// # Example
    /// ```rust,ignore
    /// use axum::http::HeaderMap;
    ///
    /// let plaintext_body_hash = [0u8; 32]; // Your 32-byte hash
    /// let headers = HeaderMap::new(); // Headers with signature
    /// let endpoint = "/v1/chat/completions";
    ///
    /// match verify_plaintext_body_hash(&plaintext_body_hash, &headers, endpoint) {
    ///     Ok(()) => println!("Signature verification successful"),
    ///     Err(e) => eprintln!("Verification failed: {}", e),
    /// }
    /// ```
    #[instrument(level = "trace", skip_all, err)]
    pub fn verify_plaintext_body_hash(
        plaintext_body_hash: &[u8; PAYLOAD_HASH_SIZE],
        headers: &HeaderMap,
        endpoint: &str,
    ) -> Result<(), AtomaServiceError> {
        let base64_signature = headers
            .get(atoma_utils::constants::SIGNATURE)
            .ok_or_else(|| AtomaServiceError::MissingHeader {
                header: atoma_utils::constants::SIGNATURE.to_string(),
                endpoint: endpoint.to_string(),
            })?;
        let base64_signature =
            base64_signature
                .to_str()
                .map_err(|e| AtomaServiceError::InvalidHeader {
                    message: format!("Signature cannot be converted to a string, with error: {e}"),
                    endpoint: endpoint.to_string(),
                })?;
        verify_signature(base64_signature, plaintext_body_hash).map_err(|e| {
            AtomaServiceError::InvalidHeader {
                message: format!("Failed to verify signature, with error: {e}"),
                endpoint: endpoint.to_string(),
            }
        })
    }

    /// Decrypts a confidential compute request.
    ///
    /// This function decrypts a confidential compute request using the provided state and request data.
    ///
    /// # Arguments
    /// * `state` - The application state containing decryption capabilities
    /// * `confidential_compute_request` - The confidential compute request to decrypt
    /// * `endpoint` - The API endpoint path being accessed (used for error context)
    ///
    /// # Returns
    /// * `Ok(ConfidentialComputeDecryptionResponse)` - The decrypted confidential compute response
    /// * `Err(AtomaServiceError)` - If any step fails, with variants:
    ///   - `InvalidHeader` - If the signature is malformed
    ///   - `MissingHeader` - If the signature header is missing
    ///
    /// # Errors
    /// This function will return an error if:
    /// - The signature header is missing
    /// - The signature cannot be parsed as a string
    /// - The signature verification fails
    ///
    /// # Example
    /// ```rust,ignore
    /// use atoma_service::middleware::utils;
    ///
    /// let state = AppState::new();
    /// let confidential_compute_request = ConfidentialComputeRequest::new();
    /// let endpoint = "/v1/chat/completions";
    ///
    /// match utils::decrypt_confidential_compute_request(&state, &confidential_compute_request, endpoint) {
    ///     Ok(decrypted_response) => println!("Decrypted response: {:?}", decrypted_response),
    ///     Err(e) => eprintln!("Decryption failed: {}", e),
    /// }
    /// ```
    #[instrument(level = "trace", skip_all, err)]
    pub async fn decrypt_confidential_compute_request(
        state: &AppState,
        confidential_compute_request: &ConfidentialComputeRequest,
        endpoint: &str,
    ) -> Result<DecryptionMetadata, AtomaServiceError> {
        let salt_bytes = STANDARD
            .decode(&confidential_compute_request.salt)
            .map_err(|e| AtomaServiceError::InvalidHeader {
                message: format!("Salt cannot be converted to a string, with error: {e}"),
                endpoint: endpoint.to_string(),
            })?;
        let salt_bytes: [u8; SALT_SIZE] = salt_bytes.try_into().map_err(|e| {
            AtomaServiceError::InvalidHeader {
                message: format!(
                    "Failed to convert salt bytes to {SALT_SIZE}-byte array, incorrect length, with error: {:?}",
                    e
                ),
                endpoint: endpoint.to_string(),
            }
        })?;
        let nonce_bytes = STANDARD
            .decode(&confidential_compute_request.nonce)
            .map_err(|e| AtomaServiceError::InvalidHeader {
                message: format!("Nonce cannot be converted to a string, with error: {e}"),
                endpoint: endpoint.to_string(),
            })?;
        let nonce_bytes: [u8; NONCE_SIZE] = nonce_bytes.try_into().map_err(|e| {
        AtomaServiceError::InvalidHeader {
            message: format!(
                "Failed to convert nonce bytes to {NONCE_SIZE}-byte array, incorrect length, with error: {:?}",
                e
            ),
            endpoint: endpoint.to_string(),
        }
    })?;
        let client_dh_public_key_bytes: [u8; DH_PUBLIC_KEY_SIZE] = STANDARD
        .decode(&confidential_compute_request.client_dh_public_key)
        .map_err(|e| {
            AtomaServiceError::InvalidHeader {
                message: format!("Failed to decode Proxy X25519 public key from base64 encoding, with error: {e}"),
                endpoint: endpoint.to_string(),
            }
        })?
        .try_into()
        .map_err(|e| {
            AtomaServiceError::InvalidHeader {
                message: format!(
                    "Failed to convert Proxy X25519 public key bytes to 32-byte array, incorrect length, with error: {:?}",
                    e
                ),
                endpoint: endpoint.to_string(),
            }
        })?;
        let node_dh_public_key_bytes: [u8; DH_PUBLIC_KEY_SIZE] = STANDARD
        .decode(&confidential_compute_request.node_dh_public_key)
        .map_err(|e| {
            AtomaServiceError::InvalidHeader {
                message: format!("Failed to decode Node X25519 public key from base64 encoding, with error: {e}"),
                endpoint: endpoint.to_string(),
            }
        })?
        .try_into()
        .map_err(|e| {
            AtomaServiceError::InvalidHeader {
                message: format!(
                    "Failed to convert Node X25519 public key bytes to 32-byte array, incorrect length, with error: {:?}",
                    e
                ),
                endpoint: endpoint.to_string(),
            }
        })?;
        let ciphertext_bytes = STANDARD
            .decode(&confidential_compute_request.ciphertext)
            .map_err(|e| AtomaServiceError::InvalidBody {
                message: format!(
                    "Failed to decode ciphertext from base64 encoding, with error: {e}"
                ),
                endpoint: endpoint.to_string(),
            })?;
        let confidential_compute_decryption_request = ConfidentialComputeDecryptionRequest {
            ciphertext: ciphertext_bytes,
            nonce: nonce_bytes,
            salt: salt_bytes,
            client_dh_public_key: client_dh_public_key_bytes,
            node_dh_public_key: node_dh_public_key_bytes,
        };
        let (result_sender, result_receiver) = oneshot::channel();
        state
            .decryption_sender
            .send((confidential_compute_decryption_request, result_sender))
            .map_err(|e| AtomaServiceError::InternalError {
                message: format!("Failed to send confidential compute request, with error: {e}"),
                endpoint: endpoint.to_string(),
            })?;
        let result = result_receiver
            .await
            .map_err(|e| AtomaServiceError::InternalError {
                message: format!(
                    "Failed to receive confidential compute response, with error: {e}"
                ),
                endpoint: endpoint.to_string(),
            })?;
        let plaintext = result
            .map_err(|e| AtomaServiceError::InvalidBody {
                message: format!("Failed to decrypt confidential compute request, with error: {e}"),
                endpoint: endpoint.to_string(),
            })?
            .plaintext;
        Ok(DecryptionMetadata {
            plaintext,
            client_dh_public_key: client_dh_public_key_bytes,
            salt: salt_bytes,
        })
    }

    /// Checks if the computed plaintext body hash matches the expected plaintext body hash.
    ///
    /// This function performs a hash comparison between the computed plaintext body hash and the expected plaintext body hash.
    ///
    /// # Arguments
    /// * `plaintext_body_hash_bytes` - The expected plaintext body hash as a 32-byte array
    /// * `plaintext` - The plaintext to hash and compare
    /// * `endpoint` - The API endpoint path being accessed (used for error context)
    ///
    /// # Returns
    /// * `Ok(())` - If the hashes match
    /// * `Err(AtomaServiceError)` - If the hashes do not match
    ///
    /// # Errors
    /// This function will return an error if:
    /// - The computed plaintext body hash cannot be converted to a 32-byte array
    /// - The computed plaintext body hash does not match the expected plaintext body hash
    ///
    /// # Example
    /// ```rust,ignore
    /// use atoma_service::middleware::utils;
    ///
    /// let plaintext_body_hash_bytes = [0u8; 32]; // Your 32-byte hash
    /// let plaintext = "Hello, world!".as_bytes();
    /// let endpoint = "/v1/chat/completions";
    ///
    /// match utils::check_plaintext_body_hash(&plaintext_body_hash_bytes, plaintext, endpoint) {
    ///     Ok(()) => println!("Plaintext body hash matches"),
    ///     Err(e) => eprintln!("Plaintext body hash does not match: {}", e),
    /// }
    /// ```
    #[instrument(level = "trace", skip_all, err)]
    pub fn check_plaintext_body_hash(
        plaintext_body_hash_bytes: [u8; PAYLOAD_HASH_SIZE],
        plaintext: &[u8],
        endpoint: &str,
    ) -> Result<(), AtomaServiceError> {
        let computed_plaintext_body_hash: [u8; PAYLOAD_HASH_SIZE] = blake2b_hash(plaintext).into();
        if computed_plaintext_body_hash != plaintext_body_hash_bytes {
            return Err(AtomaServiceError::InvalidBody {
                message: format!(
                    "Plaintext body hash does not match, computed hash: {:?}, expected hash: {:?}",
                    computed_plaintext_body_hash, plaintext_body_hash_bytes
                ),
                endpoint: endpoint.to_string(),
            });
        }
        Ok(())
    }

    /// Checks if a given model is currently flagged for "too many requests" and,
    /// if so, whether it should be unflagged based on a timeout or current service metrics.
    ///
    /// This function implements a cooldown mechanism for models that have recently
    /// triggered a "too many requests" (429) status.
    ///
    /// # Arguments
    ///
    /// * `state`: A reference to the application's shared state (`AppState`), which includes:
    ///     - `too_many_requests`: A `DashMap` tracking models currently in a "too many requests" state and when they entered it.
    ///     - `too_many_requests_timeout_ms`: The duration (in milliseconds) a model stays flagged before its status is re-evaluated based on metrics.
    ///     - `chat_completions_service_urls`: A `DashMap` containing the service URLs for different models.
    ///     - `memory_lower_threshold`: A threshold used to determine if a service's memory usage is low enough to consider it recovered.
    /// * `model`: The name of the model to check.
    /// * `endpoint`: The API endpoint path where the request was received (used for error reporting).
    ///
    /// # Returns
    ///
    /// * `Ok(())`: If the model is not currently restricted, or if it was restricted but has now been unflagged.
    /// * `Err(AtomaServiceError::ChatCompletionsServiceUnavailable)`: If the model is currently flagged for "too many requests" and the timeout period has not yet elapsed.
    /// * `Err(AtomaServiceError::InternalError)`: If there's an issue fetching service URLs or metrics.
    ///
    /// # Logic Flow
    ///
    /// 1.  **Initial "Too Many Requests" Check:**
    ///     - It first attempts to access the `model` in the `state.too_many_requests` map using `entry()`.
    ///     - If the model is found (Occupied entry):
    ///         - It retrieves the `Instant` when the model was flagged.
    ///         - It calculates the time elapsed since flagging.
    ///         - If `elapsed_ms` is less than `state.too_many_requests_timeout_ms`, the function immediately
    ///           returns `Err(AtomaServiceError::ChatCompletionsServiceUnavailable)`, indicating the model is still in a cooldown period.
    ///         - If the timeout has passed, the entry for the model is removed from `state.too_many_requests`. This effectively
    ///           clears the "too many requests" flag based on the timeout, regardless of current metrics at this stage.
    ///     - If the model is not found (Vacant entry), it means the model isn't currently flagged for "too many requests" from a previous direct 429 response.
    ///
    /// 2.  **Metrics-Based Re-evaluation (if not returned early):**
    ///     - The function proceeds to fetch the service URLs for the given `model`.
    ///     - It then asynchronously calls `get_all_metrics` to retrieve current operational metrics for these services.
    ///     - It checks if any of the retrieved metrics indicate that the service's memory usage is now below `state.memory_lower_threshold`.
    ///
    /// 3.  **Final "Too Many Requests" State Update:**
    ///     - If the metrics show that the service is under the lower memory threshold (indicating potential recovery):
    ///         - It attempts to remove the `model` from `state.too_many_requests` again. This handles cases where the model might have been
    ///           re-added by another concurrent request between the initial check and metrics retrieval, or if it was never there but
    ///           the metrics now allow it.
    ///     - If the metrics do not show the service is under the lower threshold, no further action is taken on the `too_many_requests` map at this point
    ///       (it might have been removed by timeout earlier, or was never there).
    ///
    /// 4.  The function then returns `Ok(())` if it hasn't returned an error earlier.
    ///
    /// # Deadlock Safety
    ///
    /// The function is designed to be deadlock-safe with respect to `DashMap` operations:
    /// - The lock acquired by `state.too_many_requests.entry()` is released before any `.await` point.
    /// - Subsequent operations on `state.too_many_requests` (like the second `remove` call) acquire new, independent locks.
    #[instrument(level = "info", skip_all, err)]
    pub async fn check_if_too_many_requests(
        state: &AppState,
        model: &str,
        endpoint: &str,
    ) -> Result<(), AtomaServiceError> {
        match state.too_many_requests.entry(model.to_string()) {
            dashmap::mapref::entry::Entry::Occupied(occupied_entry) => {
                let elapsed_ms = occupied_entry.get().elapsed().as_millis();

                if elapsed_ms < state.too_many_requests_timeout_ms {
                    tracing::info!(
                            target = "atoma-service",
                            level = "info",
                            "Too many requests for model: {model}, endpoint: {endpoint}, elapsed trigger time: {elapsed_ms} and timeout: {}",
                            state.too_many_requests_timeout_ms
                        );
                    return Err(AtomaServiceError::ChatCompletionsServiceUnavailable {
                        message: "Too many requests".to_string(),
                        endpoint: endpoint.to_string(),
                    });
                }
                occupied_entry.remove();
            }
            dashmap::mapref::entry::Entry::Vacant(_) => {
                tracing::debug!(
                    target = "atoma-service",
                    level = "debug",
                    "Model is not in the `too_many_requests` map, so no action is needed here. Processing can continue."
                );
            }
        }
        let chat_completions_service_urls = state
                .chat_completions_service_urls
                .get(&model.to_lowercase())
                .ok_or_else(|| {
                    AtomaServiceError::InternalError {
                        message: format!(
                            "Chat completions service URL not found, likely that model is not supported by the current node: {}",
                            model
                        ),
                        endpoint: endpoint.to_string(),
                    }
                })?;
        let metrics = get_all_metrics(chat_completions_service_urls, model)
            .await
            .map_err(|e| AtomaServiceError::InternalError {
                message: format!("Failed to get metrics for model {model}, with error: {e}"),
                endpoint: endpoint.to_string(),
            })?;
        if metrics
            .iter()
            .any(|metric| metric.under_lower_threshold(state.memory_lower_threshold))
        {
            state.too_many_requests.remove(model);
            tracing::debug!(
                    target = "atoma-service",
                    level = "debug",
                    "Model {} is in the `too_many_requests` map, but metrics indicate that it is no longer exceeding the lower threshold. Removing from the map.",
                    model
                );
        } else {
            tracing::debug!(
                    target = "atoma-service",
                    level = "debug",
                    "Model {} is in the `too_many_requests` map, but metrics indicate that it is still exceeding the lower threshold. Processing can continue.",
                    model
                );
        }
        Ok(())
    }
}
