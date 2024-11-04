use std::str::FromStr;

use crate::server::AppState;
use atoma_state::types::AtomaAtomaStateManagerEvent;
use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use blake2::{
    digest::generic_array::{typenum::U32, GenericArray},
    Blake2b, Digest,
};
use serde_json::Value;
use sui_sdk::types::crypto::{Signature, SuiSignature};
use tokio::sync::oneshot;
use tracing::{error, instrument};

/// Body size limit for signature verification (contains the body size of the request)
const MAX_BODY_SIZE: usize = 1024 * 1024; // 1MB
const MAX_COMPLETION_TOKENS: &str = "max_completion_tokens";
const MESSAGES: &str = "messages";
const MODEL: &str = "model";

/// Metadata extracted from the request
#[derive(Clone, Debug, Default)]
pub struct RequestMetadata {
    /// The stack small ID
    pub stack_small_id: i64,
    /// The estimated total number of tokens
    pub estimated_total_tokens: i64,
    /// The payload hash
    pub payload_hash: [u8; 32],
}

impl RequestMetadata {
    /// Create a new `RequestMetadata` with the given stack info
    pub fn with_stack_info(mut self, stack_small_id: i64, estimated_total_tokens: i64) -> Self {
        self.stack_small_id = stack_small_id;
        self.estimated_total_tokens = estimated_total_tokens;
        self
    }

    /// Create a new `RequestMetadata` with the given payload hash
    pub fn with_payload_hash(mut self, payload_hash: [u8; 32]) -> Self {
        self.payload_hash = payload_hash;
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
/// - `X-PublicKey`: The public key used for verification, base64 encoded.
/// - `X-Scheme`: The signature scheme used (e.g., "ed25519").
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
#[instrument(level = "trace", skip_all)]
pub async fn signature_verification_middleware(
    req: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // Skip verification for swagger-ui requests
    #[cfg(debug_assertions)]
    if req.uri().path().starts_with("/swagger-ui") || req.uri().path().starts_with("/api-docs") {
        return Ok(next.run(req).await);
    }

    let (mut req_parts, req_body) = req.into_parts();
    let base64_signature = req_parts
        .headers
        .get("X-Signature")
        .ok_or_else(|| {
            error!("Signature header not found");
            StatusCode::BAD_REQUEST
        })?
        .to_str()
        .map_err(|e| {
            error!("Failed to exract base64 signature encoding, with error: {e}");
            StatusCode::BAD_REQUEST
        })?;
    let body_bytes = axum::body::to_bytes(req_body, MAX_BODY_SIZE)
        .await
        .map_err(|_| {
            error!("Failed to convert body to bytes");
            StatusCode::BAD_REQUEST
        })?;
    let mut blake2b_hash = Blake2b::new();
    blake2b_hash.update(&body_bytes);
    let body_blake2b_hash: GenericArray<u8, U32> = blake2b_hash.finalize();
    let body_blake2b_hash_bytes: [u8; 32] = body_blake2b_hash
        .as_slice()
        .try_into()
        .expect("Invalid Blake2b hash length");

    utils::verify_signature(base64_signature, &body_blake2b_hash_bytes)?;

    let request_metadata = req_parts
        .extensions
        .get::<RequestMetadata>()
        .cloned()
        .unwrap_or_default()
        .with_payload_hash(body_blake2b_hash_bytes);
    req_parts.extensions.insert(request_metadata);
    let req = Request::from_parts(req_parts, Body::from(body_bytes));

    Ok(next.run(req).await)
}

/// Middleware for verifying stack permissions and token usage.
///
/// This middleware performs several checks to ensure that the incoming request
/// is authorized to use the specified model and has sufficient compute units available.
///
/// # Steps:
/// 1. Extracts and validates the public key and stack ID from request headers.
/// 2. Parses the request body to extract the model and messages.
/// 3. Verifies that the requested model is supported.
/// 4. Calculates the total number of tokens required for the request.
/// 5. Checks if the user has an available stack with sufficient compute units.
///
/// # Headers
/// The middleware expects the following custom headers:
/// - `X-PublicKey`: The public key of the user, base64 encoded.
/// - `X-Stack-Small-Id`: The ID of the stack being used for this request.
///
/// # Request Body
/// The body should be a JSON object containing:
/// - `model`: The name of the AI model to be used.
/// - `messages`: An array of message objects, each containing a "content" field.
/// - `max_completion_tokens`: The maximum number of tokens for the AI's response.
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
#[instrument(level = "trace", skip_all)]
pub async fn verify_stack_permissions(
    state: State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // Skip verification for swagger-ui requests
    #[cfg(debug_assertions)]
    if req.uri().path().starts_with("/swagger-ui") || req.uri().path().starts_with("/api-docs") {
        return Ok(next.run(req).await);
    }

    let (mut req_parts, req_body) = req.into_parts();
    let base64_signature = req_parts
        .headers
        .get("X-Signature")
        .ok_or_else(|| {
            error!("Signature header not found");
            StatusCode::BAD_REQUEST
        })?
        .to_str()
        .map_err(|_| {
            error!("Failed to convert signature to string");
            StatusCode::BAD_REQUEST
        })?;
    let signature = Signature::from_str(base64_signature).map_err(|_| {
        error!("Failed to parse signature");
        StatusCode::BAD_REQUEST
    })?;
    let public_key_bytes = signature.public_key_bytes();
    let public_key_hex = format!("0x{}", hex::encode(public_key_bytes));
    let stack_small_id = req_parts.headers.get("X-Stack-Small-Id").ok_or_else(|| {
        error!("Stack ID header not found");
        StatusCode::BAD_REQUEST
    })?;
    let stack_small_id = stack_small_id
        .to_str()
        .map_err(|_| {
            error!("Stack small ID cannot be converted to a string");
            StatusCode::BAD_REQUEST
        })?
        .parse::<i64>()
        .map_err(|_| {
            error!("Stack small ID is not a valid integer");
            StatusCode::BAD_REQUEST
        })?;
    let body_bytes = axum::body::to_bytes(req_body, MAX_BODY_SIZE)
        .await
        .map_err(|_| {
            error!("Failed to convert body to bytes");
            StatusCode::BAD_REQUEST
        })?;
    let body_json: Value = serde_json::from_slice(&body_bytes).map_err(|_| {
        error!("Failed to parse body as JSON");
        StatusCode::BAD_REQUEST
    })?;
    let model = body_json
        .get(MODEL)
        .ok_or_else(|| {
            error!("Model not found in body");
            StatusCode::BAD_REQUEST
        })?
        .as_str()
        .ok_or_else(|| {
            error!("Model is not a string");
            StatusCode::BAD_REQUEST
        })?;
    if !state.models.contains(&model.to_string()) {
        error!("Model not supported");
        return Err(StatusCode::BAD_REQUEST);
    }
    let messages = body_json
        .get(MESSAGES)
        .ok_or_else(|| {
            error!("Messages not found in body");
            StatusCode::BAD_REQUEST
        })?
        .as_array()
        .ok_or_else(|| {
            error!("Messages is not an array");
            StatusCode::BAD_REQUEST
        })?;

    let tokenizer_index = state
        .models
        .iter()
        .position(|m| m == model)
        .ok_or_else(|| {
            error!("Model not supported");
            StatusCode::BAD_REQUEST
        })?;
    let mut total_num_tokens = 0;
    for message in messages {
        let content = message.get("content").ok_or_else(|| {
            error!("Message content not found");
            StatusCode::BAD_REQUEST
        })?;
        let content_str = content.as_str().ok_or_else(|| {
            error!("Message content is not a string");
            StatusCode::BAD_REQUEST
        })?;
        let num_tokens = state.tokenizers[tokenizer_index]
            .encode(content_str, true)
            .map_err(|_| {
                error!("Failed to encode message content");
                StatusCode::BAD_REQUEST
            })?
            .get_ids()
            .len() as i64;
        total_num_tokens += num_tokens;
        // add 2 tokens as a safety margin, for start and end message delimiters
        total_num_tokens += 2;
        // add 1 token as a safety margin, for the role name of the message
        total_num_tokens += 1;
    }

    total_num_tokens += body_json
        .get(MAX_COMPLETION_TOKENS)
        .and_then(|value| value.as_i64())
        .ok_or_else(|| {
            error!("Max completion tokens not found in body");
            StatusCode::BAD_REQUEST
        })?;

    let (result_sender, result_receiver) = oneshot::channel();
    state
        .state_manager_sender
        .send(
            AtomaAtomaStateManagerEvent::GetAvailableStackWithComputeUnits {
                stack_small_id,
                public_key: public_key_hex,
                total_num_tokens,
                result_sender,
            },
        )
        .map_err(|err| {
            error!("Failed to get available stacks: {}", err);
            StatusCode::UNAUTHORIZED
        })?;
    let available_stack = result_receiver
        .await
        .map_err(|_| {
            error!("Failed to get available stack with enough compute units");
            StatusCode::UNAUTHORIZED
        })?
        .map_err(|err| {
            error!(
                "Failed to get available stack with enough compute units: {}",
                err
            );
            StatusCode::UNAUTHORIZED
        })?;
    if available_stack.is_none() {
        error!("No available stack with enough compute units");
        return Err(StatusCode::UNAUTHORIZED);
    }
    let request_metadata =
        RequestMetadata::default().with_stack_info(stack_small_id, total_num_tokens);
    req_parts.extensions.insert(request_metadata);
    let req = Request::from_parts(req_parts, Body::from(body_bytes));
    Ok(next.run(req).await)
}

pub(crate) mod utils {
    use super::*;
    use fastcrypto::{
        ed25519::{Ed25519PublicKey, Ed25519Signature},
        secp256k1::{Secp256k1PublicKey, Secp256k1Signature},
        secp256r1::{Secp256r1PublicKey, Secp256r1Signature},
        traits::{ToFromBytes, VerifyingKey},
    };
    use sui_sdk::types::crypto::{PublicKey, SignatureScheme, SuiSignature};

    pub(crate) fn verify_signature(
        base64_signature: &str,
        body_hash: &[u8; 32],
    ) -> Result<(), StatusCode> {
        let signature = Signature::from_str(base64_signature).map_err(|_| {
            error!("Failed to parse signature");
            StatusCode::BAD_REQUEST
        })?;
        let signature_bytes = signature.signature_bytes();
        let public_key_bytes = signature.public_key_bytes();
        let signature_scheme = signature.scheme();
        let public_key =
            PublicKey::try_from_bytes(signature_scheme, public_key_bytes).map_err(|e| {
                error!("Failed to extract public key from bytes, with error: {e}");
                StatusCode::BAD_REQUEST
            })?;

        match signature_scheme {
            SignatureScheme::ED25519 => {
                let public_key = Ed25519PublicKey::from_bytes(public_key.as_ref()).unwrap();
                let signature = Ed25519Signature::from_bytes(signature_bytes).unwrap();
                public_key.verify(body_hash, &signature).map_err(|_| {
                    error!("Failed to verify signature");
                    StatusCode::UNAUTHORIZED
                })?;
            }
            SignatureScheme::Secp256k1 => {
                let public_key = Secp256k1PublicKey::from_bytes(public_key.as_ref()).unwrap();
                let signature = Secp256k1Signature::from_bytes(signature_bytes).unwrap();
                public_key.verify(body_hash, &signature).map_err(|_| {
                    error!("Failed to verify signature");
                    StatusCode::UNAUTHORIZED
                })?;
            }
            SignatureScheme::Secp256r1 => {
                let public_key = Secp256r1PublicKey::from_bytes(public_key.as_ref()).unwrap();
                let signature = Secp256r1Signature::from_bytes(signature_bytes).unwrap();
                public_key.verify(body_hash, &signature).map_err(|_| {
                    error!("Failed to verify signature");
                    StatusCode::UNAUTHORIZED
                })?;
            }
            _ => {
                error!("Currently unsupported signature scheme");
                return Err(StatusCode::BAD_REQUEST);
            }
        }
        Ok(())
    }
}
