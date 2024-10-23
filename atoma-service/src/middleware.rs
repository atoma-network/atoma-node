use std::str::FromStr;

use crate::{authentication::SignatureScheme, server::AppState};
use atoma_state::StateManager;
use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use base64::{prelude::BASE64_STANDARD, Engine};
use serde_json::Value;
use sha2::Digest;
use tracing::{error, instrument};

/// Body size limit for signature verification (contains the body size of the request)
const MAX_BODY_SIZE: usize = 1024 * 1024; // 1MB
const MAX_COMPLETION_TOKENS: &str = "max_completion_tokens";
const MESSAGES: &str = "messages";
const MODEL: &str = "model";

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
    let (mut req_parts, req_body) = req.into_parts();
    let signature = req_parts.headers.get("X-Signature").ok_or_else(|| {
        error!("Signature header not found");
        StatusCode::BAD_REQUEST
    })?;
    let public_key = req_parts.headers.get("X-PublicKey").ok_or_else(|| {
        error!("Public key header not found");
        StatusCode::BAD_REQUEST
    })?;
    let signature_scheme = req_parts.headers.get("X-Scheme").ok_or_else(|| {
        error!("Signature scheme header not found");
        StatusCode::BAD_REQUEST
    })?;
    let signature_scheme_str = signature_scheme
        .to_str()
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let public_key_bytes = Engine::decode(&BASE64_STANDARD, public_key).map_err(|_| {
        error!("Failed to decode public key");
        StatusCode::BAD_REQUEST
    })?;

    let signature_bytes = Engine::decode(&BASE64_STANDARD, signature).map_err(|_| {
        error!("Failed to decode signature");
        StatusCode::BAD_REQUEST
    })?;
    let signature_scheme = SignatureScheme::from_str(signature_scheme_str).map_err(|_| {
        error!("Failed to parse signature scheme");
        StatusCode::BAD_REQUEST
    })?;

    let body_bytes = axum::body::to_bytes(req_body, MAX_BODY_SIZE)
        .await
        .map_err(|_| {
            error!("Failed to convert body to bytes");
            StatusCode::BAD_REQUEST
        })?;
    let body_sha256_hash = sha2::Sha256::digest(&body_bytes);
    let body_sha256_hash_bytes: [u8; 32] = body_sha256_hash
        .as_slice()
        .try_into()
        .expect("Invalid SHA256 hash length");

    signature_scheme
        .verify(&body_sha256_hash_bytes, &signature_bytes, &public_key_bytes) // Signature second, public key third
        .map_err(|_| {
            error!(
                "Failed to verify signature scheme: {}",
                signature_scheme_str
            );
            StatusCode::UNAUTHORIZED
        })?;

    req_parts.extensions.insert(body_sha256_hash_bytes);
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
/// - `X-Stack-Id`: The ID of the stack being used for this request.
///
/// # Request Body
/// The body should be a JSON object containing:
/// - `model`: The name of the AI model to be used.
/// - `messages`: An array of message objects, each containing a "content" field.
/// - `max_completion_tokens`: The maximum number of tokens for the AI's response.
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
    let (mut req_parts, req_body) = req.into_parts();
    let public_key = req_parts.headers.get("X-PublicKey").ok_or_else(|| {
        error!("Public key header not found");
        StatusCode::BAD_REQUEST
    })?;
    let public_key_hex = Engine::decode(&BASE64_STANDARD, public_key)
        .map_err(|_| {
            error!("Failed to decode public key");
            StatusCode::BAD_REQUEST
        })
        .map(|bytes| {
            let encoding = hex::encode(bytes);
            format!("0x{}", encoding)
        })?;
    let stack_small_id = req_parts.headers.get("X-Stack-Id").ok_or_else(|| {
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
        let num_tokens = state
            .tokenizer
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

    let state_manager = StateManager::new(state.state.clone());
    let available_stack = state_manager
        .get_available_stack_with_compute_units(stack_small_id, &public_key_hex, total_num_tokens)
        .await
        .map_err(|err| {
            error!("Failed to get available stacks: {}", err);
            StatusCode::UNAUTHORIZED
        })?;
    if available_stack.is_none() {
        error!("No available stack with enough compute units");
        return Err(StatusCode::UNAUTHORIZED);
    }
    req_parts
        .extensions
        .insert((stack_small_id, total_num_tokens));
    let req = Request::from_parts(req_parts, Body::from(body_bytes));
    Ok(next.run(req).await)
}
