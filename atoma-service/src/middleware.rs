use std::str::FromStr;

use crate::authentication::SignatureScheme;
use axum::{
    body::Body,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use base64::{prelude::BASE64_STANDARD, Engine};
use sha2::Digest;
use tracing::{error, instrument};

/// Body size limit for signature verification (contains the body size of the request)
const MAX_BODY_SIZE: usize = 1024 * 1024; // 1MB

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
    let (req_parts, req_body) = req.into_parts();
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
    let body_sha256_hash_bytes = body_sha256_hash.as_slice();

    signature_scheme
        .verify(body_sha256_hash_bytes, &signature_bytes, &public_key_bytes) // Signature second, public key third
        .map_err(|_| {
            error!(
                "Failed to verify signature scheme: {}",
                signature_scheme_str
            );
            StatusCode::UNAUTHORIZED
        })?;

    let req = Request::from_parts(req_parts, Body::from(body_bytes));

    Ok(next.run(req).await)
}
