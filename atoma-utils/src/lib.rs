pub mod compression;
pub mod encryption;
pub mod hashing;

use anyhow::{Context, Error, Result};
use axum::http::StatusCode;
use fastcrypto::{
    ed25519::{Ed25519PublicKey, Ed25519Signature},
    secp256k1::{Secp256k1PublicKey, Secp256k1Signature},
    secp256r1::{Secp256r1PublicKey, Secp256r1Signature},
    traits::{ToFromBytes, VerifyingKey},
};
use std::str::FromStr;
use sui_sdk::types::crypto::{PublicKey, Signature, SignatureScheme, SuiSignature};
use tokio::sync::watch;
use tracing::{error, instrument};

pub mod constants {
    /// HTTP header name for the user ID.
    /// Contains the user ID.
    pub const USER_ID: &str = "X-User-Id";

    /// HTTP header name for the Stack Small ID.
    /// Used to identify specific stacks in the system.
    pub const STACK_SMALL_ID: &str = "X-Stack-Small-Id";

    /// HTTP header name for the cryptographic signature.
    /// Contains the signature used for request authentication.
    pub const SIGNATURE: &str = "X-Signature";

    /// HTTP header name for the transaction digest.
    /// Contains a unique identifier for a blockchain transaction.
    pub const TX_DIGEST: &str = "X-Tx-Digest";

    /// HTTP header name for the request ID.
    /// Contains a unique identifier for a request.
    pub const REQUEST_ID: &str = "X-Request-Id";

    /// Field name for encrypted data in the request/response body.
    /// Contains the encrypted payload of the message.
    pub const CIPHERTEXT: &str = "ciphertext";

    /// Size of a cryptographic nonce in bytes
    pub const NONCE_SIZE: usize = 12;

    /// Size of a cryptographic salt in bytes
    pub const SALT_SIZE: usize = 16;

    /// Size of a Diffie-Hellman public key in bytes
    pub const X25519_PUBLIC_KEY_BYTES_SIZE: usize = 32;

    /// Size of a payload hash in bytes
    pub const PAYLOAD_HASH_SIZE: usize = 32;
}

/// Spawns a task that will automatically trigger shutdown if it encounters an error
///
/// This helper function wraps a future in a tokio task that monitors its execution.
/// If the wrapped future returns an error, it will automatically trigger a shutdown
/// signal through the provided sender.
///
/// # Arguments
///
/// * `f` - The future to execute, which must return a `Result<()>`
/// * `shutdown_sender` - A channel sender used to signal shutdown to other parts of the application
///
/// # Returns
///
/// Returns a `JoinHandle` for the spawned task
///
/// # Example
///
/// ```rust,ignore
/// let (shutdown_tx, shutdown_rx) = watch::channel(false);
/// let handle = spawn_with_shutdown(some_fallible_task(), shutdown_tx);
/// ```
pub fn spawn_with_shutdown<F, E>(
    f: F,
    shutdown_sender: watch::Sender<bool>,
) -> tokio::task::JoinHandle<Result<()>>
where
    E: Into<Error>,
    F: std::future::Future<Output = Result<(), E>> + Send + 'static,
{
    tokio::task::spawn(async move {
        let res = f.await;
        if res.is_err() {
            // Only send shutdown signal if the task failed
            shutdown_sender
                .send(true)
                .context("Failed to send shutdown signal")?;
        }
        res.map_err(Into::into)
    })
}

/// Verifies the authenticity of a request by checking its signature against the provided hash.
///
/// # Arguments
/// * `base64_signature` - A base64-encoded signature string that contains:
///   - The signature itself
///   - The public key
///   - The signature scheme used
/// * `body_hash` - A 32-byte Blake2b hash of the request body
///
/// # Returns
/// * `Ok(())` if the signature is valid
/// * `Err(StatusCode)` if:
///   - The signature cannot be parsed (`BAD_REQUEST`)
///   - The public key is invalid (`BAD_REQUEST`)
///   - The signature scheme is unsupported (`BAD_REQUEST`)
///   - The signature verification fails (`UNAUTHORIZED`)
///
/// # Supported Signature Schemes
/// - ED25519
/// - Secp256k1
/// - Secp256r1
///
/// # Security Note
/// This function is critical for ensuring request authenticity. It verifies that:
/// 1. The request was signed by the owner of the public key
/// 2. The request body hasn't been tampered with since signing
///
/// # Panics
/// This function panics if:
/// - The signature cannot be parsed
/// - The public key cannot be extracted
/// - The signature verification fails
///
/// # Errors
/// This function returns an error if:
/// - The signature cannot be parsed
/// - The public key cannot be extracted
/// - The signature verification fails
#[instrument(level = "trace", skip_all)]
pub fn verify_signature(
    base64_signature: &str,
    body_hash: &[u8; constants::PAYLOAD_HASH_SIZE],
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

/// Converts a JSON array of numbers into a vector of bytes.
///
/// # Arguments
///
/// * `value` - A JSON value that contains an array of numbers
/// * `field` - The field name in the JSON object that contains the byte array
///
/// # Type Parameters
///
/// * `E` - The error type that implements `std::error::Error` and can be created from a `String`
/// * `T` - Unused type parameter (consider removing if not needed)
///
/// # Returns
///
/// Returns a `Result` containing either:
/// * `Ok(Vec<u8>)` - A vector of bytes parsed from the JSON array
/// * `Err(E)` - An error if:
///   - The field doesn't exist in the JSON
///   - The field's value is not an array
///   - Any array element cannot be converted to a `u8`
///
/// # Example
///
/// ```rust,ignore
/// use serde_json::json;
///
/// let json = json!({
///     "bytes": [255, 0, 127]
/// });
///
/// let bytes: Result<Vec<u8>, String> = parse_json_byte_array(&json, "bytes");
/// assert_eq!(bytes.unwrap(), vec![255, 0, 127]);
/// ```
///
/// # Logging
///
/// This function logs errors at the trace level using the `tracing` crate,
/// including the field name in the log context.
///
/// # Panics
/// This function panics if:
/// - The field doesn't exist in the JSON
/// - The field's value is not an array
/// - Any array element cannot be converted to a `u8`
///
/// # Errors
/// This function returns an error if:
/// - The field doesn't exist in the JSON
/// - The field's value is not an array
/// - Any array element cannot be converted to a `u8`
#[instrument(
    level = "trace",
    skip_all,
    fields(field = %field)
)]
pub fn parse_json_byte_array(value: &serde_json::Value, field: &str) -> Result<Vec<u8>, String> {
    let array = value.get(field).and_then(|v| v.as_array()).ok_or_else(|| {
        error!("Error getting field array {field} from JSON");
        format!("Error getting field array {field} from JSON")
    })?;

    array
        .iter()
        .map(|b| {
            b.as_u64()
                .and_then(|u| u8::try_from(u).ok())
                .ok_or_else(|| {
                    error!("Error parsing field array {field} values as bytes from JSON");
                    format!("Error parsing field array {field} values as bytes from JSON")
                })
        })
        .collect()
}

pub mod test {
    pub const POSTGRES_TEST_DB_URL: &str = "postgres://atoma:atoma@localhost:5432/atoma";
}
