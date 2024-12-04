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
    pub const STACK_SMALL_ID: &str = "X-Stack-Small-Id";
    pub const SIGNATURE: &str = "X-Signature";
    pub const NONCE: &str = "X-Nonce";
    pub const SALT: &str = "X-Salt";
    pub const TX_DIGEST: &str = "X-Tx-Digest";
    pub const NODE_X25519_PUBLIC_KEY: &str = "X-Node-X25519-PublicKey";
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
#[instrument(level = "trace", skip_all)]
pub fn verify_signature(base64_signature: &str, body_hash: &[u8; 32]) -> Result<(), StatusCode> {
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

pub mod test {
    pub const POSTGRES_TEST_DB_URL: &str = "postgres://atoma:atoma@localhost:5432/atoma";
}
