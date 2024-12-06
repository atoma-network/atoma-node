pub(crate) mod chat_completions;
pub(crate) mod embeddings;
pub(crate) mod image_generations;
pub(crate) mod prometheus;

use atoma_confidential::types::{
    ConfidentialComputeEncryptionRequest, ConfidentialComputeEncryptionResponse,
};
use atoma_utils::hashing::blake2b_hash;
use axum::http::StatusCode;
use serde_json::{json, Value};
use tracing::{error, info, instrument};

use crate::{
    middleware::EncryptionMetadata,
    server::{utils, AppState},
};
use atoma_state::types::AtomaAtomaStateManagerEvent;

/// Updates response signature and stack hash state
///
/// # Arguments
///
/// * `response_body` - Mutable reference to the response JSON
/// * `payload_hash` - Hash of the original request payload
/// * `state` - Application state containing keystore and state manager
/// * `stack_small_id` - Identifier for the current stack
///
/// # Returns
///
/// Returns Result<(), StatusCode> indicating success or failure
#[instrument(
    level = "info",
    skip(response_body, state),
    fields(event = "sign-response-and-update-stack-hash",)
)]
async fn sign_response_and_update_stack_hash(
    response_body: &mut Value,
    payload_hash: [u8; 32],
    state: &AppState,
    stack_small_id: i64,
) -> Result<(), StatusCode> {
    // Sign the response body byte content and add the base64 encoded signature to the response body
    let (response_hash, signature) =
        utils::sign_response_body(response_body, &state.keystore, state.address_index).map_err(
            |e| {
                error!("Error signing response body: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            },
        )?;
    response_body["signature"] = json!(signature);

    // Update the stack total hash
    let total_hash = blake2b_hash(&[payload_hash, response_hash].concat());
    let total_hash_bytes: [u8; 32] = total_hash
        .as_slice()
        .try_into()
        .expect("Invalid BLAKE2b hash length");

    // Send the update stack total hash event to the state manager
    state
        .state_manager_sender
        .send(AtomaAtomaStateManagerEvent::UpdateStackTotalHash {
            stack_small_id,
            total_hash: total_hash_bytes,
        })
        .map_err(|e| {
            error!("Error updating stack total hash: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(())
}

#[instrument(
    level = "info",
    skip(state, response_body, client_encryption_metadata),
    fields(event = "confidential-compute-encryption-response",)
)]
pub(crate) async fn handle_confidential_compute_encryption_response(
    state: &AppState,
    response_body: Value,
    client_encryption_metadata: Option<EncryptionMetadata>,
) -> Result<Value, StatusCode> {
    if let Some(EncryptionMetadata {
        proxy_x25519_public_key,
        salt,
    }) = client_encryption_metadata
    {
        info!(
            target = "atoma-service",
            event = "chat-completions-handler",
            "Confidential chat completions response: {:#?}",
            response_body
        );
        let (sender, receiver) = tokio::sync::oneshot::channel();
        state
            .encryption_sender
            .send((
                ConfidentialComputeEncryptionRequest {
                    plaintext: response_body.to_string().as_bytes().to_vec(),
                    salt,
                    proxy_x25519_public_key,
                },
                sender,
            ))
            .map_err(|_| {
                error!(
                    target = "atoma-service",
                    event = "chat-completions-handler",
                    "Error sending encryption request"
                );
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
        let ConfidentialComputeEncryptionResponse { ciphertext, nonce } =
            receiver.await.map_err(|_| {
                error!(
                    target = "atoma-service",
                    event = "chat-completions-handler",
                    "Error receiving encryption response"
                );
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
        Ok(json!({
            "nonce": nonce,
            "ciphertext": ciphertext,
        }))
    } else {
        Ok(response_body)
    }
}
