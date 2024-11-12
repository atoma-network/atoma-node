pub(crate) mod chat_completions;
pub(crate) mod embeddings;
pub(crate) mod image_generations;

use axum::http::StatusCode;
use blake2::{
    digest::generic_array::{typenum::U32, GenericArray},
    Digest,
};
use serde_json::{json, Value};
use tracing::error;

use crate::server::{utils, AppState};
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
    let mut blake2b = blake2::Blake2b::new();
    blake2b.update([payload_hash, response_hash].concat());
    let total_hash: GenericArray<u8, U32> = blake2b.finalize();
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
