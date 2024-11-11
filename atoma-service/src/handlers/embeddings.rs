use crate::{
    middleware::RequestMetadata,
    server::{utils, AppState},
};
use atoma_state::types::AtomaAtomaStateManagerEvent;
use axum::{extract::State, http::StatusCode, Extension, Json};
use blake2::{
    digest::generic_array::{typenum::U32, GenericArray},
    Digest,
};
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{error, info, instrument};
use utoipa::OpenApi;

pub const EMBEDDINGS_PATH: &str = "/v1/embeddings";

#[derive(OpenApi)]
#[openapi(paths(embeddings_handler))]
pub(crate) struct EmbeddingsOpenApi;

/// Handles embedding requests by proxying them to the embeddings service.
///
/// This handler simply forwards the request to the embeddings service and returns the response.
///
/// # Arguments
///
/// * `state` - Application state containing service URLs
/// * `payload` - The embedding request body
///
/// # Returns
///
/// Returns the JSON response from the embeddings service
///
/// # Errors
///
/// Returns a `StatusCode::INTERNAL_SERVER_ERROR` if:
/// - The embeddings service request fails
/// - Response parsing fails
#[utoipa::path(
    post,
    path = "",
    tag = "embeddings",
    request_body = Value,
    responses(
        (status = OK, description = "Embeddings generated successfully", body = Value),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
#[instrument(
    level = "info",
    skip(state, payload),
    fields(path = EMBEDDINGS_PATH)
)]
pub async fn embeddings_handler(
    Extension(request_metadata): Extension<RequestMetadata>,
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, StatusCode> {
    info!("Received embeddings request, with payload: {payload}");
    let RequestMetadata {
        stack_small_id,
        estimated_total_compute_units,
        payload_hash,
    } = request_metadata;
    let client = Client::new();
    let response = client
        .post(format!(
            "{}{}",
            state.inference_service_url, EMBEDDINGS_PATH
        ))
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            error!("Error sending request to embeddings service: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    let mut response_body = response.json::<Value>().await.map_err(|e| {
        error!("Error reading response body: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Sign the response body byte content and add the base64 encoded signature to the response body
    let (response_hash, signature) =
        utils::sign_response_body(&response_body, &state.keystore, state.address_index).map_err(
            |e| {
                error!("Error signing response body: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            },
        )?;
    response_body["signature"] = json!(signature);

    let mut blake2b = blake2::Blake2b::new();
    blake2b.update([payload_hash, response_hash].concat());
    let total_hash: GenericArray<u8, U32> = blake2b.finalize();
    let total_hash_bytes: [u8; 32] = total_hash
        .as_slice()
        .try_into()
        .expect("Invalid BLAKE2b hash length");
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

    Ok(Json(response_body))
}
