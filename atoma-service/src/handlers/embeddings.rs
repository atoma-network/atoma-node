use crate::{
    handlers::sign_response_and_update_stack_hash, middleware::RequestMetadata, server::AppState,
};
use axum::{extract::State, http::StatusCode, Extension, Json};
use reqwest::Client;
use serde_json::Value;
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
        estimated_total_compute_units: _,
        payload_hash,
        request_type: _,
    } = request_metadata;

    let client = Client::new();
    let response = client
        .post(format!(
            "{}{}",
            state.chat_completions_service_url, EMBEDDINGS_PATH
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

    sign_response_and_update_stack_hash(&mut response_body, payload_hash, &state, stack_small_id)
        .await?;

    Ok(Json(response_body))
}
