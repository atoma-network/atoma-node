use crate::server::AppState;
use axum::{extract::State, http::StatusCode, Json};
use reqwest::Client;
use serde_json::Value;
use tracing::{error, instrument};
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
    skip(_state, payload),
    fields(path = EMBEDDINGS_PATH)
)]
pub async fn embeddings_handler(
    State(_state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, StatusCode> {
    let client = Client::new();
    let response = client
        .post("http://embeddings:3000/v1/embeddings")
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            error!("Error sending request to embeddings service: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let response_body = response.json::<Value>().await.map_err(|e| {
        error!("Error reading response body: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(response_body))
}
