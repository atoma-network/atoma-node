use crate::server::AppState;
use axum::{extract::State, http::StatusCode, Json};
use reqwest::Client;
use serde_json::Value;
use tracing::{error, instrument};
use utoipa::OpenApi;

pub const IMAGE_GENERATIONS_PATH: &str = "/v1/images/generations";

#[derive(OpenApi)]
#[openapi(paths(image_generations_handler))]
pub(crate) struct ImageGenerationsOpenApi;

/// Handles image generation requests by proxying them to the image generations service.
///
/// This handler simply forwards the request to the image generations service and returns the response.
///
/// # Arguments
///
/// * `state` - Application state containing service URLs
/// * `payload` - The image generation request body
///
/// # Returns
///
/// Returns the JSON response from the image generations service
///
/// # Errors
///
/// Returns a `StatusCode::INTERNAL_SERVER_ERROR` if:
/// - The image generations service request fails
/// - Response parsing fails
#[utoipa::path(
    post,
    path = "",
    tag = "images",
    request_body = Value,
    responses(
        (status = OK, description = "Images generated successfully", body = Value),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
#[instrument(
    level = "info",
    skip(_state, payload),
    fields(path = IMAGE_GENERATIONS_PATH)
)]
pub async fn image_generations_handler(
    State(_state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, StatusCode> {
    let client = Client::new();
    let response = client
        .post("http://image-generator:3000/v1/images/generations")
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            error!("Error sending request to image generations service: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let response_body = response.json::<Value>().await.map_err(|e| {
        error!("Error reading response body: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(response_body))
}
