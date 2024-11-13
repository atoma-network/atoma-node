use crate::{middleware::RequestMetadata, server::AppState};
use axum::{extract::State, http::StatusCode, Extension, Json};
use reqwest::Client;
use serde_json::Value;
use tracing::{error, info, instrument};
use utoipa::OpenApi;

use super::sign_response_and_update_stack_hash;

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
    skip(state, payload),
    fields(path = IMAGE_GENERATIONS_PATH)
)]
pub async fn image_generations_handler(
    Extension(request_metadata): Extension<RequestMetadata>,
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, StatusCode> {
    info!("Received image generations request, with payload: {payload}");

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
            state.image_generations_service_url, IMAGE_GENERATIONS_PATH
        ))
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            error!("Error sending request to image generations service: {}", e);
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