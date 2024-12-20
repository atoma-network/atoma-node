use crate::{
    handlers::prometheus::{IMAGE_GEN_LATENCY_METRICS, IMAGE_GEN_NUM_REQUESTS},
    middleware::RequestMetadata,
    server::AppState,
};
use axum::{extract::State, http::StatusCode, Extension, Json};
use reqwest::Client;
use serde_json::Value;
use tracing::{error, info, instrument};
use utoipa::OpenApi;

use super::{handle_confidential_compute_encryption_response, sign_response_and_update_stack_hash};

/// The path for confidential image generations requests
pub const CONFIDENTIAL_IMAGE_GENERATIONS_PATH: &str = "/v1/confidential/images/generations";

/// The path for image generations requests
pub const IMAGE_GENERATIONS_PATH: &str = "/v1/images/generations";

/// OpenAPI documentation structure for the image generations endpoint.
///
/// This struct defines the OpenAPI (Swagger) documentation for the image generations API,
/// including all request and response schemas. It uses the `utoipa` framework to generate
/// the API documentation.
#[derive(OpenApi)]
#[openapi(paths(image_generations_handler))]
pub(crate) struct ImageGenerationsOpenApi;

/// Create image generation
///
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
    fields(path = request_metadata.endpoint_path)
)]
pub async fn image_generations_handler(
    Extension(request_metadata): Extension<RequestMetadata>,
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, StatusCode> {
    info!("Received image generations request, with payload: {payload}");
    let model = payload
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");

    IMAGE_GEN_NUM_REQUESTS.with_label_values(&[model]).inc();
    let timer = IMAGE_GEN_LATENCY_METRICS
        .with_label_values(&[model])
        .start_timer();

    let RequestMetadata {
        stack_small_id,
        estimated_total_compute_units: _,
        payload_hash,
        client_encryption_metadata,
        request_type: _,
        endpoint_path: _,
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

    // Sign the response and update the stack hash
    if let Err(e) = sign_response_and_update_stack_hash(
        &mut response_body,
        payload_hash,
        &state,
        stack_small_id,
    )
    .await
    {
        error!(
            target = "atoma-service",
            event = "image-generations-handler",
            "Error signing response and updating stack hash: {}",
            e
        );
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Handle confidential compute encryption response
    match handle_confidential_compute_encryption_response(
        &state,
        response_body,
        client_encryption_metadata,
    )
    .await
    {
        Ok(response_body) => {
            // Stop the timer before returning the valid response
            timer.observe_duration();
            Ok(Json(response_body))
        }
        Err(e) => {
            error!(
                target = "atoma-service",
                event = "image-generations-handler",
                "Error handling confidential compute encryption response: {}",
                e
            );
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
