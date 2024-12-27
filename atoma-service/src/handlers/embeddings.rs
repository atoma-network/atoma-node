use crate::{
    error::AtomaServiceError,
    handlers::{
        handle_confidential_compute_encryption_response,
        prometheus::{TEXT_EMBEDDINGS_LATENCY_METRICS, TEXT_EMBEDDINGS_NUM_REQUESTS},
        sign_response_and_update_stack_hash,
    },
    middleware::RequestMetadata,
    server::AppState,
};
use axum::{extract::State, Extension, Json};
use reqwest::Client;
use serde_json::Value;
use tracing::{info, instrument};
use utoipa::OpenApi;

/// The path for confidential embeddings requests
pub const CONFIDENTIAL_EMBEDDINGS_PATH: &str = "/v1/confidential/embeddings";

/// The path for embeddings requests
pub const EMBEDDINGS_PATH: &str = "/v1/embeddings";

/// OpenAPI documentation structure for the embeddings endpoint.
///
/// This struct defines the OpenAPI (Swagger) documentation for the embeddings API,
/// including all request and response schemas. It uses the `utoipa` framework to generate
/// the API documentation.
#[derive(OpenApi)]
#[openapi(paths(embeddings_handler))]
pub(crate) struct EmbeddingsOpenApi;

/// Create embeddings
///
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
/// Returns a `AtomaServiceError::InternalError` if:
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
    fields(path = request_metadata.endpoint_path)
)]
pub async fn embeddings_handler(
    Extension(request_metadata): Extension<RequestMetadata>,
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, AtomaServiceError> {
    info!("Received embeddings request, with payload: {payload}");
    let model = payload
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");
    let endpoint = request_metadata.endpoint_path.clone();

    TEXT_EMBEDDINGS_NUM_REQUESTS
        .with_label_values(&[model])
        .inc();
    let timer = TEXT_EMBEDDINGS_LATENCY_METRICS
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
            state.chat_completions_service_url, EMBEDDINGS_PATH
        ))
        .json(&payload)
        .send()
        .await
        .map_err(|e| AtomaServiceError::InternalError {
            message: format!("Error sending request to embeddings service: {}", e),
            num_processed_tokens: 0,
            endpoint: endpoint.clone(),
        })?;
    let mut response_body =
        response
            .json::<Value>()
            .await
            .map_err(|e| AtomaServiceError::InternalError {
                message: format!("Error reading response body: {}", e),
                num_processed_tokens: 0,
                endpoint: endpoint.clone(),
            })?;

    // Sign the response and update the stack hash
    if let Err(e) = sign_response_and_update_stack_hash(
        &mut response_body,
        payload_hash,
        &state,
        stack_small_id,
        endpoint.clone(),
    )
    .await
    {
        return Err(AtomaServiceError::InternalError {
            message: format!("Error signing response and updating stack hash: {}", e),
            num_processed_tokens: 0,
            endpoint: endpoint.clone(),
        });
    }

    // Handle confidential compute encryption response
    match handle_confidential_compute_encryption_response(
        &state,
        response_body,
        client_encryption_metadata,
        endpoint.clone(),
    )
    .await
    {
        Ok(response_body) => {
            // Stop the timer before returning the valid response
            timer.observe_duration();
            Ok(Json(response_body))
        }
        Err(e) => Err(AtomaServiceError::InternalError {
            message: format!(
                "Error handling confidential compute encryption response: {}",
                e
            ),
            num_processed_tokens: 0,
            endpoint,
        }),
    }
}
