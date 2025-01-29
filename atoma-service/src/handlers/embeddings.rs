use crate::{
    error::AtomaServiceError,
    handlers::{
        handle_confidential_compute_encryption_response,
        prometheus::{
            TEXT_EMBEDDINGS_LATENCY_METRICS, TEXT_EMBEDDINGS_NUM_REQUESTS,
            TOTAL_COMPLETED_REQUESTS, TOTAL_FAILED_REQUESTS,
        },
        sign_response_and_update_stack_hash, update_stack_num_compute_units,
    },
    middleware::{EncryptionMetadata, RequestMetadata},
    server::AppState,
    types::{ConfidentialComputeRequest, ConfidentialComputeResponse},
};
use axum::{extract::State, Extension, Json};
use prometheus::HistogramTimer;
use reqwest::Client;
use serde_json::Value;
use tracing::{info, instrument};
use utoipa::OpenApi;

use super::handle_status_code_error;

/// The path for confidential embeddings requests
pub const CONFIDENTIAL_EMBEDDINGS_PATH: &str = "/v1/confidential/embeddings";

/// The path for embeddings requests
pub const EMBEDDINGS_PATH: &str = "/v1/embeddings";

/// The key for the model parameter in the request body
pub const MODEL_KEY: &str = "model";

/// OpenAPI documentation structure for the embeddings endpoint.
///
/// This struct defines the OpenAPI (Swagger) documentation for the embeddings API,
/// including all request and response schemas. It uses the `utoipa` framework to generate
/// the API documentation.
#[derive(OpenApi)]
#[openapi(paths(embeddings_handler))]
pub struct EmbeddingsOpenApi;

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
        .get(MODEL_KEY)
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");
    let RequestMetadata {
        stack_small_id,
        estimated_total_compute_units,
        payload_hash,
        client_encryption_metadata,
        endpoint_path: endpoint,
        request_type: _,
    } = request_metadata;

    TEXT_EMBEDDINGS_NUM_REQUESTS
        .with_label_values(&[model])
        .inc();
    let timer = TEXT_EMBEDDINGS_LATENCY_METRICS
        .with_label_values(&[model])
        .start_timer();

    let model = payload
        .get(MODEL_KEY)
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");

    match handle_embeddings_response(
        &state,
        &payload,
        stack_small_id,
        estimated_total_compute_units,
        payload_hash,
        client_encryption_metadata,
        &endpoint,
        timer,
    )
    .await
    {
        Ok(response) => {
            TOTAL_COMPLETED_REQUESTS.with_label_values(&[model]).inc();
            Ok(response)
        }
        Err(e) => {
            TOTAL_FAILED_REQUESTS.with_label_values(&[model]).inc();
            update_stack_num_compute_units(
                &state.state_manager_sender,
                stack_small_id,
                estimated_total_compute_units,
                0,
                &endpoint,
            )?;
            Err(e)
        }
    }
}

/// OpenAPI documentation structure for the embeddings endpoint.
///
/// This struct defines the OpenAPI (Swagger) documentation for the embeddings API,
/// including all request and response schemas. It uses the `utoipa` framework to generate
/// the API documentation.
#[derive(OpenApi)]
#[openapi(paths(confidential_embeddings_handler))]
pub struct ConfidentialEmbeddingsOpenApi;

/// Handler for confidential embeddings requests
///
/// This endpoint processes embedding requests with additional confidential computing guarantees.
/// It forwards the request to the embeddings service and returns an encrypted response that can
/// only be decrypted by the client.
///
/// # Arguments
///
/// * `request_metadata` - Extension containing request context like stack ID and encryption details
/// * `state` - Application state containing service URLs and shared resources
/// * `payload` - The embedding request body as JSON
///
/// # Returns
///
/// Returns a `Result` containing either:
/// * `Json<Value>` - The encrypted embeddings response
/// * `AtomaServiceError` - Error details if the request processing fails
///
/// # Errors
///
/// Returns `AtomaServiceError::InternalError` if:
/// * The embeddings service request fails
/// * Response processing or encryption fails
/// * Stack compute unit updates fail
///
/// # Example Request
///
/// ```json
/// {
///     "model": "text-embedding-ada-002",
///     "input": "The quick brown fox jumps over the lazy dog"
/// }
/// ```
#[utoipa::path(
    post,
    path = "",
    tag = "confidential-embeddings",
    request_body = ConfidentialComputeRequest,
    responses(
        (status = OK, description = "Confidential embeddings generated successfully", body = ConfidentialComputeResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
#[instrument(
    level = "info",
    skip(state, payload),
    fields(path = request_metadata.endpoint_path)
)]
pub async fn confidential_embeddings_handler(
    Extension(request_metadata): Extension<RequestMetadata>,
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, AtomaServiceError> {
    info!("Received embeddings request, with payload: {payload}");
    let model = payload
        .get(MODEL_KEY)
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");
    let RequestMetadata {
        stack_small_id,
        estimated_total_compute_units,
        payload_hash,
        client_encryption_metadata,
        endpoint_path: endpoint,
        request_type: _,
    } = request_metadata;

    TEXT_EMBEDDINGS_NUM_REQUESTS
        .with_label_values(&[model])
        .inc();
    let timer = TEXT_EMBEDDINGS_LATENCY_METRICS
        .with_label_values(&[model])
        .start_timer();

    match handle_embeddings_response(
        &state,
        &payload,
        stack_small_id,
        estimated_total_compute_units,
        payload_hash,
        client_encryption_metadata,
        &endpoint,
        timer,
    )
    .await
    {
        Ok(response) => Ok(response),
        Err(e) => {
            update_stack_num_compute_units(
                &state.state_manager_sender,
                stack_small_id,
                estimated_total_compute_units,
                0,
                &endpoint,
            )?;
            Err(e)
        }
    }
}

/// Handles the processing and transformation of embeddings requests and responses
///
/// This function serves as the core processing pipeline for embeddings requests, performing
/// several key operations in sequence:
///
/// 1. Forwards the original request to the embeddings service
/// 2. Processes the response through signature verification and stack hash updates
/// 3. Applies confidential compute encryption if required
/// 4. Tracks request timing metrics
///
/// # Arguments
///
/// * `state` - Application state containing service URLs and shared resources
/// * `payload` - The original embedding request payload to be forwarded
/// * `stack_small_id` - Unique identifier for the stack making the request
/// * `estimated_total_compute_units` - Expected computational cost of the request
/// * `payload_hash` - 32-byte hash of the original request payload
/// * `client_encryption_metadata` - Optional encryption details for confidential compute
/// * `endpoint` - The API endpoint path being called
/// * `timer` - Prometheus histogram timer for tracking request duration
///
/// # Returns
///
/// Returns a `Result` containing either:
/// * `Json<Value>` - The processed response, potentially encrypted for confidential compute
/// * `AtomaServiceError` - Error details if any step in the pipeline fails
///
/// # Errors
///
/// Returns `AtomaServiceError::InternalError` if:
/// * Network request to the embeddings service fails
/// * Response parsing encounters an error
/// * Response signing or stack hash updates fail
/// * Confidential compute encryption processing fails
///
/// # Example
///
/// ```rust,ignore
/// let response = handle_embeddings_response(
///     &app_state,
///     &request_payload,
///     stack_id,
///     compute_units,
///     payload_hash,
///     encryption_metadata,
///     "/v1/embeddings",
///     metrics_timer
/// ).await?;
/// ```
#[instrument(
    level = "info",
    skip(state, payload),
    fields(path = endpoint)
)]
#[allow(clippy::too_many_arguments)]
async fn handle_embeddings_response(
    state: &AppState,
    payload: &Value,
    stack_small_id: i64,
    estimated_total_compute_units: i64,
    payload_hash: [u8; 32],
    client_encryption_metadata: Option<EncryptionMetadata>,
    endpoint: &str,
    timer: HistogramTimer,
) -> Result<Json<Value>, AtomaServiceError> {
    let client = Client::new();
    let response = client
        .post(format!(
            "{}{}",
            state.embeddings_service_url, EMBEDDINGS_PATH
        ))
        .json(&payload)
        .send()
        .await
        .map_err(|e| AtomaServiceError::InternalError {
            message: format!("Error sending request to embeddings service: {}", e),
            endpoint: endpoint.to_string(),
        })?;

    if !response.status().is_success() {
        let error = response
            .status()
            .canonical_reason()
            .unwrap_or("Unknown error");
        handle_status_code_error(response.status(), endpoint, error)?;
    }

    let mut response_body =
        response
            .json::<Value>()
            .await
            .map_err(|e| AtomaServiceError::InternalError {
                message: format!("Error reading response body: {}", e),
                endpoint: endpoint.to_string(),
            })?;

    // Sign the response and update the stack hash
    if let Err(e) = sign_response_and_update_stack_hash(
        &mut response_body,
        payload_hash,
        state,
        stack_small_id,
        endpoint.to_string(),
    )
    .await
    {
        return Err(AtomaServiceError::InternalError {
            message: format!("Error signing response and updating stack hash: {}", e),
            endpoint: endpoint.to_string(),
        });
    }

    // Handle confidential compute encryption response
    match handle_confidential_compute_encryption_response(
        state,
        response_body,
        client_encryption_metadata,
        endpoint.to_string(),
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
            endpoint: endpoint.to_string(),
        }),
    }
}
