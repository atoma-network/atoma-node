use std::time::Instant;

use crate::{
    error::AtomaServiceError,
    handlers::{
        handle_concurrent_requests_count_updates,
        metrics::{
            IMAGE_GEN_CONFIDENTIAL_NUM_REQUESTS, IMAGE_GEN_LATENCY_METRICS, IMAGE_GEN_NUM_REQUESTS,
            TOTAL_COMPLETED_REQUESTS, TOTAL_FAILED_IMAGE_CONFIDENTIAL_GENERATION_REQUESTS,
            TOTAL_FAILED_IMAGE_GENERATION_REQUESTS, TOTAL_FAILED_REQUESTS,
        },
        update_stack_num_compute_units,
    },
    middleware::{EncryptionMetadata, RequestMetadata},
    server::AppState,
    types::{ConfidentialComputeRequest, ConfidentialComputeResponse},
};
use axum::{extract::State, Extension, Json};
use opentelemetry::KeyValue;
use reqwest::Client;
use serde_json::Value;
use tokenizers::Tokenizer;
use tracing::{info, instrument};
use utoipa::OpenApi;

use super::{
    handle_confidential_compute_encryption_response, handle_status_code_error,
    request_model::RequestModel, sign_response_and_update_stack_hash,
};

/// The path for confidential image generations requests
pub const CONFIDENTIAL_IMAGE_GENERATIONS_PATH: &str = "/v1/confidential/images/generations";

/// The path for image generations requests
pub const IMAGE_GENERATIONS_PATH: &str = "/v1/images/generations";

/// The key for the model parameter in the request body
pub const MODEL_KEY: &str = "model";

/// The key for the n parameter in the request body
pub const N_KEY: &str = "n";

/// The key for the size parameter in the request body
pub const SIZE_KEY: &str = "size";

/// OpenAPI documentation structure for the image generations endpoint.
///
/// This struct defines the OpenAPI (Swagger) documentation for the image generations API,
/// including all request and response schemas. It uses the `utoipa` framework to generate
/// the API documentation.
#[derive(OpenApi)]
#[openapi(paths(image_generations_handler))]
pub struct ImageGenerationsOpenApi;

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
/// Returns a `AtomaServiceError::InternalError` if:
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
) -> Result<Json<Value>, AtomaServiceError> {
    info!("Received image generations request, with payload: {payload}");
    let model = payload
        .get(MODEL_KEY)
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");

    IMAGE_GEN_NUM_REQUESTS.add(1, &[KeyValue::new("model", model.to_owned())]);
    let timer = Instant::now();

    let RequestMetadata {
        stack_small_id,
        estimated_total_compute_units,
        payload_hash,
        client_encryption_metadata,
        endpoint_path: endpoint,
        request_type: _,
    } = request_metadata;

    let model = payload
        .get(MODEL_KEY)
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");

    match handle_image_generations_response(
        &state,
        payload.clone(),
        payload_hash,
        stack_small_id,
        estimated_total_compute_units,
        client_encryption_metadata,
        &endpoint,
        timer,
        model.to_string(),
    )
    .await
    {
        Ok(response) => {
            TOTAL_COMPLETED_REQUESTS.add(1, &[KeyValue::new("model", model.to_owned())]);
            Ok(response)
        }
        Err(e) => {
            TOTAL_FAILED_REQUESTS.add(1, &[KeyValue::new("model", model.to_owned())]);
            TOTAL_FAILED_IMAGE_GENERATION_REQUESTS
                .add(1, &[KeyValue::new("model", model.to_owned())]);
            let concurrent_requests = handle_concurrent_requests_count_updates(
                &state.concurrent_requests_per_stack,
                stack_small_id,
                "image-generations/image_generations_handler",
            );
            update_stack_num_compute_units(
                &state.state_manager_sender,
                stack_small_id,
                estimated_total_compute_units,
                0,
                &endpoint,
                concurrent_requests,
            )?;
            Err(AtomaServiceError::InternalError {
                message: format!("Error handling image generations response: {}", e),
                endpoint: endpoint.to_string(),
            })
        }
    }
}

/// OpenAPI documentation structure for the confidential image generations endpoint.
///
/// This struct defines the OpenAPI (Swagger) documentation for the confidential image generations API,
/// including all request and response schemas. It uses the `utoipa` framework to generate
/// the API documentation for endpoints that handle image generation requests with confidential
/// computing requirements.
#[derive(OpenApi)]
#[openapi(paths(confidential_image_generations_handler))]
pub struct ConfidentialImageGenerationsOpenApi;

/// Handles confidential image generation requests
///
/// This handler processes image generation requests with confidential computing requirements,
/// tracking metrics and managing the encryption of responses. It follows the same core flow
/// as the standard image generations handler but ensures the response is encrypted according
/// to the client's confidential computing requirements.
///
/// # Arguments
///
/// * `request_metadata` - Extension containing request context including encryption metadata
/// * `state` - Application state containing service URLs and shared resources
/// * `payload` - The image generation request body as JSON
///
/// # Returns
///
/// Returns a `Result` containing either:
/// * `Ok(Json<Value>)` - The encrypted response from the image service
/// * `Err(AtomaServiceError)` - An error if the request processing fails
///
/// # Metrics
///
/// * Increments `IMAGE_GEN_NUM_REQUESTS` counter with model label
/// * Records request duration in `IMAGE_GEN_LATENCY_METRICS` histogram
///
/// # Errors
///
/// Returns `AtomaServiceError::InternalError` if:
/// * Image generation request fails
/// * Response encryption fails
/// * Stack compute units update fails
#[utoipa::path(
    post,
    path = "",
    tag = "confidential-images",
    request_body = ConfidentialComputeRequest,
    responses(
        (status = OK, description = "Confidential images generated successfully", body = ConfidentialComputeResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
#[instrument(
    level = "info",
    skip(state, payload),
    fields(path = request_metadata.endpoint_path)
)]
pub async fn confidential_image_generations_handler(
    Extension(request_metadata): Extension<RequestMetadata>,
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, AtomaServiceError> {
    info!("Received image generations request, with payload: {payload}");
    let model = payload
        .get(MODEL_KEY)
        .and_then(|m| m.as_str())
        .unwrap_or("unknown")
        .to_string();

    IMAGE_GEN_CONFIDENTIAL_NUM_REQUESTS
        .add(1, &[KeyValue::new("model", model.as_str().to_owned())]);
    let timer = Instant::now();

    let RequestMetadata {
        stack_small_id,
        estimated_total_compute_units,
        payload_hash,
        client_encryption_metadata,
        endpoint_path: endpoint,
        request_type: _,
    } = request_metadata;

    match handle_image_generations_response(
        &state,
        payload.clone(),
        payload_hash,
        stack_small_id,
        estimated_total_compute_units,
        client_encryption_metadata,
        &endpoint,
        timer,
        model.to_string(),
    )
    .await
    {
        Ok(response) => {
            TOTAL_COMPLETED_REQUESTS.add(1, &[KeyValue::new("model", model.clone())]);
            Ok(response)
        }
        Err(e) => {
            TOTAL_FAILED_REQUESTS.add(1, &[KeyValue::new("model", model.clone())]);
            TOTAL_FAILED_IMAGE_CONFIDENTIAL_GENERATION_REQUESTS
                .add(1, &[KeyValue::new("model", model.clone())]);
            let concurrent_requests = handle_concurrent_requests_count_updates(
                &state.concurrent_requests_per_stack,
                stack_small_id,
                "image-generations/confidential_image_generations_handler",
            );
            update_stack_num_compute_units(
                &state.state_manager_sender,
                stack_small_id,
                estimated_total_compute_units,
                0,
                &endpoint,
                concurrent_requests,
            )?;
            Err(AtomaServiceError::InternalError {
                message: format!("Error handling image generations response: {}", e),
                endpoint: endpoint.to_string(),
            })
        }
    }
}

/// Handles the core logic for processing image generation requests and responses
///
/// This function performs several key operations:
/// 1. Forwards the image generation request to the image generations service
/// 2. Signs the response and updates the stack hash
/// 3. Handles confidential compute encryption if needed
/// 4. Records timing metrics for the operation
///
/// # Arguments
///
/// * `state` - Application state containing service URLs and other shared resources
/// * `payload` - The JSON payload containing the image generation request parameters
/// * `payload_hash` - A 32-byte hash of the original request payload
/// * `stack_small_id` - Identifier for the current stack
/// * `estimated_total_compute_units` - Expected computational cost of the operation
/// * `client_encryption_metadata` - Optional encryption metadata for confidential compute
/// * `endpoint` - The API endpoint path being accessed
/// * `timer` - Prometheus histogram timer for measuring request duration
///
/// # Returns
///
/// Returns a `Result` containing either:
/// * `Ok(Json<Value>)` - The processed and possibly encrypted response from the image service
/// * `Err(AtomaServiceError)` - An error if any step in the process fails
///
/// # Errors
///
/// This function can return `AtomaServiceError::InternalError` in several cases:
/// * Failed to send request to the image generations service
/// * Failed to parse the service response
/// * Failed to sign the response or update the stack hash
/// * Failed to handle confidential compute encryption
#[instrument(
    level = "info",
    skip(state, payload),
    fields(path = endpoint)
)]
#[allow(clippy::too_many_arguments)]
async fn handle_image_generations_response(
    state: &AppState,
    payload: Value,
    payload_hash: [u8; 32],
    stack_small_id: i64,
    estimated_total_compute_units: i64,
    client_encryption_metadata: Option<EncryptionMetadata>,
    endpoint: &str,
    timer: Instant,
    model: String,
) -> Result<Json<Value>, AtomaServiceError> {
    let client = Client::new();
    let response = client
        .post(format!(
            "{}{}",
            state.image_generations_service_url, IMAGE_GENERATIONS_PATH
        ))
        .json(&payload)
        .send()
        .await
        .map_err(|e| AtomaServiceError::InternalError {
            message: format!("Error sending request to image generations service: {}", e),
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
            IMAGE_GEN_LATENCY_METRICS.record(
                timer.elapsed().as_secs_f64(),
                &[KeyValue::new("model", model.clone())],
            );
            Ok(Json(response_body))
        }
        Err(e) => {
            Err(AtomaServiceError::InternalError {
                message: format!(
                    "Error handling confidential compute encryption response, for request with payload hash: {:?}, and stack small id: {}, with error: {}",
                    payload_hash,
                    stack_small_id,
                    e
                ),
                endpoint: endpoint.to_string(),
            })
        }
    }
}

/// A model representing the parameters for an image generation request.
///
/// This struct encapsulates the required parameters for generating images through
/// the API endpoint.
pub struct RequestModelImageGenerations {
    /// The number of sampling generation to be performed for this request
    n: u64,
    /// The desired dimensions of the generated images in the format "WIDTHxHEIGHT"
    /// (e.g., "1024x1024")
    size: String,
}

impl RequestModel for RequestModelImageGenerations {
    fn new(request: &Value) -> Result<Self, AtomaServiceError> {
        let n = request
            .get(N_KEY)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| AtomaServiceError::InvalidBody {
                message: "N field is required".to_string(),
                endpoint: IMAGE_GENERATIONS_PATH.to_string(),
            })?;
        let size = request
            .get(SIZE_KEY)
            .and_then(|s| s.as_str())
            .ok_or_else(|| AtomaServiceError::InvalidBody {
                message: "Size field is required".to_string(),
                endpoint: IMAGE_GENERATIONS_PATH.to_string(),
            })?;

        Ok(Self {
            n,
            size: size.to_string(),
        })
    }

    fn get_compute_units_estimate(
        &self,
        _tokenizer: Option<&Tokenizer>,
    ) -> Result<u64, AtomaServiceError> {
        // Parse dimensions from size string (e.g., "1024x1024")
        let dimensions: Vec<u64> = self
            .size
            .split('x')
            .filter_map(|s| s.parse().ok())
            .collect();

        if dimensions.len() != 2 {
            return Err(AtomaServiceError::InvalidBody {
                message: format!(
                    "Invalid size format, expected two dimensional image, but got: {}",
                    self.size
                ),
                endpoint: IMAGE_GENERATIONS_PATH.to_string(),
            });
        }

        let width = dimensions[0];
        let height = dimensions[1];

        // Calculate compute units based on number of images and pixel count
        Ok(self.n * width * height)
    }
}
