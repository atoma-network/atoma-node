use crate::{
    handlers::{
        handle_concurrent_requests_count_decrement,
        metrics::{
            CHAT_COMPLETIONS_CONFIDENTIAL_NUM_REQUESTS, CHAT_COMPLETIONS_ESTIMATED_TOTAL_TOKENS,
            TOTAL_FAILED_CHAT_CONFIDENTIAL_REQUESTS, TOTAL_FAILED_CHAT_REQUESTS,
        },
        sign_response_and_update_stack_hash, update_fiat_amount, update_stack_num_compute_units,
    },
    middleware::EncryptionMetadata,
    server::AppState,
    streamer::{Streamer, StreamingEncryptionMetadata},
    types::{ConfidentialComputeRequest, ConfidentialComputeResponse},
};
use atoma_confidential::types::{
    ConfidentialComputeSharedSecretRequest, ConfidentialComputeSharedSecretResponse,
};
use atoma_utils::constants::{PAYLOAD_HASH_SIZE, REQUEST_ID};
use axum::{
    body::Body,
    extract::State,
    response::{IntoResponse, Response, Sse},
    Extension, Json,
};
use hyper::{HeaderMap, StatusCode};
use openai_api_completions::{
    CompletionChoice, CompletionTokensDetails, CompletionsPrompt, CompletionsRequest,
    CompletionsResponse, LogProbs, PromptTokensDetails, Usage,
};
use opentelemetry::KeyValue;
use reqwest::Client;
use serde_json::{json, Value};
use tokenizers::Tokenizer;
use tracing::{debug, info, instrument};
use utoipa::OpenApi;

use serde::Deserialize;
use std::time::{Duration, Instant};

use crate::{
    error::AtomaServiceError,
    handlers::metrics::{
        CHAT_COMPLETIONS_INPUT_TOKENS_METRICS, CHAT_COMPLETIONS_NUM_REQUESTS,
        CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS, TOTAL_COMPLETED_REQUESTS, TOTAL_FAILED_REQUESTS,
    },
    middleware::RequestMetadata,
};

use super::{
    handle_confidential_compute_encryption_response, handle_status_code_error,
    request_model::{ComputeUnitsEstimate, RequestModel},
    vllm_metrics::get_best_available_chat_completions_service_url,
    DEFAULT_MAX_TOKENS,
};

/// The path for confidential chat completions requests
pub const CONFIDENTIAL_COMPLETIONS_PATH: &str = "/v1/confidential/completions";

/// The key for the content parameter in the request body
pub const CONTENT_KEY: &str = "content";

/// The path for chat completions requests
pub const COMPLETIONS_PATH: &str = "/v1/completions";

/// The keep-alive interval in seconds
const STREAM_KEEP_ALIVE_INTERVAL_IN_SECONDS: u64 = 15;

/// The key for the max_completion_tokens parameter in the request body
const MAX_COMPLETION_TOKENS_KEY: &str = "max_completion_tokens";

/// The key for the max_tokens parameter in the request body
const MAX_TOKENS_KEY: &str = "max_tokens";

/// The key for the model parameter in the request body
const MODEL_KEY: &str = "model";

/// The key for the prompt parameter in the request body
const PROMPT_KEY: &str = "prompt";

/// The key for the stream parameter in the request body
const STREAM_KEY: &str = "stream";

/// The default model to use if the model is not found in the request body
const UNKNOWN_MODEL: &str = "unknown";

/// OpenAPI documentation structure for the chat completions endpoint.
///
/// This struct defines the OpenAPI (Swagger) documentation for the chat completions API,
/// including all request and response schemas. It uses the `utoipa` framework to generate
/// the API documentation.
///
/// # Components
///
/// The documentation includes the following schema components:
/// - `ChatCompletionsRequest`: The request body for chat completion requests
/// - `Message`: A message in the conversation (system, user, assistant, or tool)
/// - `MessageContent`: Content of a message (text or array of content parts)
/// - `MessageContentPart`: Individual content parts (text or image)
/// - `MessageContentPartImageUrl`: Image URL configuration
/// - `ToolCall`: Information about a tool call made by the model
/// - `ToolCallFunction`: Function call details within a tool call
/// - `Tool`: Available tools that the model can use
/// - `ToolFunction`: Function definition within a tool
/// - `StopCondition`: Conditions for stopping token generation
/// - `FinishReason`: Reasons why the model stopped generating
/// - `Usage`: Token usage statistics
/// - `Choice`: A single completion choice
/// - `ChatCompletionsResponse`: The complete response structure
/// - `ChatCompletionRequest`: The request body for chat completion requests
/// - `ChatCompletionMessage`: A message in the conversation (system, user, assistant, or tool)
/// - `ChatCompletionResponse`: The complete response structure
/// - `ChatCompletionChoice`: A single completion choice
/// - `CompletionUsage`: Token usage statistics
/// - `PromptTokensDetails`: Details about the prompt tokens
///
/// # Paths
///
/// Documents the following endpoint:
/// - `chat_completions_handler`: POST endpoint for chat completions
#[derive(OpenApi)]
#[openapi(
    paths(completions_handler),
    components(schemas(
        CompletionsRequest,
        CompletionsResponse,
        CompletionChoice,
        LogProbs,
        Usage,
        CompletionTokensDetails,
        PromptTokensDetails,
    ))
)]
pub struct CompletionsOpenApi;

/// Create chat completion
///
/// This handler performs several key operations:
/// 1. Forwards the chat completion request to the inference service
/// 2. Signs the response using the node's keystore
/// 3. Tracks token usage for the stack
///
/// # Arguments
///
/// * `Extension((stack_small_id, estimated_total_compute_units))` - Stack ID and estimated compute units count from middleware
/// * `state` - Application state containing the inference client and keystore
/// * `payload` - The chat completion request body
///
/// # Returns
///
/// Returns a JSON response containing:
/// - The inference service's response
/// - A cryptographic signature of the response
///
/// # Errors
///
/// Returns a `AtomaServiceError::InternalError` if:
/// - The inference service request fails
/// - Response parsing fails
/// - Response signing fails
/// - Token usage update fails
#[utoipa::path(
    post,
    path = "",
    tag = "chat",
    request_body = CompletionsRequest,
    responses(
        (status = OK, description = "Chat completion successful", body = CompletionsResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
#[instrument(
    level = "info",
    skip_all,
    fields(path = request_metadata.endpoint_path),
    err
)]
pub async fn completions_handler(
    Extension(request_metadata): Extension<RequestMetadata>,
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Result<Response<Body>, AtomaServiceError> {
    let RequestMetadata {
        stack_small_id,
        estimated_total_compute_units,
        num_input_tokens,
        payload_hash,
        client_encryption_metadata,
        user_address,
        price_per_one_million_compute_units,
        ..
    } = request_metadata;
    info!(
        target = "atoma-service",
        level = "info",
        event = "chat-completions-handler",
        "Received chat completions request, with payload hash: {payload_hash:?}"
    );

    let is_stream = payload
        .get(STREAM_KEY)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or_default();
    let endpoint = request_metadata.endpoint_path.clone();

    let model = payload
        .get(MODEL_KEY)
        .and_then(|m| m.as_str())
        .unwrap_or_default();

    match handle_response(
        &state,
        endpoint.clone(),
        payload_hash,
        stack_small_id,
        price_per_one_million_compute_units,
        user_address.clone(),
        is_stream,
        payload.clone(),
        num_input_tokens,
        estimated_total_compute_units,
        client_encryption_metadata,
        headers,
    )
    .await
    {
        Ok(response) => {
            CHAT_COMPLETIONS_ESTIMATED_TOTAL_TOKENS.add(
                estimated_total_compute_units,
                &[KeyValue::new(MODEL_KEY, model.to_owned())],
            );
            if !is_stream {
                TOTAL_COMPLETED_REQUESTS.add(1, &[KeyValue::new(MODEL_KEY, model.to_owned())]);
            }
            Ok(response)
        }
        Err(e) => {
            TOTAL_FAILED_CHAT_REQUESTS.add(1, &[KeyValue::new(MODEL_KEY, model.to_owned())]);
            TOTAL_FAILED_REQUESTS.add(1, &[KeyValue::new(MODEL_KEY, model.to_owned())]);
            // NOTE: We need to update the stack number of tokens as the service failed to generate
            // a proper response. For this reason, we set the total number of tokens to 0.
            // This will ensure that the stack number of tokens is not updated, and the stack
            // will not be penalized for the request.
            //
            // NOTE: We also decrement the concurrent requests count, as we are done processing the request.
            if let Some(stack_small_id) = stack_small_id {
                let concurrent_requests = handle_concurrent_requests_count_decrement(
                    &state.concurrent_requests_per_stack,
                    stack_small_id,
                    "chat-completions/chat_completions_handler",
                );
                update_stack_num_compute_units(
                    &state.state_manager_sender,
                    stack_small_id,
                    estimated_total_compute_units,
                    0,
                    &endpoint,
                    concurrent_requests,
                )?;
            } else {
                update_fiat_amount(
                    &state.state_manager_sender,
                    user_address,
                    estimated_total_compute_units,
                    0,
                    price_per_one_million_compute_units,
                    &endpoint,
                )?;
            }
            match e {
                // We want to propagate the error if the inference service is unavailable
                AtomaServiceError::ChatCompletionsServiceUnavailable { .. } => Err(e),
                _ => Err(AtomaServiceError::InternalError {
                    message: format!("Error handling chat completions response: {}", e),
                    endpoint: request_metadata.endpoint_path.clone(),
                }),
            }
        }
    }
}

/// OpenAPI documentation structure for the confidential chat completions endpoint.
///
/// This struct defines the OpenAPI (Swagger) documentation for the confidential chat completions API,
/// which provides an encrypted variant of the standard chat completions endpoint. It uses the
/// `utoipa` framework to generate the API documentation.
///
/// # Components
///
/// The documentation includes the following schema components:
/// - `ChatCompletionsRequest`: The request body for chat completion requests
/// - `ConfidentialComputeResponse`: The encrypted response structure for confidential compute
///
/// # Paths
///
/// Documents the following endpoint:
/// - `chat_completions_handler`: POST endpoint for confidential chat completions
///
/// # Security
///
/// The confidential variant ensures end-to-end encryption of the chat completion responses,
/// making it suitable for sensitive or private conversations.
#[derive(OpenApi)]
#[openapi(
    paths(confidential_completions_handler),
    components(schemas(ConfidentialComputeRequest, ConfidentialComputeResponse))
)]
pub struct ConfidentialCompletionsOpenApi;

/// Handles confidential chat completion requests by providing end-to-end encrypted responses.
///
/// This handler processes chat completion requests in a confidential computing context, where
/// responses are encrypted to ensure privacy. It supports both streaming and non-streaming
/// responses, with encryption handled appropriately for each mode.
///
/// # Flow
/// 1. Extracts metadata from the request (stack ID, compute units, encryption data)
/// 2. Checks if streaming mode is requested
/// 3. Forwards request to the inference service with appropriate encryption
/// 4. Handles response encryption and error cases
/// 5. Updates stack compute unit tracking
///
/// # Arguments
/// * `request_metadata` - Extension containing request context including:
///   - `stack_small_id` - Unique identifier for the requesting stack
///   - `estimated_total_compute_units` - Predicted compute cost
///   - `payload_hash` - Hash of the request payload
///   - `client_encryption_metadata` - Encryption parameters for confidential compute
/// * `state` - Application state containing service connections and configuration
/// * `payload` - The chat completion request body as JSON
///
/// # Returns
/// Returns a `Result` containing either:
/// - `Ok(Response)` - The encrypted chat completion response
/// - `Err(AtomaServiceError)` - Error details if the request failed
///
/// # Errors
/// Returns `AtomaServiceError::InternalError` if:
/// - The inference service request fails
/// - Response encryption fails
/// - State manager updates fail
///
/// # Example Request
/// ```json
/// POST /v1/confidential/chat/completions
/// {
///   "model": "gpt-4",
///   "messages": [
///     {"role": "user", "content": "Hello"}
///   ],
///   "stream": false
/// }
/// ```
///
/// # Notes
/// - On error, updates stack compute units to 0 to avoid penalizing failed requests
/// - Supports both streaming and non-streaming responses
/// - Automatically handles encryption/decryption of messages
/// - Integrates with monitoring via structured logging
#[utoipa::path(
    post,
    path = "",
    tag = "confidential-chat",
    request_body = ConfidentialComputeRequest,
    responses(
        (status = OK, description = "Confidential chat completion successful", body = ConfidentialComputeResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
#[instrument(
    level = "info",
    skip_all,
    fields(path = request_metadata.endpoint_path),
    err
)]
pub async fn confidential_completions_handler(
    Extension(request_metadata): Extension<RequestMetadata>,
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Result<Response<Body>, AtomaServiceError> {
    let RequestMetadata {
        stack_small_id,
        num_input_tokens,
        estimated_total_compute_units,
        payload_hash,
        client_encryption_metadata,
        user_address,
        price_per_one_million_compute_units,
        ..
    } = request_metadata;
    info!(
        target = "atoma-service",
        level = "info",
        event = "chat-completions-handler",
        "Received chat completions request, with payload hash: {payload_hash:?}"
    );

    // Check if streaming is requested
    let is_stream = payload
        .get(STREAM_KEY)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or_default();

    let model = payload
        .get(MODEL_KEY)
        .and_then(|m| m.as_str())
        .unwrap_or(UNKNOWN_MODEL);

    CHAT_COMPLETIONS_CONFIDENTIAL_NUM_REQUESTS
        .add(1, &[KeyValue::new(MODEL_KEY, model.to_owned())]);

    let endpoint = request_metadata.endpoint_path.clone();

    match handle_response(
        &state,
        endpoint.clone(),
        payload_hash,
        stack_small_id,
        price_per_one_million_compute_units,
        user_address.clone(),
        is_stream,
        payload.clone(),
        num_input_tokens,
        estimated_total_compute_units,
        client_encryption_metadata,
        headers,
    )
    .await
    {
        Ok(response) => {
            CHAT_COMPLETIONS_ESTIMATED_TOTAL_TOKENS.add(
                estimated_total_compute_units,
                &[KeyValue::new(MODEL_KEY, model.to_owned())],
            );
            if !is_stream {
                TOTAL_COMPLETED_REQUESTS.add(1, &[KeyValue::new(MODEL_KEY, model.to_owned())]);
            }
            Ok(response)
        }
        Err(e) => {
            TOTAL_FAILED_CHAT_CONFIDENTIAL_REQUESTS
                .add(1, &[KeyValue::new(MODEL_KEY, model.to_owned())]);
            TOTAL_FAILED_REQUESTS.add(1, &[KeyValue::new(MODEL_KEY, model.to_owned())]);
            if let Some(stack_small_id) = stack_small_id {
                // NOTE: We need to update the stack number of tokens as the service failed to generate
                // a proper response. For this reason, we set the total number of tokens to 0.
                // This will ensure that the stack number of tokens is not updated, and the stack
                // will not be penalized for the request.
                //
                // NOTE: We also decrement the concurrent requests count, as we are done processing the request.

                let concurrent_requests = handle_concurrent_requests_count_decrement(
                    &state.concurrent_requests_per_stack,
                    stack_small_id,
                    "chat-completions/confidential_chat_completions_handler",
                );
                update_stack_num_compute_units(
                    &state.state_manager_sender,
                    stack_small_id,
                    estimated_total_compute_units,
                    0,
                    &endpoint,
                    concurrent_requests,
                )?;
            } else {
                update_fiat_amount(
                    &state.state_manager_sender,
                    user_address,
                    estimated_total_compute_units,
                    0,
                    price_per_one_million_compute_units,
                    &endpoint,
                )?;
            }
            return Err(AtomaServiceError::InternalError {
                message: format!("Error handling chat completions response: {}", e),
                endpoint: request_metadata.endpoint_path.clone(),
            });
        }
    }
}

/// Handles both streaming and non-streaming chat completion requests by routing them to appropriate handlers.
///
/// This function serves as a router that determines whether to process the request as a streaming
/// or non-streaming chat completion based on the `is_stream` parameter. For streaming requests,
/// it also handles the setup of encryption metadata when confidential compute is enabled.
///
/// # Arguments
///
/// * `state` - Application state containing service configuration and connections
/// * `endpoint` - The API endpoint path where the request was received
/// * `payload_hash` - BLAKE2b hash of the original request payload
/// * `stack_small_id` - Unique identifier for the stack making the request
/// * `is_stream` - Boolean flag indicating whether this is a streaming request
/// * `payload` - The JSON payload containing the chat completion request
/// * `estimated_total_compute_units` - Estimated compute units for the request
/// * `client_encryption_metadata` - Optional encryption metadata for confidential compute
///
/// # Returns
///
/// Returns a `Result` containing either:
/// - For non-streaming: A JSON response with the complete chat completion
/// - For streaming: An SSE stream that will emit completion chunks
///
/// # Errors
///
/// Returns `AtomaServiceError::InternalError` if:
/// - Encryption metadata setup fails for streaming requests
/// - Either the streaming or non-streaming handler encounters an error
///
/// # Example
///
/// ```rust,ignore
/// let response = handle_response(
///     state,
///     "/v1/chat/completions".to_string(),
///     payload_hash,
///     stack_id,
///     false, // non-streaming
///     payload,
///     estimated_units,
///     None,
/// ).await?;
/// ```
#[instrument(
    level = "info",
    skip_all,
    fields(
        path = COMPLETIONS_PATH,
        stack_small_id,
        estimated_total_compute_units,
        payload_hash
    ),
    err
)]
#[allow(clippy::too_many_arguments)]
async fn handle_response(
    state: &AppState,
    endpoint: String,
    payload_hash: [u8; PAYLOAD_HASH_SIZE],
    stack_small_id: Option<i64>,
    price_per_one_million_compute_units: i64,
    user_address: String,
    is_stream: bool,
    payload: Value,
    num_input_tokens: i64,
    estimated_total_compute_units: i64,
    client_encryption_metadata: Option<EncryptionMetadata>,
    headers: HeaderMap,
) -> Result<Response<Body>, AtomaServiceError> {
    if is_stream {
        let streaming_encryption_metadata = utils::get_streaming_encryption_metadata(
            state,
            client_encryption_metadata,
            payload_hash,
            stack_small_id,
            &endpoint,
        )
        .await?;

        handle_streaming_response(
            state,
            payload,
            stack_small_id,
            num_input_tokens,
            estimated_total_compute_units,
            price_per_one_million_compute_units,
            user_address,
            payload_hash,
            streaming_encryption_metadata,
            endpoint,
            headers,
        )
        .await
    } else {
        handle_non_streaming_response(
            state,
            payload,
            stack_small_id,
            estimated_total_compute_units,
            price_per_one_million_compute_units,
            user_address,
            payload_hash,
            client_encryption_metadata,
            endpoint,
        )
        .await
    }
}

/// Handles non-streaming chat completion requests by processing them through the inference service.
///
/// This function performs several key operations in the following order:
/// 1. Forwards the request to the inference service
/// 2. Processes and signs the response
/// 3. Updates token usage tracking
/// 4. Handles confidential compute encryption (if enabled)
/// 5. Updates the stack's compute units count (final step)
///
/// The update of compute units is intentionally performed as the last operation to ensure
/// database consistency. If any earlier steps fail (e.g., encryption errors), we avoid
/// updating the compute units count prematurely.
///
/// # Arguments
///
/// * `state` - Application state containing service configuration and keystore
/// * `payload` - The JSON payload containing the chat completion request
/// * `stack_small_id` - Unique identifier for the stack making the request
/// * `estimated_total_compute_units` - Estimated compute units count for the request
/// * `payload_hash` - BLAKE2b hash of the original request payload
/// * `client_encryption_metadata` - The client encryption metadata for the request
/// * `endpoint` - The endpoint where the request was made
///
/// # Returns
///
/// Returns a `Result` containing the JSON response with added signature, or a `AtomaServiceError`.
///
/// # Errors
///
/// Returns `AtomaServiceError::InternalError` if:
/// - The inference service request fails
/// - Response parsing fails
/// - Response signing fails
/// - Confidential compute encryption fails
/// - State manager updates fail
///
/// # Example Response Structure
///
/// ```json
/// {
///     "choices": [...],
///     "usage": {
///         "total_tokens": 123,
///         "prompt_tokens": 45,
///         "completion_tokens": 78
///     },
///     "signature": "base64_encoded_signature"
/// }
/// ```
#[instrument(
    level = "info",
    skip_all,
    fields(
        path = COMPLETIONS_PATH,
        completion_type = "non-streaming",
        stack_small_id,
        estimated_total_compute_units,
        payload_hash
    ),
    err
)]
#[allow(clippy::too_many_arguments)]
async fn handle_non_streaming_response(
    state: &AppState,
    payload: Value,
    stack_small_id: Option<i64>,
    estimated_total_compute_units: i64,
    price_per_one_million_compute_units: i64,
    user_address: String,
    payload_hash: [u8; PAYLOAD_HASH_SIZE],
    client_encryption_metadata: Option<EncryptionMetadata>,
    endpoint: String,
) -> Result<Response<Body>, AtomaServiceError> {
    // Record token metrics and extract the response total number of tokens
    let model = payload
        .get(MODEL_KEY)
        .and_then(|m| m.as_str())
        .unwrap_or(UNKNOWN_MODEL);

    CHAT_COMPLETIONS_NUM_REQUESTS.add(1, &[KeyValue::new(MODEL_KEY, model.to_owned())]);
    let timer = Instant::now();
    debug!(
        target = "atoma-service",
        level = "debug",
        "Sending non-streaming chat completions request to {endpoint}"
    );
    let response_body = utils::send_request_to_inference_service(
        state,
        &payload,
        stack_small_id,
        payload_hash,
        &endpoint,
    )
    .await?;
    debug!(
        target = "atoma-service",
        level = "debug",
        "Received non-streaming chat completions response from {endpoint}"
    );
    let total_compute_units = utils::extract_total_num_tokens(&response_body, model);

    utils::serve_non_streaming_response(
        state,
        response_body,
        stack_small_id,
        estimated_total_compute_units,
        price_per_one_million_compute_units,
        user_address,
        total_compute_units,
        payload_hash,
        client_encryption_metadata,
        endpoint,
        timer,
        model,
    )
    .await
}

/// Handles streaming chat completion requests by establishing a Server-Sent Events (SSE) connection.
///
/// This function processes streaming chat completion requests by:
/// 1. Adding required streaming options to the payload
/// 2. Forwarding the request to the inference service
/// 3. Establishing an SSE connection with keep-alive functionality
/// 4. Setting up a Streamer to handle the response chunks and manage token usage
///
/// # Arguments
///
/// * `state` - Application state containing service configuration and connections
/// * `payload` - The JSON payload containing the chat completion request
/// * `stack_small_id` - Unique identifier for the stack making the request
/// * `estimated_total_compute_units` - Estimated compute units count for the request
/// * `payload_hash` - BLAKE2b hash of the original request payload
/// * `streaming_encryption_metadata` - The client encryption metadata for the streaming request
/// * `endpoint` - The endpoint where the request was made
///
/// # Returns
///
/// Returns a `Result` containing an SSE stream response, or a `AtomaServiceError`.
///
/// # Errors
///
/// Returns `AtomaServiceError::InternalError` if:
/// - The inference service request fails
/// - The inference service returns a non-success status code
///
/// # Example Response Stream
///
/// The SSE stream will emit events in the following format:
/// ```text
/// data: {"choices": [...], "usage": null}
/// data: {"choices": [...], "usage": null}
/// data: {"choices": [...], "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
/// ```
#[instrument(
    level = "info",
    skip_all,
    fields(
        path = COMPLETIONS_PATH,
        completion_type = "streaming",
        stack_small_id,
        estimated_total_compute_units,
        payload_hash
    ),
    err
)]
#[allow(clippy::too_many_arguments)]
async fn handle_streaming_response(
    state: &AppState,
    mut payload: Value,
    stack_small_id: Option<i64>,
    num_input_tokens: i64,
    estimated_total_compute_units: i64,
    price_per_one_million_compute_units: i64,
    user_address: String,
    payload_hash: [u8; 32],
    streaming_encryption_metadata: Option<StreamingEncryptionMetadata>,
    endpoint: String,
    headers: HeaderMap,
) -> Result<Response<Body>, AtomaServiceError> {
    // NOTE: If streaming is requested, add the include_usage option to the payload
    // so that the atoma node state manager can be updated with the total number of tokens
    // that were processed for this request.
    payload["stream_options"] = json!({
        "include_usage": true
    });

    let request_id = headers
        .get(REQUEST_ID)
        .ok_or_else(|| AtomaServiceError::MissingHeader {
            header: REQUEST_ID.to_string(),
            endpoint: endpoint.clone(),
        })?
        .to_str()
        .map_err(|_| AtomaServiceError::InvalidHeader {
            message: "Request ID header is invalid, cannot be converted to string".to_string(),
            endpoint: endpoint.clone(),
        })?
        .to_string();

    let model = payload
        .get(MODEL_KEY)
        .and_then(|m| m.as_str())
        .unwrap_or(UNKNOWN_MODEL);
    CHAT_COMPLETIONS_NUM_REQUESTS.add(1, &[KeyValue::new(MODEL_KEY, model.to_owned())]);
    let timer = Instant::now();

    let chat_completions_service_urls = state
        .chat_completions_service_urls
        .get(&model.to_lowercase())
        .ok_or_else(|| {
            AtomaServiceError::InternalError {
                message: format!(
                    "Chat completions service URL not found, likely that model is not supported by the current node: {}",
                    model
                ),
                endpoint: endpoint.clone(),
            }
        })?;
    let (chat_completions_service_url, status_code) =
        get_best_available_chat_completions_service_url(chat_completions_service_urls, model)
            .await
            .map_err(|e| AtomaServiceError::ChatCompletionsServiceUnavailable {
                message: e.to_string(),
                endpoint: endpoint.clone(),
            })?;
    if status_code == StatusCode::TOO_MANY_REQUESTS {
        return Err(AtomaServiceError::ChatCompletionsServiceUnavailable {
            message: "Too many requests".to_string(),
            endpoint: endpoint.clone(),
        });
    }
    let client = Client::new();
    let response = client
        .post(format!(
            "{}{}",
            chat_completions_service_url, COMPLETIONS_PATH
        ))
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            AtomaServiceError::InternalError {
                message: format!(
                    "Error sending request to inference service, for request with payload hash: {:?}, and stack small id: {:?}, with error: {}",
                    payload_hash,
                    stack_small_id,
                    e
                ),
                endpoint: endpoint.clone(),
            }
        })?;

    if !response.status().is_success() {
        let error = response
            .status()
            .canonical_reason()
            .unwrap_or("Unknown error");
        handle_status_code_error(response.status(), &endpoint, error)?;
    }

    let stream = response.bytes_stream();
    // Create the SSE stream
    let stream = Sse::new(Streamer::new(
        stream,
        state.state_manager_sender.clone(),
        state.concurrent_requests_per_stack.clone(),
        state.client_dropped_streamer_connections.clone(),
        stack_small_id,
        num_input_tokens,
        estimated_total_compute_units,
        payload_hash,
        state.keystore.clone(),
        state.address_index,
        model.to_string(),
        streaming_encryption_metadata,
        endpoint,
        request_id,
        timer,
        price_per_one_million_compute_units,
        user_address,
    ))
    .keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_millis(STREAM_KEEP_ALIVE_INTERVAL_IN_SECONDS))
            .text("keep-alive"),
    );

    Ok(stream.into_response())
}

/// Represents a chat completion request model following the OpenAI API format
pub struct RequestModelCompletions {
    /// The prompt to generate completions for
    prompt: CompletionsPrompt,

    /// The maximum number of tokens to generate in the completion
    /// This limits the length of the model's response
    max_completion_tokens: u64,
}

impl RequestModel for RequestModelCompletions {
    fn new(request: &Value) -> Result<Self, AtomaServiceError> {
        let prompt = request
            .get(PROMPT_KEY)
            .map(CompletionsPrompt::deserialize)
            .transpose()
            .map_err(|e| AtomaServiceError::InvalidBody {
                message: format!("Invalid 'prompt' field for `RequestModelCompletions`: {e:?}"),
                endpoint: COMPLETIONS_PATH.to_string(),
            })?
            .ok_or_else(|| AtomaServiceError::InvalidBody {
                message: "Missing or invalid 'prompt' field".to_string(),
                endpoint: COMPLETIONS_PATH.to_string(),
            })?;

        let max_completion_tokens = request
            .get(MAX_COMPLETION_TOKENS_KEY)
            .or_else(|| request.get(MAX_TOKENS_KEY))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(DEFAULT_MAX_TOKENS);

        Ok(Self {
            prompt,
            max_completion_tokens,
        })
    }

    /// Computes the total number of tokens for the completion request.
    ///
    /// This is used to estimate the cost of the completion request, on the proxy side.
    /// We support either string or array of content parts. We further assume that all content messages
    /// share the same previous messages. That said, we further assume that content parts formatted into arrays
    /// are to be concatenated and treated as a single message, by the model and from the estimate point of view.
    fn get_compute_units_estimate(
        &self,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<ComputeUnitsEstimate, AtomaServiceError> {
        // Helper function to count the number of tokens in a text prompt
        let count_text_tokens =
            |text: &str, tokenizer: &tokenizers::Tokenizer| -> Result<u64, AtomaServiceError> {
                Ok(tokenizer
                    .encode(text, true)
                    .map_err(|err| AtomaServiceError::InternalError {
                        message: format!("Failed to encode message: {err:?}"),
                        endpoint: COMPLETIONS_PATH.to_string(),
                    })?
                    .get_ids()
                    .len() as u64)
            };
        match &self.prompt {
            CompletionsPrompt::Single(prompt) => {
                let tokenizer = tokenizer.ok_or_else(|| AtomaServiceError::InternalError {
                    message: "Tokenizer is required for `RequestModelCompletions`".to_string(),
                    endpoint: COMPLETIONS_PATH.to_string(),
                })?;
                let num_input_compute_units =
                    count_text_tokens(prompt, tokenizer).map_err(|err| {
                        AtomaServiceError::InternalError {
                            message: format!("Failed to count text tokens: {err:?}"),
                            endpoint: COMPLETIONS_PATH.to_string(),
                        }
                    })?;
                Ok(ComputeUnitsEstimate {
                    num_input_compute_units,
                    max_total_compute_units: self.max_completion_tokens,
                })
            }
            CompletionsPrompt::List(prompts) => {
                let tokenizer = tokenizer.ok_or_else(|| AtomaServiceError::InternalError {
                    message: "Tokenizer is required for `RequestModelCompletions`".to_string(),
                    endpoint: COMPLETIONS_PATH.to_string(),
                })?;
                let num_input_compute_units = prompts
                    .iter()
                    .map(|prompt| count_text_tokens(prompt, tokenizer).unwrap_or(0))
                    .sum();
                Ok(ComputeUnitsEstimate {
                    num_input_compute_units,
                    max_total_compute_units: self.max_completion_tokens,
                })
            }
            CompletionsPrompt::Tokens(tokens) => {
                let num_input_compute_units = tokens.len() as u64;
                Ok(ComputeUnitsEstimate {
                    num_input_compute_units,
                    max_total_compute_units: self.max_completion_tokens,
                })
            }
            CompletionsPrompt::TokenArrays(token_arrays) => {
                let num_input_compute_units =
                    token_arrays.iter().map(|tokens| tokens.len() as u64).sum();
                Ok(ComputeUnitsEstimate {
                    num_input_compute_units,
                    max_total_compute_units: self.max_completion_tokens,
                })
            }
        }
    }
}

pub mod utils {
    use std::time::Instant;

    use atoma_utils::constants::PAYLOAD_HASH_SIZE;
    use hyper::StatusCode;
    use opentelemetry::KeyValue;

    use crate::handlers::{
        handle_concurrent_requests_count_decrement, handle_status_code_error,
        metrics::CHAT_COMPLETIONS_LATENCY_METRICS, update_fiat_amount,
        vllm_metrics::get_best_available_chat_completions_service_url, COMPLETION_TOKENS_KEY,
        PROMPT_TOKENS_KEY, USAGE_KEY,
    };

    use super::{
        handle_confidential_compute_encryption_response, info, instrument,
        sign_response_and_update_stack_hash, update_stack_num_compute_units, AppState,
        AtomaServiceError, Body, Client, ConfidentialComputeSharedSecretRequest,
        ConfidentialComputeSharedSecretResponse, EncryptionMetadata, IntoResponse, Json, Response,
        StreamingEncryptionMetadata, Value, CHAT_COMPLETIONS_INPUT_TOKENS_METRICS,
        CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS, COMPLETIONS_PATH, MODEL_KEY, UNKNOWN_MODEL,
    };

    /// Retrieves encryption metadata for streaming chat completions when confidential compute is enabled.
    ///
    /// This function handles the setup of encryption parameters for secure streaming communications by:
    /// 1. Checking if client encryption metadata is present
    /// 2. If present, computing a shared secret with the proxy using X25519
    /// 3. Combining the shared secret with a nonce and salt for secure streaming
    ///
    /// # Arguments
    ///
    /// * `state` - Application state containing service configuration and channels
    /// * `client_encryption_metadata` - Optional encryption metadata from the client containing the proxy's public key and salt
    /// * `payload_hash` - BLAKE2b hash of the original request payload
    /// * `stack_small_id` - Unique identifier for the stack making the request
    /// * `request_metadata` - Metadata about the incoming request including endpoint path
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing:
    /// - `Some(StreamingEncryptionMetadata)` if client encryption metadata was provided
    /// - `None` if no client encryption metadata was provided
    /// - `AtomaServiceError` if encryption setup fails
    ///
    /// # Errors
    ///
    /// Returns `AtomaServiceError::InternalError` if:
    /// - Failed to send encryption request through channel
    /// - Failed to receive shared secret response
    ///
    /// # Instrumentation
    ///
    /// This function is instrumented with debug-level tracing that includes:
    /// - payload_hash
    /// - stack_small_id
    /// - endpoint_path
    #[instrument(
        level = "debug",
        skip_all,
        fields(
            payload_hash,
            stack_small_id,
            endpoint_path = endpoint
        ),
        err
    )]
    pub async fn get_streaming_encryption_metadata(
        state: &AppState,
        client_encryption_metadata: Option<EncryptionMetadata>,
        payload_hash: [u8; PAYLOAD_HASH_SIZE],
        stack_small_id: Option<i64>,
        endpoint: &str,
    ) -> Result<Option<StreamingEncryptionMetadata>, AtomaServiceError> {
        let streaming_encryption_metadata = if let Some(client_encryption_metadata) =
            client_encryption_metadata
        {
            let (sender, receiver) = tokio::sync::oneshot::channel();
            state
                .compute_shared_secret_sender
                .send((
                    ConfidentialComputeSharedSecretRequest {
                        client_x25519_public_key: client_encryption_metadata.client_x25519_public_key,
                    },
                    sender,
                ))
                .map_err(|e| {
                    AtomaServiceError::InternalError {
                        message: format!(
                            "Error sending encryption request, for request with payload hash: {:?}, and stack small id: {:?}, with error: {}",
                            payload_hash,
                            stack_small_id,
                            e
                        ),
                        endpoint: endpoint.to_string(),
                    }
                })?;
            let ConfidentialComputeSharedSecretResponse {
                shared_secret,
                nonce,
            } = receiver.await.map_err(|e| {
                AtomaServiceError::InternalError {
                    message: format!(
                        "Error receiving encryption response, for request with payload hash: {:?}, and stack small id: {:?}, with error: {}",
                        payload_hash,
                        stack_small_id,
                        e
                    ),
                    endpoint: endpoint.to_string(),
                }
            })?;
            Some(StreamingEncryptionMetadata {
                shared_secret,
                nonce,
                salt: client_encryption_metadata.salt,
            })
        } else {
            None
        };
        Ok(streaming_encryption_metadata)
    }

    /// Sends a chat completion request to the inference service and parses the response.
    ///
    /// This function handles the HTTP communication with the inference service by:
    /// 1. Creating a new HTTP client
    /// 2. Sending the request with the provided payload
    /// 3. Parsing the JSON response
    ///
    /// # Arguments
    ///
    /// * `state` - Application state containing service configuration including the inference service URL
    /// * `payload` - The JSON payload containing the chat completion request parameters
    /// * `stack_small_id` - Unique identifier for the stack making the request
    /// * `payload_hash` - BLAKE2b hash of the original request payload
    /// * `endpoint` - The API endpoint path where the request was received
    ///
    /// # Returns
    ///
    /// Returns a `Result<Value, AtomaServiceError>` where:
    /// - `Value` is the parsed JSON response from the inference service
    /// - `AtomaServiceError` is returned if the request fails or response parsing fails
    ///
    /// # Errors
    ///
    /// Returns `AtomaServiceError::InternalError` if:
    /// - The HTTP request to the inference service fails
    /// - The response body cannot be parsed as valid JSON
    ///
    /// # Instrumentation
    ///
    /// This function is instrumented with info-level tracing that includes:
    /// - stack_small_id
    /// - payload_hash
    /// - endpoint
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let response = send_request_to_inference_service(
    ///     &state,
    ///     serde_json::json!({
    ///         "model": "gpt-4",
    ///         "messages": [{"role": "user", "content": "Hello"}]
    ///     }),
    ///     123,
    ///     [0u8; 32],
    ///     "/v1/chat/completions"
    /// ).await?;
    /// ```
    #[instrument(
        level = "info",
        skip_all,
        fields(stack_small_id, payload_hash, endpoint),
        err
    )]
    pub async fn send_request_to_inference_service(
        state: &AppState,
        payload: &Value,
        stack_small_id: Option<i64>,
        payload_hash: [u8; PAYLOAD_HASH_SIZE],
        endpoint: &str,
    ) -> Result<Value, AtomaServiceError> {
        let client = Client::new();
        let model = payload
            .get(MODEL_KEY)
            .and_then(|m| m.as_str())
            .unwrap_or(UNKNOWN_MODEL);
        let completions_service_url_services = state
            .chat_completions_service_urls
            .get(&model.to_lowercase())
            .ok_or_else(|| {
                AtomaServiceError::InternalError {
                    message: format!(
                        "Chat completions service URL not found, likely that model is not supported by the current node: {}",
                        model
                    ),
                    endpoint: endpoint.to_string(),
                }
            })?;
        let (completions_service_url, status_code) =
            get_best_available_chat_completions_service_url(
                completions_service_url_services,
                model,
            )
            .await
            .map_err(|e| AtomaServiceError::ChatCompletionsServiceUnavailable {
                message: e.to_string(),
                endpoint: endpoint.to_string(),
            })?;
        if status_code == StatusCode::TOO_MANY_REQUESTS {
            return Err(AtomaServiceError::ChatCompletionsServiceUnavailable {
                message: "Too many requests".to_string(),
                endpoint: endpoint.to_string(),
            });
        }
        let response = client
        .post(format!(
            "{}{}",
            completions_service_url, COMPLETIONS_PATH
        ))
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            AtomaServiceError::InternalError {
                message: format!(
                    "Error sending request to inference service, for request with payload hash: {:?}, and stack small id: {:?}, with error: {}",
                    payload_hash,
                    stack_small_id,
                    e
                ),
                endpoint: endpoint.to_string(),
            }
        })?;

        if !response.status().is_success() {
            let error = response
                .status()
                .canonical_reason()
                .unwrap_or("Unknown error");
            handle_status_code_error(response.status(), endpoint, error)?;
        }

        response.json::<Value>().await.map_err(|e| {
            AtomaServiceError::InternalError {
                message: format!(
                    "Error reading response body, for request with payload hash: {:?}, and stack small id: {:?}, with error: {}",
                    payload_hash,
                    stack_small_id,
                    e
                ),
                // NOTE: We don't know the number of tokens processed for this request,
                // as the returned output is invalid JSON. For this reason, we set it to 0.
                endpoint: endpoint.to_string(),
            }
        })
    }

    /// Extracts and tracks token usage metrics from a chat completion response.
    ///
    /// This function processes the "usage" field of a chat completion response to:
    /// 1. Extract prompt and completion token counts
    /// 2. Record token usage metrics for monitoring
    /// 3. Calculate total compute units used
    ///
    /// # Arguments
    ///
    /// * `response_body` - The JSON response body from the chat completion API containing usage statistics
    /// * `model` - The name of the model used for the completion (e.g., "gpt-4", "gpt-3.5-turbo")
    ///
    /// # Returns
    ///
    /// Returns the total number of compute units used (sum of prompt and completion tokens) as an i64
    ///
    /// # Metrics
    ///
    /// Records two Prometheus metrics:
    /// * `CHAT_COMPLETIONS_INPUT_TOKENS_METRICS`: Number of tokens in the prompt
    /// * `CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS`: Number of tokens in the completion
    ///
    /// # Example Response Structure
    ///
    /// The function expects a response body with this structure:
    /// ```json
    /// {
    ///     "usage": {
    ///         "prompt_tokens": 56,
    ///         "completion_tokens": 31,
    ///         "total_tokens": 87
    ///     }
    /// }
    /// ```
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let response_body = serde_json::json!({
    ///     "usage": {
    ///         "prompt_tokens": 10,
    ///         "completion_tokens": 20
    ///     }
    /// });
    /// let total = extract_total_num_tokens(&response_body, "gpt-4");
    /// assert_eq!(total, 30);
    /// ```
    pub fn extract_total_num_tokens(response_body: &Value, model: &str) -> i64 {
        let mut total_compute_units = 0;
        if let Some(usage) = response_body.get(USAGE_KEY) {
            if let Some(prompt_tokens) = usage.get(PROMPT_TOKENS_KEY) {
                let prompt_tokens = prompt_tokens.as_u64().unwrap_or(0);
                CHAT_COMPLETIONS_INPUT_TOKENS_METRICS
                    .add(prompt_tokens, &[KeyValue::new(MODEL_KEY, model.to_owned())]);
                total_compute_units += prompt_tokens;
            }
            if let Some(completion_tokens) = usage.get(COMPLETION_TOKENS_KEY) {
                let completion_tokens = completion_tokens.as_u64().unwrap_or(0);
                CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS.add(
                    completion_tokens,
                    &[KeyValue::new(MODEL_KEY, model.to_owned())],
                );
                total_compute_units += completion_tokens;
            }
        }
        total_compute_units as i64
    }

    /// Processes and serves a non-streaming chat completion response by handling signature verification,
    /// encryption, and compute unit tracking.
    ///
    /// This function performs several key operations in sequence:
    /// 1. Signs the response and updates the stack hash
    /// 2. Handles confidential compute encryption if enabled
    /// 3. Updates compute unit tracking in the state manager
    ///
    /// The function intentionally updates compute units as the final step to maintain database consistency
    /// in case of earlier failures.
    ///
    /// # Arguments
    ///
    /// * `state` - Application state containing service configuration and connections
    /// * `response_body` - The JSON response body from the inference service
    /// * `stack_small_id` - Unique identifier for the stack making the request
    /// * `estimated_total_compute_units` - Initially estimated compute units for the request
    /// * `total_compute_units` - Actual compute units used by the request
    /// * `payload_hash` - BLAKE2b hash of the original request payload
    /// * `client_encryption_metadata` - Optional encryption metadata for confidential compute
    /// * `endpoint` - The API endpoint path where the request was received
    /// * `timer` - Prometheus histogram timer for tracking response latency
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the HTTP response with the processed chat completion,
    /// or an error if any step fails.
    ///
    /// # Errors
    ///
    /// Returns `AtomaServiceError::InternalError` if:
    /// - Response signing or stack hash update fails
    /// - Confidential compute encryption fails
    /// - State manager update for compute units fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let response = serve_non_streaming_response(
    ///     &state,
    ///     response_body,
    ///     stack_id,
    ///     estimated_units,
    ///     actual_units,
    ///     payload_hash,
    ///     encryption_metadata,
    ///     "/v1/chat/completions".to_string(),
    ///     timer
    /// ).await?;
    /// ```
    ///
    /// # Instrumentation
    ///
    /// This function is instrumented with:
    /// - Info-level tracing with fields: stack_small_id, estimated_total_compute_units, payload_hash, endpoint
    /// - Prometheus metrics for response timing
    /// - Detailed logging of compute unit usage
    #[instrument(
        level = "info",
        skip_all,
        fields(stack_small_id, estimated_total_compute_units, payload_hash, endpoint),
        err
    )]
    #[allow(clippy::too_many_arguments)]
    pub async fn serve_non_streaming_response(
        state: &AppState,
        mut response_body: Value,
        stack_small_id: Option<i64>,
        estimated_total_compute_units: i64,
        price_per_one_million_compute_units: i64,
        user_address: String,
        total_compute_units: i64,
        payload_hash: [u8; PAYLOAD_HASH_SIZE],
        client_encryption_metadata: Option<EncryptionMetadata>,
        endpoint: String,
        timer: Instant,
        model: &str,
    ) -> Result<Response<Body>, AtomaServiceError> {
        info!(
            target = "atoma-service",
            level = "info",
            endpoint = "handle_non_streaming_response",
            stack_small_id = stack_small_id,
            estimated_total_compute_units = estimated_total_compute_units,
            payload_hash = hex::encode(payload_hash),
            "Total compute units: {}",
            total_compute_units,
        );

        if let Err(e) = sign_response_and_update_stack_hash(
            &mut response_body,
            payload_hash,
            state,
            stack_small_id,
            endpoint.clone(),
        )
        .await
        {
            return Err(AtomaServiceError::InternalError {
                message: format!(
                    "Error updating state manager, for request with payload hash: {:?}, and stack small id: {:?}, with error: {}",
                    payload_hash,
                    stack_small_id,
                    e
                ),
                endpoint: endpoint.clone(),
            });
        }

        // Handle confidential compute encryption response
        let response_body = match handle_confidential_compute_encryption_response(
            state,
            response_body,
            client_encryption_metadata.clone(),
            endpoint.clone(),
        )
        .await
        {
            Ok(response_body) => {
                // Stop the timer before returning the valid response
                CHAT_COMPLETIONS_LATENCY_METRICS.record(
                    timer.elapsed().as_secs_f64(),
                    &[KeyValue::new(MODEL_KEY, model.to_owned()), KeyValue::new("privacy_level", if client_encryption_metadata.is_some() { "confidential" } else { "non-confidential" })],
                );
                Json(response_body).into_response()
            }
            Err(e) => {
                return Err(AtomaServiceError::InternalError {
                    message: format!(
                        "Error handling confidential compute encryption response, for request with payload hash: {:?}, and stack small id: {:?}, with error: {}",
                        payload_hash,
                        stack_small_id,
                        e
                    ),
                    endpoint: endpoint.clone(),
                })
            }
        };

        if let Some(stack_small_id) = stack_small_id {
            // NOTE: We need to update the stack num tokens, because the inference response might have produced
            // less tokens than estimated what we initially estimated, from the middleware.
            //
            // NOTE: We update the total number of tokens as a final step, as if some error occurs, we don't want
            // to update the stack num tokens beforehand.
            //
            // NOTE: We also decrement the concurrent requests count, as we are done processing the request.
            info!(
                target = "atoma-service",
                level = "info",
                "Decrementing concurrent requests count for stack small id: {stack_small_id}"
            );
            let concurrent_requests = handle_concurrent_requests_count_decrement(
                &state.concurrent_requests_per_stack,
                stack_small_id,
                "chat-completions/serve_non_streaming_response",
            );
            info!(
            target = "atoma-service",
            level = "info",
            "Concurrent requests have been decremented. Updating stack num compute units for stack small id: {stack_small_id:?}"
        );
            update_stack_num_compute_units(
                &state.state_manager_sender,
                stack_small_id,
                estimated_total_compute_units,
                total_compute_units,
                &endpoint,
                concurrent_requests,
            )?;
        } else {
            update_fiat_amount(
                &state.state_manager_sender,
                user_address,
                estimated_total_compute_units,
                total_compute_units,
                price_per_one_million_compute_units,
                &endpoint,
            )?;
        }

        Ok(response_body)
    }
}

pub mod openai_api_completions {
    use std::collections::HashMap;

    use serde::{Deserialize, Serialize};
    use utoipa::ToSchema;

    use crate::handlers::chat_completions::openai_api::stream_options;

    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    pub struct CompletionsRequest {
        /// ID of the model to use
        #[schema(example = "meta-llama/Llama-3.3-70B-Instruct")]
        pub model: String,

        /// The prompt to generate completions for
        #[schema(example = json!(["Hello!"]))]
        pub prompt: CompletionsPrompt,

        #[schema(example = 1, default = 1)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub best_of: Option<i32>,

        #[schema(example = false, default = false)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub echo: Option<bool>,

        /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their
        /// existing frequency in the text so far
        #[schema(example = 0.0)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub frequency_penalty: Option<f32>,

        /// Modify the likelihood of specified tokens appearing in the completion.
        ///
        /// Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer)
        /// to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits
        /// generated by the model prior to sampling. The exact effect will vary per model, but values
        /// between -1 and 1 should decrease or increase likelihood of selection; values like -100 or
        /// 100 should result in a ban or exclusive selection of the relevant token.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[schema(example = json!({
            "1234567890": 0.5,
            "1234567891": -0.5
        }))]
        pub logit_bias: Option<std::collections::HashMap<String, f32>>,

        /// An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability.
        #[schema(example = 1)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<i32>,

        /// The maximum number of tokens to generate in the chat completion
        #[schema(example = 4096, default = 16)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_tokens: Option<i32>,

        /// How many chat completion choices to generate for each input message
        #[schema(example = 1)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub n: Option<i32>,

        /// Number between -2.0 and 2.0. Positive values penalize new tokens based on
        /// whether they appear in the text so far
        #[schema(example = 0.0)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub presence_penalty: Option<f32>,

        /// If specified, our system will make a best effort to sample deterministically
        #[schema(example = 123)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub seed: Option<i64>,

        /// Up to 4 sequences where the API will stop generating further tokens
        #[schema(example = "json([\"stop\", \"halt\"])", default = "[]")]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stop: Option<Vec<String>>,

        /// Whether to stream back partial progress
        #[schema(example = false)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream: Option<bool>,

        /// Options for streaming response. Only set this when you set stream: true.
        #[schema(example = json!({"include_usage": true}))]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream_options: Option<stream_options::StreamOptions>,

        /// The suffix that comes after a completion of inserted text.
        #[schema(example = "json(\"\\n\")")]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub suffix: Option<String>,

        /// What sampling temperature to use, between 0 and 2
        #[schema(example = 0.7)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f32>,

        /// An alternative to sampling with temperature
        #[schema(example = 1.0)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f32>,

        /// A unique identifier representing your end-user
        #[schema(example = "user-1234")]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user: Option<String>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
    #[serde(untagged)]
    pub enum CompletionsPrompt {
        /// A single string prompt
        #[serde(rename = "single")]
        Single(String),

        /// An array of strings prompts
        #[serde(rename = "list")]
        List(Vec<String>),

        /// An array of tokens
        #[serde(rename = "tokens")]
        Tokens(Vec<u32>),

        /// An array of token arrays
        #[serde(rename = "token_arrays")]
        TokenArrays(Vec<Vec<u32>>),
    }

    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    pub struct CompletionsResponse {
        /// Array of completion choices response
        #[schema(example = json!([
            {
                "text": "This is a test",
                "index": 0,
                "logprobs": null,
                "finish_reason": "stop"
            }
        ]))]
        pub choices: Vec<CompletionChoice>,

        /// The usage information for the request
        #[schema(example = json!({
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20
        }))]
        pub usage: Usage,

        /// The creation time of the request
        #[schema(example = "2021-01-01T00:00:00.000Z")]
        pub created: i64,

        /// The ID of the request
        #[schema(example = "cmpl-1234567890")]
        pub id: String,

        /// The model used for the request
        #[schema(example = "meta-llama/Llama-3.3-70B-Instruct")]
        pub model: String,

        /// The object type
        #[schema(example = "text_completion")]
        pub object: String,

        /// The system fingerprint
        #[schema(example = "system-fingerprint")]
        pub system_fingerprint: String,
    }

    /// A completion choice response
    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    pub struct CompletionChoice {
        /// The generated text
        #[schema(example = "This is a test")]
        pub text: String,

        /// The index of the choice in the list of choices
        #[schema(example = 0)]
        pub index: i32,

        /// The log probabilities of the chosen tokens
        #[schema(example = "null")]
        pub logprobs: Option<LogProbs>,

        /// The reason the model stopped generating tokens
        #[schema(example = "stop")]
        pub finish_reason: String,
    }

    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    pub struct LogProbs {
        /// The tokens
        #[schema(example = json!([
            "Hello ",
            "world"
        ]))]
        pub tokens: Vec<String>,

        /// The log probabilities of the tokens
        #[schema(example = json!([
            0.5,
            -0.5
        ]))]
        pub token_logprobs: Vec<f32>,

        /// The top log probabilities
        #[schema(example = json!([
            {
                "Hello ": -0.2,
                "world": -0.8
            }
        ]))]
        pub top_logprobs: Vec<HashMap<String, f32>>,

        /// The text offset of the tokens
        #[schema(example = json!([
            0,
            10
        ]))]
        pub text_offset: Vec<u32>,
    }

    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    pub struct Usage {
        /// The number of prompt tokens used
        #[schema(example = 10)]
        pub prompt_tokens: u32,

        /// The number of completion tokens used
        #[schema(example = 10)]
        pub completion_tokens: u32,

        /// The total number of tokens used
        #[schema(example = 20)]
        pub total_tokens: u32,

        /// The details of the completion tokens
        #[schema(example = json!({
            "accepted_prediction_tokens": 10,
            "audio_tokens": 0,
            "reasoning_tokens": 10,
            "rejected_prediction_tokens": 0
        }))]
        pub completion_tokens_details: CompletionTokensDetails,

        /// The details of the prompt tokens
        #[schema(example = json!({
            "audio_tokens": 0,
            "cached_tokens": 10,
        }))]
        pub prompt_tokens_details: PromptTokensDetails,
    }

    /// The details of the completion tokens
    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    #[allow(clippy::struct_field_names)]
    pub struct CompletionTokensDetails {
        /// The number of tokens in the completion
        #[schema(example = 10)]
        pub accepted_prediction_tokens: u32,

        /// The number of audio tokens
        #[schema(example = 0)]
        pub audio_tokens: u32,

        /// The number of reasoning tokens
        #[schema(example = 10)]
        pub reasoning_tokens: u32,

        /// The number of rejected prediction tokens
        #[schema(example = 0)]
        pub rejected_prediction_tokens: u32,
    }

    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    pub struct PromptTokensDetails {
        /// The number of audio tokens
        #[schema(example = 0)]
        pub audio_tokens: u32,

        /// The number of cached tokens
        #[schema(example = 10)]
        pub cached_tokens: u32,
    }

    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    pub struct CreateCompletionsStreamRequest {
        #[serde(flatten)]
        pub completion_request: CompletionsRequest,

        /// Whether to stream back partial progress. Must be true for this request type.
        #[schema(default = true)]
        pub stream: bool,
    }

    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    pub struct CompletionsStreamResponse {
        /// Array of completion choices response
        #[schema(example = json!([
            {
                "text": "This is a test",
                "index": 0,
                "logprobs": null,
                "finish_reason": "stop"
            }
        ]))]
        pub choices: Vec<CompletionChoice>,

        /// The creation time of the request
        #[schema(example = "2021-01-01T00:00:00.000Z")]
        pub created: i64,

        /// The ID of the request
        #[schema(example = "cmpl-1234567890")]
        pub id: String,

        /// The model used for the request
        #[schema(example = "meta-llama/Llama-3.3-70B-Instruct")]
        pub model: String,

        /// The object type
        #[schema(example = "text_completion")]
        pub object: String,

        /// The system fingerprint
        #[schema(example = "system-fingerprint")]
        pub system_fingerprint: String,
    }
}
