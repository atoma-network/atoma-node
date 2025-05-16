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
use openai_api::{
    completion_choice::{
        ChatCompletionChoice, ChatCompletionChunkChoice, ChatCompletionChunkDelta,
    },
    logprobs::{ChatCompletionLogProb, ChatCompletionLogProbs, ChatCompletionLogProbsContent},
    message::ChatCompletionMessage,
    message_content::{MessageContent, MessageContentPart},
    stop_reason::StopReason,
    token_details::PromptTokensDetails,
    tools::{
        ChatCompletionChunkDeltaToolCall, ChatCompletionChunkDeltaToolCallFunction, Tool, ToolCall,
        ToolCallFunction, ToolFunction,
    },
    usage::CompletionUsage,
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
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
    request_model::{RequestModel, TokensEstimate},
    vllm_metrics::get_best_available_chat_completions_service_url,
    DEFAULT_MAX_TOKENS,
};

/// The path for confidential chat completions requests
pub const CONFIDENTIAL_CHAT_COMPLETIONS_PATH: &str = "/v1/confidential/chat/completions";

/// The key for the content parameter in the request body
pub const CONTENT_KEY: &str = "content";

/// The path for chat completions requests
pub const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

/// The keep-alive interval in seconds
const STREAM_KEEP_ALIVE_INTERVAL_IN_SECONDS: u64 = 15;

/// The key for the max_completion_tokens parameter in the request body
const MAX_COMPLETION_TOKENS_KEY: &str = "max_completion_tokens";

/// The key for the max_tokens parameter in the request body
const MAX_TOKENS_KEY: &str = "max_tokens";

/// The key for the model parameter in the request body
const MODEL_KEY: &str = "model";

/// The key for the messages parameter in the request body
const MESSAGES_KEY: &str = "messages";

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
    paths(chat_completions_handler),
    components(schemas(
        ChatCompletionRequest,
        ChatCompletionMessage,
        ChatCompletionResponse,
        ChatCompletionChoice,
        CompletionUsage,
        PromptTokensDetails,
        ChatCompletionChunk,
        ChatCompletionChunkChoice,
        ChatCompletionChunkDelta,
        ToolCall,
        Tool,
        ToolCallFunction,
        ToolFunction,
        MessageContent,
        MessageContentPart,
        StopReason,
        ChatCompletionChunkDeltaToolCall,
        ChatCompletionChunkDeltaToolCallFunction,
        ChatCompletionLogProbs,
        ChatCompletionLogProbsContent,
        ChatCompletionLogProb,
    ))
)]
pub struct ChatCompletionsOpenApi;

/// Create chat completion
///
/// This handler performs several key operations:
/// 1. Forwards the chat completion request to the inference service
/// 2. Signs the response using the node's keystore
/// 3. Tracks token usage for the stack
///
/// # Arguments
///
/// * `Extension((stack_small_id, estimated_total_tokens))` - Stack ID and estimated tokens count from middleware
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
    request_body = ChatCompletionRequest,
    responses(
        (status = OK, description = "Chat completion successful", body = ChatCompletionResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
#[instrument(
    level = "info",
    skip_all,
    fields(path = request_metadata.endpoint_path),
    err
)]
pub async fn chat_completions_handler(
    Extension(request_metadata): Extension<RequestMetadata>,
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Result<Response<Body>, AtomaServiceError> {
    let RequestMetadata {
        stack_small_id,
        estimated_output_tokens,
        price_per_one_million_tokens,
        user_address,
        num_input_tokens,
        payload_hash,
        client_encryption_metadata,
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
        price_per_one_million_tokens,
        user_address.clone(),
        is_stream,
        payload.clone(),
        num_input_tokens,
        estimated_output_tokens,
        client_encryption_metadata,
        headers,
    )
    .await
    {
        Ok(response) => {
            CHAT_COMPLETIONS_ESTIMATED_TOTAL_TOKENS.add(
                num_input_tokens + estimated_output_tokens,
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
                    num_input_tokens + estimated_output_tokens,
                    0,
                    &endpoint,
                    concurrent_requests,
                )?;
            } else {
                update_fiat_amount(
                    &state.state_manager_sender,
                    user_address,
                    num_input_tokens,
                    0,
                    estimated_output_tokens,
                    0,
                    price_per_one_million_tokens,
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
    paths(chat_completions_handler),
    components(schemas(ChatCompletionRequest, ConfidentialComputeResponse))
)]
pub struct ConfidentialChatCompletionsOpenApi;

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
///   - `estimated_total_tokens` - Predicted compute cost
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
pub async fn confidential_chat_completions_handler(
    Extension(request_metadata): Extension<RequestMetadata>,
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Result<Response<Body>, AtomaServiceError> {
    let RequestMetadata {
        stack_small_id,
        num_input_tokens,
        estimated_output_tokens,
        user_address,
        price_per_one_million_tokens,
        payload_hash,
        client_encryption_metadata,
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
        price_per_one_million_tokens,
        user_address.clone(),
        is_stream,
        payload.clone(),
        num_input_tokens,
        estimated_output_tokens,
        client_encryption_metadata,
        headers,
    )
    .await
    {
        Ok(response) => {
            CHAT_COMPLETIONS_ESTIMATED_TOTAL_TOKENS.add(
                num_input_tokens + estimated_output_tokens,
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
                    "chat-completions/confidential_chat_completions_handler",
                );
                update_stack_num_compute_units(
                    &state.state_manager_sender,
                    stack_small_id,
                    num_input_tokens + estimated_output_tokens,
                    0,
                    &endpoint,
                    concurrent_requests,
                )?;
            } else {
                update_fiat_amount(
                    &state.state_manager_sender,
                    user_address,
                    num_input_tokens,
                    0,
                    estimated_output_tokens,
                    0,
                    price_per_one_million_tokens,
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
/// * `estimated_total_tokens` - Estimated tokens for the request
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
        path = CHAT_COMPLETIONS_PATH,
        stack_small_id,
        estimated_total_tokens,
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
    price_per_one_million_tokens: i64,
    user_address: String,
    is_stream: bool,
    payload: Value,
    num_input_tokens: i64,
    estimated_output_tokens: i64,
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
            estimated_output_tokens,
            price_per_one_million_tokens,
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
            num_input_tokens,
            estimated_output_tokens,
            price_per_one_million_tokens,
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
/// * `estimated_total_tokens` - Estimated tokens count for the request
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
        path = CHAT_COMPLETIONS_PATH,
        completion_type = "non-streaming",
        stack_small_id,
        estimated_total_tokens,
        payload_hash
    ),
    err
)]
#[allow(clippy::too_many_arguments)]
async fn handle_non_streaming_response(
    state: &AppState,
    payload: Value,
    stack_small_id: Option<i64>,
    num_input_tokens: i64,
    estimated_output_tokens: i64,
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
    let (input_tokens, output_tokens) = utils::extract_total_num_tokens(&response_body, model);

    utils::serve_non_streaming_response(
        state,
        response_body,
        stack_small_id,
        num_input_tokens,
        estimated_output_tokens,
        price_per_one_million_compute_units,
        user_address,
        input_tokens,
        output_tokens,
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
/// * `estimated_total_tokens` - Estimated compute units count for the request
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
        path = CHAT_COMPLETIONS_PATH,
        completion_type = "streaming",
        stack_small_id,
        estimated_total_tokens = num_input_tokens + estimated_output_tokens,
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
    estimated_output_tokens: i64,
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
            chat_completions_service_url, CHAT_COMPLETIONS_PATH
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
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| AtomaServiceError::InternalError {
                message: format!("Failed to read response body: {}", e),
                endpoint: endpoint.to_string(),
            })?;
        // Try to parse the error message from the response body
        let error_message = serde_json::from_slice::<Value>(&bytes).map_or_else(
            |_| {
                status
                    .canonical_reason()
                    .unwrap_or("Unknown error")
                    .to_string()
            },
            |json| {
                json.get("error")
                    .or_else(|| json.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| status.canonical_reason().unwrap_or("Unknown error"))
                    .to_string()
            },
        );
        handle_status_code_error(status, &endpoint, &error_message)?;

        return Err(AtomaServiceError::InternalError {
            message: format!("Unexpected error handling failure: {}", error_message),
            endpoint: endpoint.clone(),
        });
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
        estimated_output_tokens,
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
pub struct RequestModelChatCompletions {
    /// Array of message objects that represent the conversation history
    /// Each message should contain a "role" (system/user/assistant) and "content"
    /// The content can be a string or an array of content parts.
    messages: Vec<Value>,

    /// The maximum number of tokens to generate in the completion
    /// This limits the length of the model's response
    max_completion_tokens: u64,
}

impl RequestModel for RequestModelChatCompletions {
    fn new(request: &Value) -> Result<Self, AtomaServiceError> {
        let messages = request
            .get(MESSAGES_KEY)
            .and_then(|m| m.as_array())
            .ok_or_else(|| AtomaServiceError::InvalidBody {
                message: "Missing or invalid 'messages' field".to_string(),
                endpoint: CHAT_COMPLETIONS_PATH.to_string(),
            })?;

        let max_completion_tokens = request
            .get(MAX_COMPLETION_TOKENS_KEY)
            .or_else(|| request.get(MAX_TOKENS_KEY))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(DEFAULT_MAX_TOKENS);

        Ok(Self {
            messages: messages.clone(),
            max_completion_tokens,
        })
    }

    /// Computes the total number of tokens for the chat completion request.
    ///
    /// This is used to estimate the cost of the chat completion request, on the proxy side.
    /// We support either string or array of content parts. We further assume that all content messages
    /// share the same previous messages. That said, we further assume that content parts formatted into arrays
    /// are to be concatenated and treated as a single message, by the model and from the estimate point of view.
    fn get_tokens_estimate(
        &self,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<TokensEstimate, AtomaServiceError> {
        // In order to account for the possibility of not taking into account possible additional special tokens,
        // which might not be considered by the tokenizer, we add a small overhead to the total number of tokens, per message.
        const MESSAGE_OVERHEAD_TOKENS: u64 = 3;
        let Some(tokenizer) = tokenizer else {
            return Err(AtomaServiceError::InternalError {
                message: "Tokenizer is required for current model, but is not currently available"
                    .to_string(),
                endpoint: CHAT_COMPLETIONS_PATH.to_string(),
            });
        };
        // Helper function to count tokens for a text string
        let count_text_tokens = |text: &str| -> Result<u64, AtomaServiceError> {
            Ok(tokenizer
                .encode(text, true)
                .map_err(|err| AtomaServiceError::InternalError {
                    message: format!("Failed to encode message: {err:?}"),
                    endpoint: CHAT_COMPLETIONS_PATH.to_string(),
                })?
                .get_ids()
                .len() as u64)
        };

        let mut total_num_messages_tokens = 0;

        for message in &self.messages {
            let content = message
                .get(CONTENT_KEY)
                .and_then(|content| MessageContent::deserialize(content).ok())
                .ok_or_else(|| AtomaServiceError::InvalidBody {
                    message: "Missing or invalid message content".to_string(),
                    endpoint: CHAT_COMPLETIONS_PATH.to_string(),
                })?;

            match content {
                MessageContent::Text(text) => {
                    let num_tokens = count_text_tokens(&text)?;
                    total_num_messages_tokens += num_tokens + MESSAGE_OVERHEAD_TOKENS;
                }
                MessageContent::Array(parts) => {
                    if parts.is_empty() {
                        tracing::error!(
                            target = "atoma-service",
                            endpoint = "chat-completions/get_compute_units_estimate",
                            level = "error",
                            "Received empty array of message parts for chat completion request"
                        );
                        return Err(AtomaServiceError::InvalidBody {
                            message: "Missing or invalid message content".to_string(),
                            endpoint: CHAT_COMPLETIONS_PATH.to_string(),
                        });
                    }
                    for part in parts {
                        match part {
                            MessageContentPart::Text { text, .. } => {
                                let num_tokens = count_text_tokens(&text)?;
                                total_num_messages_tokens += num_tokens + MESSAGE_OVERHEAD_TOKENS;
                            }
                            MessageContentPart::Image { .. } => {
                                // TODO: Ensure that for image content parts, we have a way to estimate the number of tokens,
                                // which can depend on the size of the image and the output description.
                            }
                        }
                    }
                }
            }
        }
        // add the max completion tokens, to account for the response
        Ok(TokensEstimate {
            num_input_tokens: total_num_messages_tokens,
            max_output_tokens: self.max_completion_tokens,
            max_total_tokens: total_num_messages_tokens + self.max_completion_tokens,
        })
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
        CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS, CHAT_COMPLETIONS_PATH, MODEL_KEY, UNKNOWN_MODEL,
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
        let chat_completions_service_url_services = state
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
        let (chat_completions_service_url, status_code) =
            get_best_available_chat_completions_service_url(
                chat_completions_service_url_services,
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
            chat_completions_service_url, CHAT_COMPLETIONS_PATH
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
            let status = response.status();
            let bytes = response
                .bytes()
                .await
                .map_err(|e| AtomaServiceError::InternalError {
                    message: format!("Failed to read response body: {}", e),
                    endpoint: endpoint.to_string(),
                })?;
            // Try to parse the error message from the response body
            let error_message = serde_json::from_slice::<Value>(&bytes).map_or_else(
                |_| {
                    status
                        .canonical_reason()
                        .unwrap_or("Unknown error")
                        .to_string()
                },
                |json| {
                    json.get("error")
                        .or_else(|| json.get("message"))
                        .and_then(|v| v.as_str())
                        .unwrap_or_else(|| status.canonical_reason().unwrap_or("Unknown error"))
                        .to_string()
                },
            );
            handle_status_code_error(status, endpoint, &error_message)?;

            return Err(AtomaServiceError::InternalError {
                message: format!("Unexpected error handling failure: {}", error_message),
                endpoint: endpoint.to_string(),
            });
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
    pub fn extract_total_num_tokens(response_body: &Value, model: &str) -> (i64, i64) {
        let mut input_tokens = 0;
        let mut output_tokens = 0;
        if let Some(usage) = response_body.get(USAGE_KEY) {
            if let Some(prompt_tokens) = usage.get(PROMPT_TOKENS_KEY) {
                let prompt_tokens = prompt_tokens.as_u64().unwrap_or(0);
                CHAT_COMPLETIONS_INPUT_TOKENS_METRICS
                    .add(prompt_tokens, &[KeyValue::new(MODEL_KEY, model.to_owned())]);
                input_tokens += prompt_tokens;
            }
            if let Some(completion_tokens) = usage.get(COMPLETION_TOKENS_KEY) {
                let completion_tokens = completion_tokens.as_u64().unwrap_or(0);
                CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS.add(
                    completion_tokens,
                    &[KeyValue::new(MODEL_KEY, model.to_owned())],
                );
                output_tokens += completion_tokens;
            }
        }
        (input_tokens as i64, output_tokens as i64)
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
    /// * `estimated_total_tokens` - Initially estimated tokens for the request
    /// * `total_tokens` - Actual tokens used by the request
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
    /// - Info-level tracing with fields: stack_small_id, estimated_total_tokens, payload_hash, endpoint
    /// - Prometheus metrics for response timing
    /// - Detailed logging of compute unit usage
    #[instrument(
        level = "info",
        skip_all,
        fields(stack_small_id, estimated_total_tokens = estimated_input_tokens + estimated_output_tokens, payload_hash, endpoint),
        err
    )]
    #[allow(clippy::too_many_arguments)]
    pub async fn serve_non_streaming_response(
        state: &AppState,
        mut response_body: Value,
        stack_small_id: Option<i64>,
        estimated_input_tokens: i64,
        estimated_output_tokens: i64,
        price_per_one_million_tokens: i64,
        user_address: String,
        input_tokens: i64,
        output_tokens: i64,
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
            estimated_total_tokens = estimated_input_tokens + estimated_output_tokens,
            payload_hash = hex::encode(payload_hash),
            "Total compute units: {}",
            input_tokens + output_tokens,
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

        // NOTE: We need to update the stack num tokens, because the inference response might have produced
        // less tokens than estimated what we initially estimated, from the middleware.
        //
        // NOTE: We update the total number of tokens as a final step, as if some error occurs, we don't want
        // to update the stack num tokens beforehand.
        //
        // NOTE: We also decrement the concurrent requests count, as we are done processing the request.
        if let Some(stack_small_id) = stack_small_id {
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
                "Concurrent requests have been decremented. Updating stack num compute units for stack small id: {stack_small_id}"
            );
            update_stack_num_compute_units(
                &state.state_manager_sender,
                stack_small_id,
                estimated_input_tokens + estimated_output_tokens,
                input_tokens + output_tokens,
                &endpoint,
                concurrent_requests,
            )?;
        } else {
            update_fiat_amount(
                &state.state_manager_sender,
                user_address,
                estimated_input_tokens,
                input_tokens,
                estimated_output_tokens,
                output_tokens,
                price_per_one_million_tokens,
                &endpoint,
            )?;
        }

        Ok(response_body)
    }
}

pub mod openai_api {
    use serde::{Deserialize, Deserializer, Serialize};
    use serde_json::Value;
    use utoipa::ToSchema;

    /// Represents the create chat completion request.
    ///
    /// This is used to represent the create chat completion request in the chat completion request.
    /// It can be either a chat completion or a chat completion stream.
    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    pub struct CreateChatCompletionRequest {
        #[serde(flatten)]
        pub chat_completion_request: ChatCompletionRequest,

        /// Whether to stream back partial progress. Must be false for this request type.
        #[schema(default = false)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream: Option<bool>,
    }

    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    pub struct CreateChatCompletionStreamRequest {
        #[serde(flatten)]
        pub chat_completion_request: ChatCompletionRequest,

        /// Whether to stream back partial progress. Must be true for this request type.
        #[schema(default = true)]
        pub stream: bool,
    }

    /// Represents the chat completion request.
    ///
    /// This is used to represent the chat completion request in the chat completion request.
    /// It can be either a chat completion or a chat completion stream.
    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    pub struct ChatCompletionRequest {
        /// ID of the model to use
        #[schema(example = "meta-llama/Llama-3.3-70B-Instruct")]
        pub model: String,

        /// A list of messages comprising the conversation so far
        pub messages: Vec<message::ChatCompletionMessage>,

        /// What sampling temperature to use, between 0 and 2
        #[schema(example = 0.7)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f32>,

        /// An alternative to sampling with temperature
        #[schema(example = 1.0)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f32>,

        /// How many chat completion choices to generate for each input message
        #[schema(example = 1)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub n: Option<i32>,

        /// Whether to stream back partial progress
        #[schema(example = false)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream: Option<bool>,

        /// Up to 4 sequences where the API will stop generating further tokens
        #[schema(example = "json([\"stop\", \"halt\"])", default = "[]")]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stop: Option<Vec<String>>,

        /// The maximum number of tokens to generate in the chat completion
        #[schema(example = 4096)]
        #[serde(skip_serializing_if = "Option::is_none")]
        #[deprecated = "It is recommended to use max_completion_tokens instead"]
        pub max_tokens: Option<i32>,

        /// The maximum number of tokens to generate in the chat completion
        #[schema(example = 4096)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_completion_tokens: Option<i32>,

        /// Number between -2.0 and 2.0. Positive values penalize new tokens based on
        /// whether they appear in the text so far
        #[schema(example = 0.0)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub presence_penalty: Option<f32>,

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
        pub logit_bias: Option<std::collections::HashMap<u32, f32>>,

        /// An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability.
        /// logprobs must be set to true if this parameter is used.
        #[schema(example = 1)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_logprobs: Option<i32>,

        /// A unique identifier representing your end-user
        #[schema(example = "user-1234")]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user: Option<String>,

        /// A list of functions the model may generate JSON inputs for
        #[serde(skip_serializing_if = "Option::is_none")]
        pub functions: Option<Vec<Value>>,

        /// Controls how the model responds to function calls
        #[serde(skip_serializing_if = "Option::is_none")]
        pub function_call: Option<Value>,

        /// The format to return the response in
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_format: Option<response_format::ResponseFormat>,

        /// A list of tools the model may call
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tools: Option<Vec<tools::ChatCompletionToolsParam>>,

        /// Controls which (if any) tool the model should use
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_choice: Option<tools::ToolChoice>,

        /// If specified, our system will make a best effort to sample deterministically
        #[schema(example = 123)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub seed: Option<i64>,

        /// Specifies the latency tier to use for processing the request. This parameter is relevant for customers subscribed to the scale tier service:
        ///
        /// If set to 'auto', and the Project is Scale tier enabled, the system will utilize scale tier credits until they are exhausted.
        /// If set to 'auto', and the Project is not Scale tier enabled, the request will be processed using the default service tier with a lower uptime SLA and no latency guarantee.
        /// If set to 'default', the request will be processed using the default service tier with a lower uptime SLA and no latency guarantee.
        /// When not set, the default behavior is 'auto'.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub service_tier: Option<String>,

        /// Options for streaming response. Only set this when you set stream: true.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream_options: Option<stream_options::StreamOptions>,

        /// Whether to enable parallel tool calls.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub parallel_tool_calls: Option<bool>,
    }

    /// Represents the chat completion response.
    ///
    /// This is used to represent the chat completion response in the chat completion request.
    /// It can be either a chat completion or a chat completion stream.
    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    pub struct ChatCompletionResponse {
        /// A unique identifier for the chat completion.
        #[schema(example = "chatcmpl-123")]
        pub id: String,

        /// The Unix timestamp (in seconds) of when the chat completion was created.
        #[schema(example = 1_677_652_288)]
        pub created: i64,

        /// The model used for the chat completion.
        #[schema(example = "meta-llama/Llama-3.3-70B-Instruct")]
        pub model: String,

        /// A list of chat completion choices.
        pub choices: Vec<completion_choice::ChatCompletionChoice>,

        /// Usage statistics for the completion request.
        pub usage: Option<usage::CompletionUsage>,

        /// The system fingerprint for the completion, if applicable.
        #[schema(example = "fp_44709d6fcb")]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub system_fingerprint: Option<String>,

        /// The object of the chat completion.
        #[schema(example = "chat.completion")]
        pub object: String,

        /// The service tier of the chat completion.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub service_tier: Option<String>,
    }

    /// Represents the chat completion stream response.
    ///
    /// This is used to represent the chat completion stream response in the chat completion request.
    /// It can be either a chat completion chunk or a chat completion stream.
    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    pub struct ChatCompletionStreamResponse {
        /// The stream of chat completion chunks.
        pub data: ChatCompletionChunk,
    }

    /// Represents the chat completion chunk.
    ///
    /// This is used to represent the chat completion chunk in the chat completion request.
    /// It can be either a chat completion chunk or a chat completion chunk choice.
    #[derive(Debug, Serialize, Deserialize, ToSchema)]
    pub struct ChatCompletionChunk {
        /// A unique identifier for the chat completion chunk.
        #[schema(example = "chatcmpl-123")]
        pub id: String,

        /// The object of the chat completion chunk (which is always `chat.completion.chunk`)
        #[schema(example = "chat.completion.chunk")]
        pub object: String,

        /// The Unix timestamp (in seconds) of when the chunk was created.
        #[schema(example = 1_677_652_288)]
        pub created: i64,

        /// The model used for the chat completion.
        #[schema(example = "meta-llama/Llama-3.3-70B-Instruct")]
        pub model: String,

        /// A list of chat completion chunk choices.
        pub choices: Vec<completion_choice::ChatCompletionChunkChoice>,

        /// Usage statistics for the completion request.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub usage: Option<usage::CompletionUsage>,
    }

    pub mod completion_choice {
        use super::{logprobs, message, stop_reason, tools, Deserialize, Serialize, ToSchema};

        /// Represents the chat completion choice.
        ///
        /// This is used to represent the chat completion choice in the chat completion request.
        /// It can be either a chat completion message or a chat completion chunk.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct ChatCompletionChoice {
            /// The index of this choice in the list of choices.
            #[schema(example = 0)]
            pub index: i32,

            /// The chat completion message.
            pub message: message::ChatCompletionMessage,

            /// The reason the chat completion was finished.
            #[schema(example = "stop")]
            pub finish_reason: Option<String>,

            /// Log probability information for the choice, if applicable.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub logprobs: Option<logprobs::ChatCompletionLogProbs>,
        }

        /// Represents the chat completion chunk choice.
        ///
        /// This is used to represent the chat completion chunk choice in the chat completion request.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct ChatCompletionChunkChoice {
            /// The index of this choice in the list of choices.
            #[schema(example = 0)]
            pub index: i32,

            /// The chat completion delta message for streaming.
            pub delta: ChatCompletionChunkDelta,

            /// Log probability information for the choice, if applicable.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub logprobs: Option<logprobs::ChatCompletionLogProbs>,

            /// The reason the chat completion was finished, if applicable.
            #[schema(example = "stop")]
            #[serde(skip_serializing_if = "Option::is_none")]
            pub finish_reason: Option<String>,

            /// The reason the chat completion was stopped, if applicable.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub stop_reason: Option<stop_reason::StopReason>,
        }

        /// Represents the chat completion chunk delta.
        ///
        /// This is used to represent the chat completion chunk delta in the chat completion request.
        /// It can be either a chat completion chunk delta message or a chat completion chunk delta choice.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct ChatCompletionChunkDelta {
            /// The role of the message author, if present in this chunk.
            #[schema(example = "assistant")]
            #[serde(skip_serializing_if = "Option::is_none")]
            pub role: Option<String>,

            /// The content of the message, if present in this chunk.
            #[schema(example = "Hello")]
            #[serde(skip_serializing_if = "Option::is_none")]
            pub content: Option<String>,

            /// The reasoning content, if present in this chunk.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub reasoning_content: Option<String>,

            /// The tool calls information, if present in this chunk.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub tool_calls: Option<Vec<tools::ChatCompletionChunkDeltaToolCall>>,
        }
    }

    pub mod logprobs {
        use super::{Deserialize, Serialize, ToSchema};

        /// Represents the chat completion log probs.
        ///
        /// This is used to represent the chat completion log probs in the chat completion request.
        /// It can be either a chat completion log probs or a chat completion log probs choice.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct ChatCompletionLogProbs {
            /// The log probs of the chat completion.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub content: Option<Vec<ChatCompletionLogProbsContent>>,
        }

        /// Represents the chat completion log probs content.
        ///
        /// This is used to represent the chat completion log probs content in the chat completion request.
        /// It can be either a chat completion log probs content or a chat completion log probs content choice.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct ChatCompletionLogProbsContent {
            top_logprobs: Vec<ChatCompletionLogProb>,
        }

        /// Represents the chat completion log prob.
        ///
        /// This is used to represent the chat completion log prob in the chat completion request.
        /// It can be either a chat completion log prob or a chat completion log prob choice.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct ChatCompletionLogProb {
            /// The log prob of the chat completion.
            pub logprob: f32,

            /// The token of the chat completion.
            pub token: String,

            /// A list of integers representing the UTF-8 bytes representation of the token.
            /// Useful in instances where characters are represented by multiple tokens and their byte
            /// representations must be combined to generate the correct text representation.
            /// Can be null if there is no bytes representation for the token.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub bytes: Option<Vec<i32>>,
        }
    }

    pub mod message {
        use super::{message_content, tools, Deserialize, Serialize, ToSchema};

        /// A message that is part of a conversation which is based on the role
        /// of the author of the message.
        ///
        /// This is used to represent the message in the chat completion request.
        /// It can be either a system message, a user message, an assistant message, or a tool message.
        #[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
        #[serde(tag = "role", rename_all = "snake_case")]
        pub enum ChatCompletionMessage {
            /// The role of the messages author, in this case system.
            System {
                /// The contents of the message.
                #[serde(default, skip_serializing_if = "Option::is_none")]
                content: Option<message_content::MessageContent>,
                /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
                #[serde(default, skip_serializing_if = "Option::is_none")]
                name: Option<String>,
            },
            /// The role of the messages author, in this case user.
            User {
                /// The contents of the message.
                #[serde(default, skip_serializing_if = "Option::is_none")]
                content: Option<message_content::MessageContent>,
                /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
                #[serde(default, skip_serializing_if = "Option::is_none")]
                name: Option<String>,
            },
            /// The role of the messages author, in this case assistant.
            Assistant {
                /// The contents of the message.
                #[serde(default, skip_serializing_if = "Option::is_none")]
                content: Option<message_content::MessageContent>,
                /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
                #[serde(default, skip_serializing_if = "Option::is_none")]
                name: Option<String>,
                /// The refusal message by the assistant.
                #[serde(default, skip_serializing_if = "Option::is_none")]
                refusal: Option<String>,
                /// The tool calls generated by the model, such as function calls.
                #[serde(default, skip_serializing_if = "Vec::is_empty")]
                tool_calls: Vec<tools::ToolCall>,
            },
            /// The role of the messages author, in this case tool.
            Tool {
                /// The contents of the message.
                #[serde(default, skip_serializing_if = "Option::is_none")]
                content: Option<message_content::MessageContent>,
                /// Tool call that this message is responding to.
                #[serde(default, skip_serializing_if = "String::is_empty")]
                tool_call_id: String,
            },
        }
    }

    pub mod message_content {
        use serde_json::Value;
        use std::fmt::Write;

        use super::{Deserialize, Deserializer, Serialize, ToSchema};

        /// Represents the content of a message.
        ///
        /// This is used to represent the content of a message in the chat completion request.
        /// It can be either a text or an array of content parts.
        #[derive(Debug, PartialEq, Eq, Serialize, ToSchema)]
        #[serde(untagged)]
        pub enum MessageContent {
            /// The text contents of the message.
            #[serde(rename(serialize = "text", deserialize = "text"))]
            Text(String),
            /// An array of content parts with a defined type, each can be of type text or image_url when passing in images.
            /// You can pass multiple images by adding multiple image_url content parts. Image input is only supported when using the gpt-4o model.
            #[serde(rename(serialize = "array", deserialize = "array"))]
            Array(Vec<MessageContentPart>),
        }

        /// Represents a part of a message content.
        ///
        /// This is used to represent the content of a message in the chat completion request.
        /// It can be either a text or an image.
        #[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
        #[serde(untagged)]
        pub enum MessageContentPart {
            #[serde(rename(serialize = "text", deserialize = "text"))]
            Text {
                /// The type of the content part.
                #[serde(rename(serialize = "type", deserialize = "type"))]
                r#type: String,
                /// The text content.
                text: String,
            },
            #[serde(rename(serialize = "image", deserialize = "image"))]
            Image {
                /// The type of the content part.
                #[serde(rename(serialize = "type", deserialize = "type"))]
                r#type: String,
                /// The image URL.
                image_url: MessageContentPartImageUrl,
            },
        }

        impl std::fmt::Display for MessageContent {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::Text(text) => write!(f, "{text}"),
                    Self::Array(parts) => {
                        let mut content = String::new();
                        for part in parts {
                            content.write_str(&format!("{part}\n"))?;
                        }
                        write!(f, "{content}")
                    }
                }
            }
        }

        // We manually implement Deserialize here for more control.
        impl<'de> Deserialize<'de> for MessageContent {
            fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value: Value = Value::deserialize(deserializer)?;

                if let Some(s) = value.as_str() {
                    return Ok(Self::Text(s.to_string()));
                }

                if let Some(arr) = value.as_array() {
                    let parts: std::result::Result<Vec<MessageContentPart>, _> = arr
                        .iter()
                        .map(|v| {
                            serde_json::from_value(v.clone()).map_err(serde::de::Error::custom)
                        })
                        .collect();
                    return Ok(Self::Array(parts?));
                }

                Err(serde::de::Error::custom(
                    "Expected a string or an array of content parts",
                ))
            }
        }

        impl std::fmt::Display for MessageContentPart {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::Text { r#type, text } => {
                        write!(f, "{type}: {text}")
                    }
                    Self::Image { r#type, image_url } => {
                        write!(f, "{type}: [Image URL: {image_url}]")
                    }
                }
            }
        }

        /// Represents the image URL of a message content part.
        ///
        /// This is used to represent the image URL of a message content part in the chat completion request.
        /// It can be either a URL or a base64 encoded image data.
        #[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
        #[serde(rename(serialize = "image_url", deserialize = "image_url"))]
        pub struct MessageContentPartImageUrl {
            /// Either a URL of the image or the base64 encoded image data.
            url: String,
            /// Specifies the detail level of the image.
            detail: Option<String>,
        }

        /// Implementing Display for MessageContentPartImageUrl
        impl std::fmt::Display for MessageContentPartImageUrl {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match &self.detail {
                    Some(detail) => write!(f, "Image URL: {}, Detail: {}", self.url, detail),
                    None => write!(f, "Image URL: {}", self.url),
                }
            }
        }
    }

    pub mod response_format {
        use super::{Deserialize, Serialize, ToSchema};

        /// The format to return the response in.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        #[serde(rename_all = "snake_case")]
        pub enum ResponseFormatType {
            Text,
            JsonObject,
            JsonSchema,
        }

        /// The format to return the response in.
        ///
        /// This is used to represent the format to return the response in in the chat completion request.
        /// It can be either text, json_object, or json_schema.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct JsonSchemaResponseFormat {
            /// The name of the response format.
            pub name: String,

            /// The description of the response format.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub description: Option<String>,

            /// The JSON schema of the response format.
            #[serde(skip_serializing_if = "Option::is_none")]
            #[serde(rename = "schema")]
            pub json_schema: Option<serde_json::Value>,

            /// Whether to strictly validate the JSON schema.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub strict: Option<bool>,
        }

        /// The format to return the response in.
        ///
        /// This is used to represent the format to return the response in in the chat completion request.
        /// It can be either text, json_object, or json_schema.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct ResponseFormat {
            /// The type of the response format.
            #[serde(rename = "type")]
            pub format_type: ResponseFormatType,

            /// The JSON schema of the response format.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub json_schema: Option<JsonSchemaResponseFormat>,
        }
    }

    pub mod stream_options {
        use super::{Deserialize, Serialize, ToSchema};

        /// Specifies the stream options for the request.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct StreamOptions {
            /// If set, an additional chunk will be streamed before the data: [DONE] message.
            /// The usage field on this chunk shows the token usage statistics for the entire request, and the choices field
            /// will always be an empty array. All other chunks will also include a usage field, but with a null value.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub include_usage: Option<bool>,
        }
    }

    pub mod stop_reason {
        use super::{Deserialize, Serialize, ToSchema};
        use serde::Deserializer;
        use serde_json::Value;

        /// Represents the stop reason.
        ///
        /// This is used to represent the stop reason in the chat completion request.
        /// It can be either a stop reason or a stop reason choice.
        #[derive(Debug, ToSchema)]
        pub enum StopReason {
            Int(u32),
            String(String),
        }

        // Add custom implementations for serialization/deserialization if needed
        impl<'de> Deserialize<'de> for StopReason {
            fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = Value::deserialize(deserializer)?;
                value.as_u64().map_or_else(
                    || {
                        value.as_str().map_or_else(
                            || Err(serde::de::Error::custom("Expected string or integer")),
                            |s| Ok(Self::String(s.to_string())),
                        )
                    },
                    |n| {
                        Ok(Self::Int(u32::try_from(n).map_err(|_| {
                            serde::de::Error::custom("Expected integer")
                        })?))
                    },
                )
            }
        }

        impl Serialize for StopReason {
            fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                match self {
                    Self::Int(n) => serializer.serialize_u32(*n),
                    Self::String(s) => serializer.serialize_str(s),
                }
            }
        }
    }

    pub mod token_details {
        use super::{Deserialize, Serialize, ToSchema};

        /// Represents the prompt tokens details.
        ///
        /// This is used to represent the prompt tokens details in the chat completion request.
        /// It can be either a prompt tokens details or a prompt tokens details choice.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct PromptTokensDetails {
            /// Number of tokens in the prompt that were cached.
            #[schema(example = 1)]
            pub cached_tokens: i32,
        }
    }

    pub mod tools {
        use serde_json::Value;
        use std::collections::HashMap;

        use super::{Deserialize, Serialize, ToSchema};

        /// Represents the function that the model called.
        ///
        /// This is used to represent the function that the model called in the chat completion request.
        /// It can be either a function or a tool call.
        #[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
        pub struct ToolCallFunction {
            /// The name of the function to call.
            name: String,
            /// The arguments to call the function with, as generated by the model in JSON format.
            /// Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema.
            /// Validate the arguments in your code before calling your function.
            arguments: String,
        }

        /// Represents the tool call that the model made.
        ///
        /// This is used to represent the tool call that the model made in the chat completion request.
        /// It can be either a function or a tool.
        #[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
        #[serde(rename(serialize = "tool_call", deserialize = "tool_call"))]
        pub struct ToolCall {
            /// The ID of the tool call.
            id: String,
            /// The type of the tool. Currently, only function is supported.
            #[serde(rename(serialize = "type", deserialize = "type"))]
            r#type: String,
            /// The function that the model called.
            function: ToolCallFunction,
        }

        /// Represents the tool that the model called.
        ///
        /// This is used to represent the tool that the model called in the chat completion request.
        /// It can be either a function or a tool.
        #[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
        #[serde(rename(serialize = "tool", deserialize = "tool"))]
        pub struct Tool {
            /// The type of the tool. Currently, only function is supported.
            #[serde(rename(serialize = "type", deserialize = "type"))]
            r#type: String,
            /// The function that the model called.
            function: ToolFunction,
        }

        /// Represents the function that the model called.
        ///
        /// This is used to represent the function that the model called in the chat completion request.
        /// It can be either a function or a tool.
        #[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
        pub struct ToolFunction {
            /// Description of the function to call.
            #[serde(default, skip_serializing_if = "Option::is_none")]
            description: Option<String>,
            /// The name of the function to call.
            name: String,
            /// The arguments to call the function with, as generated by the model in JSON format.
            #[serde(default, skip_serializing_if = "Option::is_none")]
            parameters: Option<Value>,
            /// Whether to enable strict schema adherence when generating the function call. If set to true, the
            /// model will follow the exact schema defined in the parameters field. Only a subset of JSON Schema is supported when strict is true
            #[serde(default, skip_serializing_if = "Option::is_none")]
            strict: Option<bool>,
        }

        /// Represents the chat completion chunk delta tool call.
        ///
        /// This is used to represent the chat completion chunk delta tool call in the chat completion request.
        /// It can be either a chat completion chunk delta tool call or a chat completion chunk delta tool call function.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct ChatCompletionChunkDeltaToolCall {
            /// The ID of the tool call.
            pub id: String,

            /// The type of the tool call.
            pub r#type: String,

            /// The index of the tool call.
            pub index: i32,

            /// The function of the tool call.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub function: Option<ChatCompletionChunkDeltaToolCallFunction>,
        }

        /// Represents the chat completion chunk delta tool call function.
        ///
        /// This is used to represent the chat completion chunk delta tool call function in the chat completion request.
        /// It can be either a chat completion chunk delta tool call function or a chat completion chunk delta tool call function arguments.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct ChatCompletionChunkDeltaToolCallFunction {
            /// The name of the tool call function.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub name: Option<String>,

            /// The arguments of the tool call function.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub arguments: Option<String>,
        }

        /// A tool that can be used in a chat completion.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct ChatCompletionToolsParam {
            /// The type of the tool.
            #[serde(rename = "type")]
            pub tool_type: String,

            /// The function that the tool will call.
            pub function: ChatCompletionToolFunctionParam,
        }

        /// A function that can be used in a chat completion.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct ChatCompletionToolFunctionParam {
            /// The name of the function.
            pub name: String,

            /// The description of the function.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub description: Option<String>,

            /// The parameters of the function.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub parameters: Option<HashMap<String, serde_json::Value>>,

            /// Whether to strictly validate the parameters of the function.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub strict: Option<bool>,
        }

        /// A tool choice that can be used in a chat completion.
        ///
        /// This is used to represent the tool choice in the chat completion request.
        /// It can be either a literal tool choice or a named tool choice.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        #[serde(untagged)]
        pub enum ToolChoice {
            Literal(ToolChoiceLiteral),
            Named(ChatCompletionNamedToolChoiceParam),
        }

        /// A literal tool choice that can be used in a chat completion.
        ///
        /// This is used to represent the literal tool choice in the chat completion request.
        /// It can be either none or auto.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        #[serde(rename_all = "lowercase")]
        pub enum ToolChoiceLiteral {
            None,
            Auto,
        }

        /// A named tool choice that can be used in a chat completion.
        ///
        /// This is used to represent the named tool choice in the chat completion request.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct ChatCompletionNamedToolChoiceParam {
            /// The type of the tool choice.
            #[serde(rename = "type")]
            pub type_field: String,

            /// The function of the tool choice.
            pub function: ChatCompletionNamedFunction,
        }

        /// A named function that can be used in a chat completion.
        ///
        /// This is used to represent the named function in the chat completion request.
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct ChatCompletionNamedFunction {
            /// The name of the function.
            pub name: String,
        }
    }

    pub mod usage {
        use super::{token_details, Deserialize, Serialize, ToSchema};

        /// Represents the completion usage.
        ///
        /// This is used to represent the completion usage in the chat completion request.
        /// It can be either a completion usage or a completion chunk usage.
        #[allow(clippy::struct_field_names)]
        #[derive(Debug, Serialize, Deserialize, ToSchema)]
        pub struct CompletionUsage {
            /// Number of tokens in the prompt.
            #[schema(example = 9)]
            pub prompt_tokens: i32,

            /// Number of tokens in the completion.
            #[schema(example = 12)]
            pub completion_tokens: i32,

            /// Total number of tokens used (prompt + completion).
            #[schema(example = 21)]
            pub total_tokens: i32,

            /// Details about the prompt tokens.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub prompt_tokens_details: Option<token_details::PromptTokensDetails>,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;
    use std::str::FromStr;
    use tokenizers::Tokenizer;

    async fn load_tokenizer() -> Tokenizer {
        let url =
            "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/tokenizer.json";
        let tokenizer_json = reqwest::get(url).await.unwrap().text().await.unwrap();

        Tokenizer::from_str(&tokenizer_json).unwrap()
    }

    #[tokio::test]
    async fn test_get_compute_units_estimate() {
        let request = RequestModelChatCompletions {
            messages: vec![json!({
                "role": "user",
                "content": "Hello from the other side of Mars"
            })],
            max_completion_tokens: 10,
        };
        let tokenizer = load_tokenizer().await;
        let result = request.get_tokens_estimate(Some(&tokenizer));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().max_total_tokens, 21); // 8 tokens + 3 overhead + 10 completion
    }

    #[tokio::test]
    async fn test_get_compute_units_estimate_multiple_messages() {
        let request = RequestModelChatCompletions {
            messages: vec![
                json!({
                    "role": "user",
                    "content": "Hello from the other side of Mars"
                }),
                json!({
                    "role": "assistant",
                    "content": "Hello from the other side of Mars"
                }),
            ],
            max_completion_tokens: 10,
        };
        let tokenizer = load_tokenizer().await;
        let result = request.get_tokens_estimate(Some(&tokenizer));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().max_total_tokens, 32); // (8+8) tokens + (3+3) overhead + 10 completion
    }

    #[tokio::test]
    async fn test_get_compute_units_estimate_array_content() {
        let request = RequestModelChatCompletions {
            messages: vec![json!({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello from the other side of Mars"
                    },
                    {
                        "type": "text",
                        "text": "Hello from the other side of Mars"
                    }
                ]
            })],
            max_completion_tokens: 10,
        };

        let tokenizer = load_tokenizer().await;
        let result = request.get_tokens_estimate(Some(&tokenizer));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().max_total_tokens, 32); // (8+8) tokens  (3 + 3) overhead + 10 completion
    }

    #[tokio::test]
    async fn test_get_compute_units_estimate_empty_message() {
        let request = RequestModelChatCompletions {
            messages: vec![json!({
                "role": "user",
                "content": ""
            })],
            max_completion_tokens: 10,
        };
        let tokenizer = load_tokenizer().await;
        let result = request.get_tokens_estimate(Some(&tokenizer));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().max_total_tokens, 14); // 1 tokens (special token) + 3 overhead + 10 completion
    }

    #[tokio::test]
    async fn test_get_compute_units_estimate_mixed_content() {
        let request = RequestModelChatCompletions {
            messages: vec![
                json!({
                    "role": "system",
                    "content": "Hello from the other side of Mars"
                }),
                json!({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello from the other side of Mars"
                        },
                        {
                            "type": "image",
                            "image_url": {
                                "url": "http://example.com/image.jpg"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Hello from the other side of Mars"
                        }
                    ]
                }),
            ],
            max_completion_tokens: 15,
        };
        let tokenizer = load_tokenizer().await;
        let result = request.get_tokens_estimate(Some(&tokenizer));
        assert!(result.is_ok());
        // System message: tokens + 15 completion
        // User message array: (2 text parts tokens) + (15 * 2 for text completion for parts)
        let tokens = result.unwrap();
        assert_eq!(tokens.max_total_tokens, 48); // 3 * 8 + 3 * 3 overhead + 15
    }

    #[tokio::test]
    async fn test_get_compute_units_estimate_invalid_content() {
        let request = RequestModelChatCompletions {
            messages: vec![json!({
                "role": "user",
                // Missing "content" field
            })],
            max_completion_tokens: 10,
        };
        let tokenizer = load_tokenizer().await;
        let result = request.get_tokens_estimate(Some(&tokenizer));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AtomaServiceError::InvalidBody { .. }
        ));
    }

    #[tokio::test]
    async fn test_get_compute_units_estimate_empty_array_content() {
        let request = RequestModelChatCompletions {
            messages: vec![json!({
                "role": "user",
                "content": []
            })],
            max_completion_tokens: 10,
        };
        let tokenizer = load_tokenizer().await;
        let result = request.get_tokens_estimate(Some(&tokenizer));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AtomaServiceError::InvalidBody { .. }
        ));
    }

    #[tokio::test]
    async fn test_get_compute_units_estimate_special_characters() {
        let request = RequestModelChatCompletions {
            messages: vec![json!({
                "role": "user",
                "content": "Hello! 👋 🌍 \n\t Special chars: &*#@"
            })],
            max_completion_tokens: 10,
        };
        let tokenizer = load_tokenizer().await;
        let result = request.get_tokens_estimate(Some(&tokenizer));
        assert!(result.is_ok());
        let tokens = result.unwrap();
        assert!(tokens.max_total_tokens > 13); // Should be more than minimum (3 overhead + 10 completion)
    }
}
