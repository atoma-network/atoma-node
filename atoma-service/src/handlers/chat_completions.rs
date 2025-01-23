use crate::{
    handlers::{sign_response_and_update_stack_hash, update_stack_num_compute_units},
    middleware::EncryptionMetadata,
    server::AppState,
    streamer::{Streamer, StreamingEncryptionMetadata},
    types::{ConfidentialComputeRequest, ConfidentialComputeResponse},
};
use atoma_confidential::types::{
    ConfidentialComputeSharedSecretRequest, ConfidentialComputeSharedSecretResponse,
};
use atoma_utils::constants::PAYLOAD_HASH_SIZE;
use axum::{
    body::Body,
    extract::State,
    response::{IntoResponse, Response, Sse},
    Extension, Json,
};
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{info, instrument};
use utoipa::OpenApi;

use serde::{Deserialize, Deserializer, Serialize};
use std::{collections::HashMap, time::Duration};
use utoipa::ToSchema;

use crate::{error::AtomaServiceError, handlers::prometheus::*, middleware::RequestMetadata};

use super::handle_confidential_compute_encryption_response;

/// The path for confidential chat completions requests
pub const CONFIDENTIAL_CHAT_COMPLETIONS_PATH: &str = "/v1/confidential/chat/completions";

/// The path for chat completions requests
pub const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

/// The keep-alive interval in seconds
const STREAM_KEEP_ALIVE_INTERVAL_IN_SECONDS: u64 = 15;

/// The key for the model parameter in the request body
const MODEL_KEY: &str = "model";

/// The key for the stream parameter in the request body
const STREAM_KEY: &str = "stream";

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
///
/// # Paths
///
/// Documents the following endpoint:
/// - `chat_completions_handler`: POST endpoint for chat completions
#[derive(OpenApi)]
#[openapi(
    paths(chat_completions_handler),
    components(schemas(
        ChatCompletionsRequest,
        Message,
        MessageContent,
        MessageContentPart,
        MessageContentPartImageUrl,
        ToolCall,
        ToolCallFunction,
        Tool,
        ToolFunction,
        StopCondition,
        FinishReason,
        Usage,
        Choice,
        ChatCompletionsResponse
    ))
)]
pub(crate) struct ChatCompletionsOpenApi;

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
    request_body = ChatCompletionsRequest,
    responses(
        (status = OK, description = "Chat completion successful", body = ChatCompletionsResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
#[instrument(
    level = "info",
    skip(state, payload),
    fields(path = request_metadata.endpoint_path)
)]
pub async fn chat_completions_handler(
    Extension(request_metadata): Extension<RequestMetadata>,
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Response<Body>, AtomaServiceError> {
    let RequestMetadata {
        stack_small_id,
        estimated_total_compute_units,
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
        .and_then(|s| s.as_bool())
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
        is_stream,
        payload.clone(),
        estimated_total_compute_units,
        client_encryption_metadata,
    )
    .await
    {
        Ok(response) => {
            TOTAL_COMPLETED_REQUESTS.with_label_values(&[model]).inc();
            Ok(response)
        }
        Err(e) => {
            TOTAL_FAILED_REQUESTS.with_label_values(&[model]).inc();
            // NOTE: We need to update the stack number of tokens as the service failed to generate
            // a proper response. For this reason, we set the total number of tokens to 0.
            // This will ensure that the stack number of tokens is not updated, and the stack
            // will not be penalized for the request.
            update_stack_num_compute_units(
                &state.state_manager_sender,
                stack_small_id,
                estimated_total_compute_units,
                0,
                &endpoint,
            )?;
            return Err(AtomaServiceError::InternalError {
                message: format!("Error handling chat completions response: {}", e),
                endpoint: request_metadata.endpoint_path.clone(),
            });
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
    components(schemas(ChatCompletionsRequest, ConfidentialComputeResponse))
)]
pub(crate) struct ConfidentialChatCompletionsOpenApi;

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
    skip(state, payload),
    fields(path = request_metadata.endpoint_path)
)]
pub async fn confidential_chat_completions_handler(
    Extension(request_metadata): Extension<RequestMetadata>,
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Response<Body>, AtomaServiceError> {
    let RequestMetadata {
        stack_small_id,
        estimated_total_compute_units,
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
        .and_then(|s| s.as_bool())
        .unwrap_or_default();

    let model = payload
        .get(MODEL_KEY)
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");

    let endpoint = request_metadata.endpoint_path.clone();

    match handle_response(
        &state,
        endpoint.clone(),
        payload_hash,
        stack_small_id,
        is_stream,
        payload.clone(),
        estimated_total_compute_units,
        client_encryption_metadata,
    )
    .await
    {
        Ok(response) => {
            TOTAL_COMPLETED_REQUESTS.with_label_values(&[model]).inc();
            Ok(response)
        }
        Err(e) => {
            TOTAL_FAILED_REQUESTS.with_label_values(&[model]).inc();
            // NOTE: We need to update the stack number of tokens as the service failed to generate
            // a proper response. For this reason, we set the total number of tokens to 0.
            // This will ensure that the stack number of tokens is not updated, and the stack
            // will not be penalized for the request.
            update_stack_num_compute_units(
                &state.state_manager_sender,
                stack_small_id,
                estimated_total_compute_units,
                0,
                &endpoint,
            )?;
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
        path = CHAT_COMPLETIONS_PATH,
        stack_small_id,
        estimated_total_compute_units,
        payload_hash
    )
)]
#[allow(clippy::too_many_arguments)]
async fn handle_response(
    state: &AppState,
    endpoint: String,
    payload_hash: [u8; PAYLOAD_HASH_SIZE],
    stack_small_id: i64,
    is_stream: bool,
    payload: Value,
    estimated_total_compute_units: i64,
    client_encryption_metadata: Option<EncryptionMetadata>,
) -> Result<Response<Body>, AtomaServiceError> {
    if !is_stream {
        handle_non_streaming_response(
            state,
            payload,
            stack_small_id,
            estimated_total_compute_units,
            payload_hash,
            client_encryption_metadata,
            endpoint,
        )
        .await
    } else {
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
            estimated_total_compute_units,
            payload_hash,
            streaming_encryption_metadata,
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
        path = CHAT_COMPLETIONS_PATH,
        completion_type = "non-streaming",
        stack_small_id,
        estimated_total_compute_units,
        payload_hash
    )
)]
async fn handle_non_streaming_response(
    state: &AppState,
    payload: Value,
    stack_small_id: i64,
    estimated_total_compute_units: i64,
    payload_hash: [u8; PAYLOAD_HASH_SIZE],
    client_encryption_metadata: Option<EncryptionMetadata>,
    endpoint: String,
) -> Result<Response<Body>, AtomaServiceError> {
    // Record token metrics and extract the response total number of tokens
    let model = payload
        .get(MODEL_KEY)
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");
    let timer = CHAT_COMPLETIONS_LATENCY_METRICS
        .with_label_values(&[model])
        .start_timer();

    let response_body = utils::send_request_to_inference_service(
        state,
        &payload,
        stack_small_id,
        payload_hash,
        &endpoint,
    )
    .await?;

    let total_compute_units = utils::extract_total_num_tokens(&response_body, model);

    utils::serve_non_streaming_response(
        state,
        response_body,
        stack_small_id,
        estimated_total_compute_units,
        total_compute_units,
        payload_hash,
        client_encryption_metadata,
        endpoint,
        timer,
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
        path = CHAT_COMPLETIONS_PATH,
        completion_type = "streaming",
        stack_small_id,
        estimated_total_compute_units,
        payload_hash
    )
)]
async fn handle_streaming_response(
    state: &AppState,
    mut payload: Value,
    stack_small_id: i64,
    estimated_total_compute_units: i64,
    payload_hash: [u8; 32],
    streaming_encryption_metadata: Option<StreamingEncryptionMetadata>,
    endpoint: String,
) -> Result<Response<Body>, AtomaServiceError> {
    // NOTE: If streaming is requested, add the include_usage option to the payload
    // so that the atoma node state manager can be updated with the total number of tokens
    // that were processed for this request.
    payload["stream_options"] = json!({
        "include_usage": true
    });

    let model = payload
        .get(MODEL_KEY)
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");
    CHAT_COMPLETIONS_NUM_REQUESTS
        .with_label_values(&[model])
        .inc();
    let timer = CHAT_COMPLETIONS_TIME_TO_FIRST_TOKEN
        .with_label_values(&[model])
        .start_timer();

    let client = Client::new();
    let response = client
        .post(format!(
            "{}{}",
            state.chat_completions_service_url, CHAT_COMPLETIONS_PATH
        ))
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            AtomaServiceError::InternalError {
                message: format!(
                    "Error sending request to inference service, for request with payload hash: {:?}, and stack small id: {}, with error: {}",
                    payload_hash,
                    stack_small_id,
                    e
                ),
                endpoint: endpoint.clone(),
            }
        })?;

    if !response.status().is_success() {
        return Err(AtomaServiceError::InternalError {
            message: "Inference service returned error".to_string(),
            endpoint,
        });
    }

    let stream = response.bytes_stream();

    // Create the SSE stream
    let stream = Sse::new(Streamer::new(
        stream,
        state.state_manager_sender.clone(),
        stack_small_id,
        estimated_total_compute_units,
        payload_hash,
        state.keystore.clone(),
        state.address_index,
        model.to_string(),
        streaming_encryption_metadata,
        endpoint,
        timer,
    ))
    .keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_millis(STREAM_KEEP_ALIVE_INTERVAL_IN_SECONDS))
            .text("keep-alive"),
    );

    Ok(stream.into_response())
}

#[derive(Debug, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename(serialize = "requestBody", deserialize = "RequestBody"))]
pub struct ChatCompletionsRequest {
    /// A list of messages comprising the conversation so far.
    messages: Vec<Message>,
    /// ID of the model to use.
    model: String,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far,
    /// decreasing the model's likelihood to repeat the same line verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Modify the likelihood of specified tokens appearing in the completion.
    /// Accepts a JSON object that maps tokens (specified as their token ID in the tokenizer) to an associated bias value from -100 to 100.
    logit_bias: Option<HashMap<String, f32>>,
    /// Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    /// An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability.
    /// logprobs must be set to true if this parameter is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<i32>,
    /// An upper bound for the number of tokens that can be generated for a completion,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// How many chat completion choices to generate for each input message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    n: Option<usize>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far,
    /// increasing the model's likelihood to talk about new topics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    /// A seed to use for random number generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stop: Option<StopCondition>,
    /// If set, the server will stream the results as they come in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
    /// while lower values like 0.2 will make it more focused and deterministic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results
    /// of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    /// A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for. A max of 128 functions are supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    /// A unique identifier representing your end-user, which can help the system to monitor and detect abuse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

/// A message that is part of a conversation which is based on the role
/// of the author of the message.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    /// The role of the messages author, in this case system.
    System {
        /// The contents of the message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// The role of the messages author, in this case user.
    User {
        /// The contents of the message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// The role of the messages author, in this case assistant.
    Assistant {
        /// The contents of the message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// The refusal message by the assistant.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        /// The tool calls generated by the model, such as function calls.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    /// The role of the messages author, in this case tool.
    Tool {
        /// The contents of the message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// Tool call that this message is responding to.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        tool_call_id: String,
    },
}

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

impl std::fmt::Display for MessageContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageContent::Text(text) => write!(f, "{}", text),
            MessageContent::Array(parts) => {
                let mut content = String::new();
                for part in parts {
                    content.push_str(&format!("{}\n", part))
                }
                write!(f, "{}", content)
            }
        }
    }
}

// We manually implement Deserialize here for more control.
impl<'de> Deserialize<'de> for MessageContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Value = Value::deserialize(deserializer)?;

        if let Some(s) = value.as_str() {
            return Ok(MessageContent::Text(s.to_string()));
        }

        if let Some(arr) = value.as_array() {
            let parts: Result<Vec<MessageContentPart>, _> = arr
                .iter()
                .map(|v| serde_json::from_value(v.clone()).map_err(serde::de::Error::custom))
                .collect();
            return Ok(MessageContent::Array(parts?));
        }

        Err(serde::de::Error::custom(
            "Expected a string or an array of content parts",
        ))
    }
}

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
        image_url: MessageContentPartImageUrl,
    },
}

impl std::fmt::Display for MessageContentPart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageContentPart::Text { r#type, text } => {
                write!(f, "{}: {}", r#type, text)
            }
            MessageContentPart::Image { r#type, image_url } => {
                write!(f, "{}: [Image URL: {}]", r#type, image_url)
            }
        }
    }
}

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

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct ToolCallFunction {
    /// The name of the function to call.
    name: String,
    /// The arguments to call the function with, as generated by the model in JSON format.
    /// Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema.
    /// Validate the arguments in your code before calling your function.
    arguments: Value,
}

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

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename(serialize = "tool", deserialize = "tool"))]
pub struct Tool {
    /// The type of the tool. Currently, only function is supported.
    #[serde(rename(serialize = "type", deserialize = "type"))]
    r#type: String,
    /// The function that the model called.
    function: ToolFunction,
}

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

/// The stop condition for the chat completion.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename(serialize = "stop", deserialize = "stop"))]
#[serde(untagged)]
pub enum StopCondition {
    Array(Vec<String>),
    String(String),
}

/// Response structure returned by the chat completion API.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionsResponse {
    /// Unique identifier for the chat completion.
    pub id: String,
    /// The object type, typically "chat.completion".
    pub object: String,
    /// Unix timestamp (in seconds) of when the chat completion was created.
    pub created: u64,
    /// The model used for the chat completion.
    pub model: String,
    /// A unique identifier for the model's configuration and version.
    pub system_fingerprint: String,
    /// Array of chat completion choices. Can be multiple if n>1 was specified in the request.
    pub choices: Vec<Choice>,
    /// Statistics about token usage for this completion.
    pub usage: Usage,
}

/// Represents a single completion choice returned by the chat completion API.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct Choice {
    /// The index of this choice in the array of choices.
    pub index: u32,
    /// The message output by the model for this choice.
    pub message: Message,
    /// Log probabilities for the output tokens, if requested in the API call.
    /// Only present if `logprobs` was set to true in the request.
    pub logprobs: Option<Value>,
    /// Indicates why the model stopped generating tokens.
    /// Can be "stopped" (API returned complete model output),
    /// "length_capped" (maximum token limit reached), or
    /// "content_filter" (content was filtered).
    pub finish_reason: FinishReason,
}

/// Indicates why the model stopped generating tokens in a chat completion response.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model completed its response naturally or encountered a stop sequence
    Stopped,
    /// The model reached the maximum token limit specified in the request
    LengthCapped,
    /// The model's output was filtered due to content safety settings
    ContentFilter,
}

impl TryFrom<Option<&str>> for FinishReason {
    type Error = String;

    fn try_from(value: Option<&str>) -> Result<Self, Self::Error> {
        match value {
            Some("stopped") => Ok(FinishReason::Stopped),
            Some("length_capped") => Ok(FinishReason::LengthCapped),
            Some("content_filter") => Ok(FinishReason::ContentFilter),
            None => Ok(FinishReason::Stopped),
            _ => Err(format!("Invalid finish reason: {}", value.unwrap())),
        }
    }
}

/// Represents the token usage statistics for a chat completion request.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct Usage {
    /// The number of tokens used in the prompt/input.
    pub prompt_tokens: u32,

    /// The number of tokens used in the completion/output.
    /// NOTE: We allow for optional completions tokens, to be also compatible with the embeddings endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,

    /// The total number of tokens used (prompt_tokens + completion_tokens).
    pub total_tokens: u32,

    /// Additional details about completion tokens, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<Value>,
}

pub(crate) mod utils {
    use atoma_utils::constants::PAYLOAD_HASH_SIZE;
    use prometheus::HistogramTimer;

    use super::*;

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
        )
    )]
    pub(crate) async fn get_streaming_encryption_metadata(
        state: &AppState,
        client_encryption_metadata: Option<EncryptionMetadata>,
        payload_hash: [u8; PAYLOAD_HASH_SIZE],
        stack_small_id: i64,
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
                            "Error sending encryption request, for request with payload hash: {:?}, and stack small id: {}, with error: {}",
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
                        "Error receiving encryption response, for request with payload hash: {:?}, and stack small id: {}, with error: {}",
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
        fields(stack_small_id, payload_hash, endpoint)
    )]
    pub(crate) async fn send_request_to_inference_service(
        state: &AppState,
        payload: &Value,
        stack_small_id: i64,
        payload_hash: [u8; PAYLOAD_HASH_SIZE],
        endpoint: &str,
    ) -> Result<Value, AtomaServiceError> {
        let client = Client::new();
        let response = client
        .post(format!(
            "{}{}",
            state.chat_completions_service_url, CHAT_COMPLETIONS_PATH
        ))
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            AtomaServiceError::InternalError {
                message: format!(
                    "Error sending request to inference service, for request with payload hash: {:?}, and stack small id: {}, with error: {}",
                    payload_hash,
                    stack_small_id,
                    e
                ),
                endpoint: endpoint.to_string(),
            }
        })?;

        if !response.status().is_success() {
            return Err(AtomaServiceError::InternalError {
                message: format!(
                    "Inference service returned non-success status code: {}",
                    response.status()
                ),
                endpoint: endpoint.to_string(),
            });
        }

        response.json::<Value>().await.map_err(|e| {
            AtomaServiceError::InternalError {
                message: format!(
                    "Error reading response body, for request with payload hash: {:?}, and stack small id: {}, with error: {}",
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
    pub(crate) fn extract_total_num_tokens(response_body: &Value, model: &str) -> i64 {
        let mut total_compute_units = 0;
        if let Some(usage) = response_body.get("usage") {
            if let Some(prompt_tokens) = usage.get("prompt_tokens") {
                let prompt_tokens = prompt_tokens.as_u64().unwrap_or(0);
                CHAT_COMPLETIONS_INPUT_TOKENS_METRICS
                    .with_label_values(&[model])
                    .inc_by(prompt_tokens as f64);
                total_compute_units += prompt_tokens;
            }
            if let Some(completion_tokens) = usage.get("completion_tokens") {
                let completion_tokens = completion_tokens.as_u64().unwrap_or(0);
                CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS
                    .with_label_values(&[model])
                    .inc_by(completion_tokens as f64);
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
        fields(stack_small_id, estimated_total_compute_units, payload_hash, endpoint)
    )]
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn serve_non_streaming_response(
        state: &AppState,
        mut response_body: Value,
        stack_small_id: i64,
        estimated_total_compute_units: i64,
        total_compute_units: i64,
        payload_hash: [u8; PAYLOAD_HASH_SIZE],
        client_encryption_metadata: Option<EncryptionMetadata>,
        endpoint: String,
        timer: HistogramTimer,
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
                    "Error updating state manager, for request with payload hash: {:?}, and stack small id: {}, with error: {}",
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
            client_encryption_metadata,
            endpoint.clone(),
        )
        .await
        {
            Ok(response_body) => {
                // Stop the timer before returning the valid response
                timer.observe_duration();
                Json(response_body).into_response()
            }
            Err(e) => {
                return Err(AtomaServiceError::InternalError {
                    message: format!(
                        "Error handling confidential compute encryption response, for request with payload hash: {:?}, and stack small id: {}, with error: {}",
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
        update_stack_num_compute_units(
            &state.state_manager_sender,
            stack_small_id,
            estimated_total_compute_units,
            total_compute_units,
            &endpoint,
        )?;

        Ok(response_body)
    }
}
