use crate::{
    handlers::sign_response_and_update_stack_hash,
    middleware::EncryptionMetadata,
    server::AppState,
    streamer::{Streamer, StreamingEncryptionMetadata},
};
use atoma_confidential::types::{
    ConfidentialComputeSharedSecretRequest, ConfidentialComputeSharedSecretResponse,
};
use atoma_state::types::AtomaAtomaStateManagerEvent;
use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, Sse},
    Extension, Json,
};
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{error, info, instrument};
use utoipa::OpenApi;

use serde::{Deserialize, Deserializer, Serialize};
use std::{collections::HashMap, time::Duration};
use utoipa::ToSchema;

use crate::{handlers::prometheus::*, middleware::RequestMetadata};

use super::handle_confidential_compute_encryption_response;

/// The path for confidential chat completions requests
pub const CONFIDENTIAL_CHAT_COMPLETIONS_PATH: &str = "/v1/confidential/chat/completions";

/// The path for chat completions requests
pub const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

/// The keep-alive interval in seconds
const STREAM_KEEP_ALIVE_INTERVAL_IN_SECONDS: u64 = 15;

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
/// Returns a `StatusCode::INTERNAL_SERVER_ERROR` if:
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
) -> Result<Response<Body>, StatusCode> {
    let RequestMetadata {
        stack_small_id,
        estimated_total_compute_units,
        payload_hash,
        client_encryption_metadata,
        ..
    } = request_metadata;
    info!("Received chat completions request, with payload hash: {payload_hash:?}");

    // Check if streaming is requested
    let is_stream = payload
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or_default();

    if !is_stream {
        handle_non_streaming_response(
            state,
            payload,
            stack_small_id,
            estimated_total_compute_units,
            payload_hash,
            client_encryption_metadata,
        )
        .await
    } else {
        let streaming_encryption_metadata = if let Some(client_encryption_metadata) =
            client_encryption_metadata
        {
            let (sender, receiver) = tokio::sync::oneshot::channel();
            state
                .compute_shared_secret_sender
                .send((
                    ConfidentialComputeSharedSecretRequest {
                        proxy_x25519_public_key: client_encryption_metadata.proxy_x25519_public_key,
                    },
                    sender,
                ))
                .map_err(|e| {
                    error!("Error sending encryption request: {}", e);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;
            let ConfidentialComputeSharedSecretResponse {
                shared_secret,
                nonce,
            } = receiver.await.map_err(|e| {
                error!("Error receiving encryption response: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
            Some(StreamingEncryptionMetadata {
                shared_secret,
                nonce,
                salt: client_encryption_metadata.salt,
            })
        } else {
            None
        };

        handle_streaming_response(
            state,
            payload,
            stack_small_id,
            estimated_total_compute_units,
            payload_hash,
            streaming_encryption_metadata,
        )
        .await
    }
}

/// Handles non-streaming chat completion requests by processing them through the inference service.
///
/// This function performs several key operations:
/// 1. Forwards the request to the inference service
/// 2. Processes and signs the response
/// 3. Updates token usage tracking
/// 4. Updates the stack's total hash
///
/// # Arguments
///
/// * `state` - Application state containing service configuration and keystore
/// * `payload` - The JSON payload containing the chat completion request
/// * `stack_small_id` - Unique identifier for the stack making the request
/// * `estimated_total_compute_units` - Estimated compute units count for the request
/// * `payload_hash` - BLAKE2b hash of the original request payload
///
/// # Returns
///
/// Returns a `Result` containing the JSON response with added signature, or a `StatusCode` error.
///
/// # Errors
///
/// Returns `StatusCode::INTERNAL_SERVER_ERROR` if:
/// - The inference service request fails
/// - Response parsing fails
/// - Response signing fails
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
    state: AppState,
    payload: Value,
    stack_small_id: i64,
    estimated_total_compute_units: i64,
    payload_hash: [u8; 32],
    client_encryption_metadata: Option<EncryptionMetadata>,
) -> Result<Response<Body>, StatusCode> {
    // Record token metrics and extract the response total number of tokens
    let model = payload
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");
    let timer = CHAT_COMPLETIONS_LATENCY_METRICS
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
            error!(
                target = "atoma-service",
                event = "chat-completions-handler",
                "Error sending request to inference service: {}",
                e
            );
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    let mut response_body = response.json::<Value>().await.map_err(|e| {
        error!(
            target = "atoma-service",
            event = "chat-completions-handler",
            "Error reading response body: {}",
            e
        );
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

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

    // Update stack num tokens
    state
        .state_manager_sender
        .send(AtomaAtomaStateManagerEvent::UpdateStackNumComputeUnits {
            stack_small_id,
            estimated_total_compute_units,
            total_compute_units: total_compute_units as i64,
        })
        .map_err(|e| {
            error!(
                target = "atoma-service",
                event = "chat-completions-handler",
                "Error updating stack num tokens: {}",
                e
            );
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // NOTE: We need to update the stack num tokens, because the inference response might have produced
    // less tokens than estimated what we initially estimated, from the middleware.
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
            event = "chat-completions-handler",
            "Error updating state manager: {}, for request with payload hash: {payload_hash:?}",
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
            Ok(Json(response_body).into_response())
        }
        Err(e) => {
            error!(
                target = "atoma-service",
                event = "chat-completions-handler",
                "Error handling confidential compute encryption response: {}",
                e
            );
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
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
///
/// # Returns
///
/// Returns a `Result` containing an SSE stream response, or a `StatusCode` error.
///
/// # Errors
///
/// Returns `StatusCode::INTERNAL_SERVER_ERROR` if:
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
    state: AppState,
    mut payload: Value,
    stack_small_id: i64,
    estimated_total_compute_units: i64,
    payload_hash: [u8; 32],
    streaming_encryption_metadata: Option<StreamingEncryptionMetadata>,
) -> Result<Response<Body>, StatusCode> {
    // NOTE: If streaming is requested, add the include_usage option to the payload
    // so that the atoma node state manager can be updated with the total number of tokens
    // that were processed for this request.
    payload["stream_options"] = json!({
        "include_usage": true
    });

    let model = payload
        .get("model")
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
            error!("Error sending request to inference service: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if !response.status().is_success() {
        error!("Inference service returned error: {}", response.status());
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
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
    pub completion_tokens: u32,
    /// The total number of tokens used (prompt_tokens + completion_tokens).
    pub total_tokens: u32,
}
