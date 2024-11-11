use std::{sync::Arc, time::Duration};

use atoma_state::types::AtomaAtomaStateManagerEvent;
use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    middleware::{from_fn, from_fn_with_state},
    response::{IntoResponse, Response, Sse},
    routing::{get, post},
    Extension, Json, Router,
};
use blake2::{
    digest::generic_array::{typenum::U32, GenericArray},
    Digest,
};
use flume::Sender as FlumeSender;

use reqwest::Client;
use serde_json::{json, Value};
use sui_keys::keystore::FileBasedKeystore;
use tokenizers::Tokenizer;
use tokio::{net::TcpListener, signal, sync::watch::Sender};
use tower::ServiceBuilder;
use tracing::{error, info, instrument};
use utoipa::OpenApi;

use crate::{
    middleware::{signature_verification_middleware, verify_stack_permissions, RequestMetadata},
    streamer::Streamer,
    types::{ChatCompletionsRequest, ChatCompletionsResponse},
};

const STREAM_KEEP_ALIVE_INTERVAL_IN_SECONDS: u64 = 15;
const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";
const HEALTH_PATH: &str = "/health";

/// OpenAPI documentation for the chat completions endpoint
#[derive(OpenApi)]
#[openapi(
    paths(
        health,
        chat_completions_handler
    ),
    components(
        schemas(ChatCompletionsRequest, ChatCompletionsResponse)
    ),
    tags(
        (name = "health", description = "Health check endpoint"),
        (name = "chat-completions", description = "Chat completions endpoint")
    )
)]
pub struct OpenApiDoc;

/// Represents the shared state of the application.
///
/// This struct holds various components and configurations that are shared
/// across different parts of the application, enabling efficient resource
/// management and communication between components.
#[derive(Clone)]
pub struct AppState {
    /// Channel sender for managing application events.
    ///
    /// This sender is used to communicate events and state changes to the
    /// state manager, allowing for efficient handling of application state
    /// updates and notifications across different components.
    pub state_manager_sender: FlumeSender<AtomaAtomaStateManagerEvent>,

    /// Tokenizer used for processing text input.
    ///
    /// The tokenizer is responsible for breaking down text input into
    /// manageable tokens, which are then used in various natural language
    /// processing tasks.
    pub tokenizers: Arc<Vec<Arc<Tokenizer>>>,

    /// List of available AI models.
    ///
    /// This list contains the names or identifiers of AI models that
    /// the application can use for inference tasks. It allows the
    /// application to dynamically select and switch between different
    /// models as needed.
    pub models: Arc<Vec<String>>,

    /// URL of the inference service.
    ///
    /// This URL points to the external service responsible for performing
    /// AI model inference. The application forwards requests to this service
    /// to obtain AI-generated responses.
    pub inference_service_url: String,

    /// The Sui keystore of the node.
    ///
    /// The keystore contains cryptographic keys used for signing and
    /// verifying messages. It is essential for ensuring the security
    /// and integrity of communications within the application.
    pub keystore: Arc<FileBasedKeystore>,

    /// The Sui address index of the node within the keystore values.
    ///
    /// This index specifies which address in the keystore is used for
    /// signing operations, allowing the application to manage multiple
    /// addresses and keys efficiently.
    pub address_index: usize,
}

/// Creates and configures the main router for the application.
///
/// This function sets up the routes and middleware for the application,
/// including chat completions, health checks, and security layers.
///
/// # Arguments
///
/// * `app_state` - The shared application state containing database connections,
///   tokenizers, and model information.
///
/// # Returns
///
/// Returns a configured `Router` instance ready to be used by the server.
///
/// # Example
///
/// ```rust,ignore
/// let app_state = AppState::new(/* ... */);
/// let router = create_router(app_state);
/// // Use the router to start the server
/// ```
pub fn create_router(app_state: AppState) -> Router {
    Router::new()
        .route(CHAT_COMPLETIONS_PATH, post(chat_completions_handler))
        .layer(
            ServiceBuilder::new()
                .layer(from_fn(signature_verification_middleware))
                .layer(from_fn_with_state(
                    app_state.clone(),
                    verify_stack_permissions,
                ))
                .into_inner(),
        )
        .with_state(app_state)
        .route(HEALTH_PATH, get(health))
}

/// Starts and runs the HTTP server with graceful shutdown handling.
///
/// This function initializes and runs the main server instance with the provided configuration.
/// It sets up graceful shutdown handling through a Ctrl+C signal and communicates the shutdown
/// status through a channel.
///
/// # Arguments
///
/// * `app_state` - The shared application state containing database connections, tokenizers,
///   and other configuration.
/// * `tcp_listener` - A configured TCP listener that specifies the address and port for the server.
/// * `shutdown_sender` - A channel sender used to communicate the shutdown status to other parts
///   of the application.
///
/// # Returns
///
/// Returns a `Result` indicating whether the server started and shut down successfully.
///
/// # Errors
///
/// This function will return an error if:
/// - The server fails to start or encounters an error while running
/// - The shutdown signal fails to be sent through the channel
///
/// # Example
///
/// ```rust,ignore
/// let app_state = AppState::new(/* ... */);
/// let listener = TcpListener::bind("127.0.0.1:3000").await?;
/// let (shutdown_tx, shutdown_rx) = watch::channel(false);
///
/// run_server(app_state, listener, shutdown_tx).await?;
/// ```
pub async fn run_server(
    app_state: AppState,
    tcp_listener: TcpListener,
    shutdown_sender: Sender<bool>,
) -> anyhow::Result<()> {
    let app = create_router(app_state);
    let shutdown_signal = async {
        signal::ctrl_c()
            .await
            .expect("Failed to parse Ctrl+C signal");
        info!("Shutting down server...");
    };
    let server =
        axum::serve(tcp_listener, app.into_make_service()).with_graceful_shutdown(shutdown_signal);
    server.await?;

    shutdown_sender.send(true)?;

    Ok(())
}

/// Handles the health check endpoint.
///
/// This function is used to verify that the server is running and responsive.
/// It's typically used by load balancers or monitoring systems to check the
/// health status of the service.
///
/// # Returns
///
/// Returns a static string "OK" to indicate that the server is healthy and
/// functioning properly.
///
/// # Examples
///
/// This function is usually mapped to a GET endpoint, for example:
///
/// ```rust,ignore
/// app.route("/health", get(health_check))
/// ```
#[utoipa::path(
    get,
    path = "/health",
    tag = "health",
    responses(
        (status = 200, description = "Service is healthy", body = Value)
    )
)]
pub async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// Handles chat completion requests by forwarding them to the inference service and managing token usage.
///
/// This handler performs several key operations:
/// 1. Forwards the chat completion request to the inference service
/// 2. Signs the response using the node's keystore
/// 3. Tracks token usage for the stack
///
/// # Arguments
///
/// * `Extension((stack_small_id, estimated_total_tokens))` - Stack ID and estimated token count from middleware
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
    path = "/v1/chat/completions",
    tag = "chat",
    request_body = ChatCompletionsRequest,
    responses(
        (status = 200, description = "Chat completion successful", body = ChatCompletionsResponse, content_type = "application/json"),
        (status = 200, description = "Streaming chat completion", content_type = "text/event-stream"),
        (status = 500, description = "Internal server error")
    )
)]
#[instrument(
    level = "info",
    skip(state, payload),
    fields(path = CHAT_COMPLETIONS_PATH)
)]
pub async fn chat_completions_handler(
    Extension(request_metadata): Extension<RequestMetadata>,
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Response<Body>, StatusCode> {
    let RequestMetadata {
        stack_small_id,
        estimated_total_tokens,
        payload_hash,
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
            estimated_total_tokens,
            payload_hash,
        )
        .await
    } else {
        handle_streaming_response(
            state,
            payload,
            stack_small_id,
            estimated_total_tokens,
            payload_hash,
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
/// * `estimated_total_tokens` - Estimated token count for the request
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
        estimated_total_tokens,
        payload_hash
    )
)]
async fn handle_non_streaming_response(
    state: AppState,
    payload: Value,
    stack_small_id: i64,
    estimated_total_tokens: i64,
    payload_hash: [u8; 32],
) -> Result<Response<Body>, StatusCode> {
    let client = Client::new();
    let response = client
        .post(format!(
            "{}{}",
            state.inference_service_url, CHAT_COMPLETIONS_PATH
        ))
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            error!("Error sending request to inference service: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let mut response_body = response.json::<Value>().await.map_err(|e| {
        error!("Error reading response body: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Sign the response body byte content and add the base64 encoded signature to the response body
    let (response_hash, signature) =
        utils::sign_response_body(&response_body, &state.keystore, state.address_index).map_err(
            |e| {
                error!("Error signing response body: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            },
        )?;
    response_body["signature"] = json!(signature);

    // Extract the response total number of tokens
    let total_tokens = response_body
        .get("usage")
        .and_then(|usage| usage.get("total_tokens"))
        .and_then(|total_tokens| total_tokens.as_u64())
        .map(|n| n as i64)
        .unwrap_or(0);

    // NOTE: We need to update the stack num tokens, because the inference response might have produced
    // less tokens than estimated what we initially estimated, from the middleware.
    if let Err(e) = utils::update_state_manager(
        &state,
        stack_small_id,
        estimated_total_tokens,
        total_tokens,
        payload_hash,
        response_hash,
    )
    .await
    {
        error!(
            "Error updating state manager: {}, for request with payload hash: {payload_hash:?}",
            e
        );
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    Ok(Json(response_body).into_response())
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
/// * `estimated_total_tokens` - Estimated token count for the request
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
        estimated_total_tokens,
        payload_hash
    )
)]
async fn handle_streaming_response(
    state: AppState,
    mut payload: Value,
    stack_small_id: i64,
    estimated_total_tokens: i64,
    payload_hash: [u8; 32],
) -> Result<Response<Body>, StatusCode> {
    // NOTE: If streaming is requested, add the include_usage option to the payload
    // so that the atoma node state manager can be updated with the total number of tokens
    // that were processed for this request.
    payload["stream_options"] = json!({
        "include_usage": true
    });

    let client = Client::new();
    let response = client
        .post(format!(
            "{}{}",
            state.inference_service_url, CHAT_COMPLETIONS_PATH
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
        estimated_total_tokens,
        payload_hash,
        state.keystore.clone(),
        state.address_index,
    ))
    .keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_millis(STREAM_KEEP_ALIVE_INTERVAL_IN_SECONDS))
            .text("keep-alive"),
    );

    Ok(stream.into_response())
}

pub(crate) mod utils {
    use super::*;

    use sui_keys::keystore::AccountKeystore;
    use sui_sdk::types::crypto::EncodeDecodeBase64;

    /// Signs a JSON response body using the node's Sui keystore.
    ///
    /// This function takes a JSON response body, converts it to bytes, creates a SHA-256 hash,
    /// and signs it using the Sui keystore with the specified address.
    ///
    /// # Arguments
    ///
    /// * `response_body` - The JSON response body to be signed
    /// * `keystore` - The Sui keystore containing the signing keys
    /// * `address_index` - The index of the address to use for signing within the keystore
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// * A 32-byte array containing the SHA-256 hash of the response body
    /// * A base64-encoded string of the signature
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * The keystore fails to sign the hash
    /// * The SHA-256 hash cannot be converted to a 32-byte array
    pub(crate) fn sign_response_body(
        response_body: &Value,
        keystore: &Arc<FileBasedKeystore>,
        address_index: usize,
    ) -> Result<([u8; 32], String), Box<dyn std::error::Error>> {
        let address = keystore.addresses()[address_index];
        let response_body_str = response_body.to_string();
        let response_body_bytes = response_body_str.as_bytes();
        let mut blake2b = blake2::Blake2b::new();
        blake2b.update(response_body_bytes);
        let blake2b_hash: GenericArray<u8, U32> = blake2b.finalize();
        let signature = keystore
            .sign_hashed(&address, blake2b_hash.as_slice())
            .expect("Failed to sign response body");
        Ok((
            blake2b_hash
                .as_slice()
                .try_into()
                .expect("Invalid BLAKE2b hash length"),
            signature.encode_base64(),
        ))
    }

    /// Updates the state manager with token usage and hash information for a stack.
    ///
    /// This function performs two main operations:
    /// 1. Updates the token count for the stack with both estimated and actual usage
    /// 2. Computes and updates a total hash combining the payload and response hashes
    ///
    /// # Arguments
    ///
    /// * `state` - Reference to the application state containing the state manager sender
    /// * `stack_small_id` - Unique identifier for the stack
    /// * `estimated_total_tokens` - The estimated number of tokens before processing
    /// * `total_tokens` - The actual number of tokens used
    /// * `payload_hash` - Hash of the request payload
    /// * `response_hash` - Hash of the response data
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if both updates succeed, or a `StatusCode::INTERNAL_SERVER_ERROR` if either update fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The state manager channel is closed
    /// - Either update operation fails to complete
    pub(crate) async fn update_state_manager(
        state: &AppState,
        stack_small_id: i64,
        estimated_total_tokens: i64,
        total_tokens: i64,
        payload_hash: [u8; 32],
        response_hash: [u8; 32],
    ) -> Result<(), StatusCode> {
        // Update stack num tokens
        state
            .state_manager_sender
            .send(AtomaAtomaStateManagerEvent::UpdateStackNumTokens {
                stack_small_id,
                estimated_total_tokens,
                total_tokens,
            })
            .map_err(|e| {
                error!("Error updating stack num tokens: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

        // Compute total hash
        let mut blake2b = blake2::Blake2b::new();
        blake2b.update([payload_hash, response_hash].concat());
        let total_hash: GenericArray<u8, U32> = blake2b.finalize();
        let total_hash_bytes: [u8; 32] = total_hash.into();

        // Update stack total hash
        state
            .state_manager_sender
            .send(AtomaAtomaStateManagerEvent::UpdateStackTotalHash {
                stack_small_id,
                total_hash: total_hash_bytes,
            })
            .map_err(|e| {
                error!("Error updating stack total hash: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

        Ok(())
    }
}
