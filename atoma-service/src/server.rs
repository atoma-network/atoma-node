use std::sync::Arc;

use atoma_state::StateManager;
use axum::{
    extract::State,
    http::StatusCode,
    middleware::{from_fn, from_fn_with_state},
    response::IntoResponse,
    routing::{get, post},
    Extension, Json, Router,
};
use blake2::{digest::generic_array::GenericArray, Digest};
use p256::U32;
use reqwest::Client;
use serde_json::{json, Value};
use sqlx::SqlitePool;
use sui_keys::keystore::FileBasedKeystore;
use tokenizers::Tokenizer;
use tokio::{net::TcpListener, signal, sync::watch::Sender};
use tower::ServiceBuilder;
use tracing::{error, info, instrument};
use utoipa::OpenApi;

use crate::{
    middleware::{signature_verification_middleware, verify_stack_permissions, RequestMetadata},
    types::{ChatCompletionsRequest, ChatCompletionsResponse},
};

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
#[derive(Clone)]
pub struct AppState {
    /// SQLite database connection pool.
    pub state: SqlitePool,

    /// Tokenizer used for processing text input.
    pub tokenizer: Arc<Tokenizer>,

    /// List of available AI models.
    pub models: Arc<Vec<String>>,

    /// URL of the node's inference service.
    pub inference_service_client: Client,

    /// The Sui keystore of the node
    pub keystore: Arc<FileBasedKeystore>,

    /// The Sui address index of the node within the keystore values
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
/// ```
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
/// ```
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
) -> Result<(), Box<dyn std::error::Error>> {
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
/// ```
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
        (status = 200, description = "Chat completion successful", body = ChatCompletionsResponse),
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
) -> Result<Json<Value>, StatusCode> {
    let RequestMetadata {
        stack_small_id,
        estimated_total_tokens,
        payload_hash,
    } = request_metadata;
    let response = state
        .inference_service_client
        .post(CHAT_COMPLETIONS_PATH)
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

    let state_manager = StateManager::new(state.state.clone());

    // NOTE: We need to update the stack num tokens, because the inference response might have produced
    // less tokens than estimated what we initially estimated, from the middleware.
    state_manager
        .update_stack_num_tokens(stack_small_id, estimated_total_tokens, total_tokens)
        .await
        .map_err(|e| {
            error!("Error updating stack num tokens: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let mut blake2b = blake2::Blake2b::new();
    blake2b.update([payload_hash, response_hash].concat());
    let total_hash: GenericArray<u8, U32> = blake2b.finalize();
    let total_hash_bytes: [u8; 32] = total_hash
        .as_slice()
        .try_into()
        .expect("Invalid BLAKE2b hash length");
    state_manager
        .update_stack_total_hash(stack_small_id, total_hash_bytes)
        .await
        .map_err(|e| {
            error!("Error updating stack total hash: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok(Json(response_body))
}

pub(crate) mod utils {
    use super::*;
    use blake2::{digest::generic_array::GenericArray, Digest};
    use p256::U32;
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
}
