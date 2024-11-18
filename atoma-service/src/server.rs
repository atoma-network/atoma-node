use std::sync::Arc;

use atoma_state::types::AtomaAtomaStateManagerEvent;
use axum::{
    body::Body,
    middleware::{from_fn, from_fn_with_state},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use blake2::{
    digest::generic_array::{typenum::U32, GenericArray},
    Digest,
};
use flume::Sender as FlumeSender;
use hyper::StatusCode;
use lazy_static::lazy_static;
use prometheus::{Encoder, Registry};
use serde_json::{json, Value};
use sui_keys::keystore::FileBasedKeystore;
use tokenizers::Tokenizer;
use tokio::{net::TcpListener, sync::watch::Receiver};
use tower::ServiceBuilder;
use tracing::error;
use utoipa::OpenApi;

use crate::{
    components::openapi::openapi_routes,
    handlers::{
        chat_completions::{chat_completions_handler, CHAT_COMPLETIONS_PATH},
        embeddings::{embeddings_handler, EMBEDDINGS_PATH},
        image_generations::{image_generations_handler, IMAGE_GENERATIONS_PATH},
        prometheus::{
            CHAT_COMPLETIONS_DECODING_TIME, CHAT_COMPLETIONS_INPUT_TOKENS_METRICS,
            CHAT_COMPLETIONS_LATENCY_METRICS, CHAT_COMPLETIONS_NUM_REQUESTS,
            CHAT_COMPLETIONS_TIME_TO_FIRST_TOKEN, IMAGE_GEN_LATENCY_METRICS,
            IMAGE_GEN_NUM_REQUESTS, TEXT_EMBEDDINGS_LATENCY_METRICS, TEXT_EMBEDDINGS_NUM_REQUESTS,
        },
    },
    middleware::{signature_verification_middleware, verify_stack_permissions},
};

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();
}

/// The path for the health check endpoint.
pub const HEALTH_PATH: &str = "/health";

/// The path for the metrics endpoint.
pub const METRICS_PATH: &str = "/metrics";

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

    /// URL of the chat completions service.
    ///
    /// This URL points to the external service responsible for performing
    /// AI model chat completions. The application forwards requests to this
    /// service to obtain AI-generated responses.
    pub chat_completions_service_url: String,

    /// URL for the embeddings service.
    ///
    /// This is an optional field that, if provided, specifies the endpoint
    /// for the embeddings service used by the Atoma Service.
    pub embeddings_service_url: String,

    /// URL for the image generations service.
    ///
    /// This is an optional field that, if provided, specifies the endpoint
    /// for the image generations service used by the Atoma Service.
    pub image_generations_service_url: String,

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
        .route(EMBEDDINGS_PATH, post(embeddings_handler))
        .route(IMAGE_GENERATIONS_PATH, post(image_generations_handler))
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
        .route(METRICS_PATH, get(metrics_handler))
        .merge(openapi_routes())
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
    mut shutdown_receiver: Receiver<bool>,
) -> anyhow::Result<()> {
    register_metrics();
    let app = create_router(app_state);
    let server =
        axum::serve(tcp_listener, app.into_make_service()).with_graceful_shutdown(async move {
            shutdown_receiver
                .changed()
                .await
                .expect("Error receiving shutdown signal")
        });
    server.await?;

    Ok(())
}

/// Registers Prometheus metrics for monitoring chat completion performance.
///
/// This function registers several metrics that track different aspects of chat completion
/// performance in the global Prometheus registry:
///
/// * Chat completions latency metrics - Overall response time for chat completion requests
/// * Chat completions time to first token - How quickly the first response token is generated
/// * Chat completions input tokens metrics - Number of tokens in the input prompts
/// * Chat completions decoding time metrics - Time spent decoding the model outputs
/// * Chat completions number of requests - Total number of received chat completions requests, so far
/// * Text embeddings latency metrics - Overall response time for image generation requests
/// * Text embeddings number of requests - Total number of received text embeddings requests, so far
/// * Image generation latency metrics - Overall response time for text embeddings requests
/// * Image generation number of requests - Total number of received image generation requests, so far
///
/// # Panics
///
/// This function will panic if any metric registration fails, as these metrics are
/// essential for monitoring system performance.
///
/// # Example
///
/// ```rust,ignore
/// register_metrics();
/// // Metrics are now available for collection at the /metrics endpoint
/// ```
pub fn register_metrics() {
    REGISTRY
        .register(Box::new(CHAT_COMPLETIONS_LATENCY_METRICS.clone()))
        .expect("Failed to register chat completions latency metrics");
    REGISTRY
        .register(Box::new(CHAT_COMPLETIONS_TIME_TO_FIRST_TOKEN.clone()))
        .expect("Failed to register chat completions time to first token metrics");
    REGISTRY
        .register(Box::new(CHAT_COMPLETIONS_INPUT_TOKENS_METRICS.clone()))
        .expect("Failed to register chat completions input tokens metrics");
    REGISTRY
        .register(Box::new(CHAT_COMPLETIONS_DECODING_TIME.clone()))
        .expect("Failed to register chat completions decoding time metrics");
    REGISTRY
        .register(Box::new(CHAT_COMPLETIONS_NUM_REQUESTS.clone()))
        .expect("Failed to register chat completions number of requests metrics");
    REGISTRY
        .register(Box::new(TEXT_EMBEDDINGS_LATENCY_METRICS.clone()))
        .expect("Failed to register text embeddings metrics");
    REGISTRY
        .register(Box::new(TEXT_EMBEDDINGS_NUM_REQUESTS.clone()))
        .expect("Failed to register text embeddings number of requests metrics");
    REGISTRY
        .register(Box::new(IMAGE_GEN_LATENCY_METRICS.clone()))
        .expect("Failed to register image generation metrics");
    REGISTRY
        .register(Box::new(IMAGE_GEN_NUM_REQUESTS.clone()))
        .expect("Failed to register image generation number of requests metrics");
}

#[derive(OpenApi)]
#[openapi(paths(health))]
pub(crate) struct HealthOpenApi;

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
    path = "",
    tag = "health",
    responses(
        (status = OK, description = "Service is healthy", body = Value)
    )
)]
async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

#[derive(OpenApi)]
#[openapi(paths(metrics_handler))]
pub(crate) struct MetricsOpenApi;

/// Handles the metrics endpoint.
///
/// This function is used to return the metrics for the service.
///
/// # Returns
///
/// Returns the metrics for the service as a plain text response.
#[utoipa::path(
    get,
    path = "",
    tag = "metrics",
    responses(
        (status = OK, description = "Metrics for the service")
    )
)]
async fn metrics_handler() -> Result<impl IntoResponse, StatusCode> {
    let encoder = prometheus::TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = vec![];
    encoder.encode(&metric_families, &mut buffer).map_err(|e| {
        error!("Failed to encode metrics: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", encoder.format_type())
        .body(Body::from(buffer))
        .map_err(|e| {
            error!("Failed to build response: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })
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
}
