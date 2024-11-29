use std::sync::Arc;

use atoma_confidential::types::{
    ConfidentialComputeDecryptionRequest, ConfidentialComputeDecryptionResponse,
    ConfidentialComputeEncryptionRequest, ConfidentialComputeEncryptionResponse,
};
use atoma_state::types::AtomaAtomaStateManagerEvent;
use axum::{
    body::Body,
    middleware::{from_fn, from_fn_with_state},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use flume::Sender as FlumeSender;
use hyper::StatusCode;
use prometheus::Encoder;
use serde_json::{json, Value};
use sui_keys::keystore::FileBasedKeystore;
use sui_sdk::types::digests::TransactionDigest;
use tokenizers::Tokenizer;
use tokio::{
    net::TcpListener,
    sync::{
        mpsc::{self, UnboundedSender},
        oneshot,
        watch::Receiver,
    },
};
use tower::ServiceBuilder;
use tracing::error;
use utoipa::OpenApi;

use crate::{
    components::openapi::openapi_routes,
    handlers::{
        chat_completions::{
            chat_completions_handler, CHAT_COMPLETIONS_PATH, CONFIDENTIAL_CHAT_COMPLETIONS_PATH,
        },
        embeddings::{embeddings_handler, CONFIDENTIAL_EMBEDDINGS_PATH, EMBEDDINGS_PATH},
        image_generations::{
            image_generations_handler, CONFIDENTIAL_IMAGE_GENERATIONS_PATH, IMAGE_GENERATIONS_PATH,
        },
    },
    middleware::{
        confidential_compute_middleware, signature_verification_middleware,
        verify_stack_permissions,
    },
};

/// The path for the health check endpoint.
pub const HEALTH_PATH: &str = "/health";

/// The path for the metrics endpoint.
pub const METRICS_PATH: &str = "/metrics";

/// A small identifier for a Stack, represented as a 64-bit unsigned integer.
type StackSmallId = u64;

/// Represents the number of compute units available, stored as a 64-bit unsigned integer.
type ComputeUnits = u64;

/// Represents the result of a blockchain query for stack information.
type StackQueryResult = (Option<StackSmallId>, Option<ComputeUnits>);

/// Represents a request for confidential compute decryption.
type DecryptionRequest = (
    ConfidentialComputeDecryptionRequest,
    oneshot::Sender<ConfidentialComputeDecryptionResponse>,
);

type EncryptionRequest = (
    ConfidentialComputeEncryptionRequest,
    oneshot::Sender<ConfidentialComputeEncryptionResponse>,
);

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

    /// Channel sender for confidential compute decryption requests.
    ///
    /// This sender is used to communicate decryption requests to the
    /// confidential compute service, allowing for efficient handling of
    /// confidential data processing across different components.
    pub decryption_sender: UnboundedSender<DecryptionRequest>,

    /// Channel sender for confidential compute encryption requests.
    ///
    /// This sender is used to communicate encryption requests to the
    /// confidential compute service, allowing for efficient handling of
    /// confidential data processing across different components.
    pub encryption_sender: UnboundedSender<EncryptionRequest>,

    /// Channel sender for requesting compute units from the blockchain.
    pub stack_retrieve_sender:
        mpsc::UnboundedSender<(TransactionDigest, i64, oneshot::Sender<StackQueryResult>)>,

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
    let confidential_routes = Router::new()
        .route(
            CONFIDENTIAL_CHAT_COMPLETIONS_PATH,
            post(chat_completions_handler),
        )
        .route(CONFIDENTIAL_EMBEDDINGS_PATH, post(embeddings_handler))
        .route(
            CONFIDENTIAL_IMAGE_GENERATIONS_PATH,
            post(image_generations_handler),
        )
        .layer(
            ServiceBuilder::new()
                .layer(from_fn(signature_verification_middleware))
                .layer(from_fn_with_state(
                    app_state.clone(),
                    confidential_compute_middleware,
                ))
                .layer(from_fn_with_state(
                    app_state.clone(),
                    verify_stack_permissions,
                ))
                .into_inner(),
        )
        .with_state(app_state.clone());
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
        .merge(confidential_routes)
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

/// OpenAPI documentation for the metrics endpoint.
///
/// This struct is used to generate OpenAPI/Swagger documentation for the metrics
/// endpoint of the service. It provides a standardized way to describe the API
/// endpoint that exposes service metrics in a format compatible with Prometheus
/// and other monitoring systems.
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
    let metric_families = prometheus::gather();
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

    use atoma_utils::hashing::blake2b_hash;
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
        let blake2b_hash = blake2b_hash(response_body_bytes);
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
