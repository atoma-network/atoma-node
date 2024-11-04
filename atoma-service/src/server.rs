use std::sync::Arc;

use atoma_state::types::AtomaAtomaStateManagerEvent;
use axum::{
    middleware::{from_fn, from_fn_with_state},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use blake2::{
    digest::generic_array::{typenum::U32, GenericArray},
    Digest,
};
use flume::Sender as FlumeSender;
use serde_json::{json, Value};
use sui_keys::keystore::FileBasedKeystore;
use tokenizers::Tokenizer;
use tokio::{net::TcpListener, signal, sync::watch::Sender};
use tower::ServiceBuilder;
use tracing::info;
use utoipa::OpenApi;

use crate::{
    components::openapi::openapi_routes,
    handlers::{
        chat_completions::{chat_completions_handler, CHAT_COMPLETIONS_PATH},
        embeddings::{embeddings_handler, EMBEDDINGS_PATH},
        image_generations::{image_generations_handler, IMAGE_GENERATIONS_PATH},
    },
    middleware::{signature_verification_middleware, verify_stack_permissions},
};

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

pub const HEALTH_PATH: &str = "/health";
#[derive(OpenApi)]
#[openapi(paths(health))]
pub(crate) struct HealthOpenApi;

#[utoipa::path(
    get,
    path = "",
    tag = "health",
    responses(
        (status = OK, description = "Service is healthy", body = Value)
    )
)]
pub async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
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
