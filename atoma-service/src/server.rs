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
use reqwest::Client;
use serde_json::{json, Value};
use sqlx::SqlitePool;
use sui_keys::keystore::FileBasedKeystore;
use tokenizers::Tokenizer;
use tokio::{net::TcpListener, signal, sync::watch::Sender};
use tower::ServiceBuilder;
use tracing::{error, info, instrument};

use crate::middleware::{signature_verification_middleware, verify_stack_permissions};

const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";
const HEALTH_PATH: &str = "/health";

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
        .route(HEALTH_PATH, get(health_check))
}

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
pub async fn health_check() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

#[instrument(
    level = "info",
    skip(state, payload),
    fields(path = CHAT_COMPLETIONS_PATH)
)]
pub async fn chat_completions_handler(
    Extension((stack_small_id, estimated_total_tokens)): Extension<(i64, i64)>,
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, StatusCode> {
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
    let signature = utils::sign_response_body(&response_body, &state.keystore, state.address_index)
        .map_err(|e| {
            error!("Error signing response body: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    response_body["signature"] = json!(signature);

    // Extract the response total number of tokens
    let total_tokens = response_body
        .get("usage")
        .and_then(|usage| usage.get("total_tokens"))
        .and_then(|total_tokens| total_tokens.as_u64())
        .map(|n| n as i64)
        .unwrap_or(0);

    let state_manager = StateManager::new(state.state.clone());

    state_manager
        .update_stack_num_tokens(stack_small_id, estimated_total_tokens, total_tokens)
        .await
        .map_err(|e| {
            error!("Error updating stack num tokens: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(Json(response_body))
}

pub(crate) mod utils {
    use super::*;
    use sha2::Digest;
    use sui_keys::keystore::AccountKeystore;
    use sui_sdk::types::crypto::EncodeDecodeBase64;

    pub(crate) fn sign_response_body(
        response_body: &Value,
        keystore: &Arc<FileBasedKeystore>,
        address_index: usize,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let address = keystore.addresses()[address_index];
        let response_body_str = response_body.to_string();
        let response_body_bytes = response_body_str.as_bytes();
        let sha256_hash = sha2::Sha256::digest(response_body_bytes);
        let signature = keystore
            .sign_hashed(&address, sha256_hash.as_slice())
            .unwrap();
        Ok(signature.encode_base64())
    }
}
