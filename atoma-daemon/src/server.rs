use atoma_state::state_manager::AtomaState;
use atoma_sui::client::AtomaSuiClient;
use axum::{http::StatusCode, routing::get, Router};
use std::sync::Arc;
use sui_sdk::types::base_types::ObjectID;
use tokio::{
    net::TcpListener,
    sync::{watch::Receiver, RwLock},
};
use tracing::instrument;

use crate::{
    components::openapi::openapi_routes,
    handlers::{
        attestation_disputes::attestation_disputes_router, claimed_stacks::claimed_stacks_router,
        nodes::nodes_router, stacks::stacks_router, subscriptions::subscriptions_router,
        tasks::tasks_router,
    },
};

/// State container for the Atoma daemon service that manages node operations and interactions.
///
/// The `DaemonState` struct serves as the central state management component for the Atoma daemon,
/// containing essential components for interacting with the Sui blockchain and managing node state.
/// It is designed to be shared across multiple request handlers and maintains thread-safe access
/// to shared resources.
///
/// # Thread Safety
///
/// This struct is designed to be safely shared across multiple threads:
/// - Implements `Clone` for easy sharing across request handlers
/// - Uses `Arc<RwLock>` for thread-safe access to the Sui client
/// - State manager and node badges vector use interior mutability patterns
///
/// # Example
///
/// ```rust,ignore
/// // Create a new daemon state instance
/// let daemon_state = DaemonState {
///     client: Arc::new(RwLock::new(AtomaSuiClient::new())),
///     state_manager: AtomaStateManager::new(),
///     node_badges: vec![(ObjectID::new([0; 32]), 1)],
/// };
///
/// // Clone the state for use in different handlers
/// let handler_state = daemon_state.clone();
/// ```
#[derive(Clone)]
pub struct DaemonState {
    /// Thread-safe reference to the Sui blockchain client that handles all blockchain interactions.
    /// Wrapped in `Arc<RwLock>` to allow multiple handlers to safely access and modify the client
    /// state concurrently.
    pub client: Arc<RwLock<AtomaSuiClient>>,

    /// Manages the persistent state of nodes, tasks, and other system components.
    /// Handles database operations and state synchronization.
    pub atoma_state: AtomaState,

    /// Vector of tuples containing node badge information, where each tuple contains:
    /// - `ObjectID`: The unique identifier of the node badge on the Sui blockchain
    /// - `u64`: The small ID associated with the node badge for efficient indexing
    pub node_badges: Vec<(ObjectID, u64)>,
}

/// Starts and runs the Atoma daemon service, handling HTTP requests and graceful shutdown.
/// This function initializes and runs the main daemon service that handles node operations,
///
/// # Arguments
///
/// * `daemon_state` - The shared state container for the daemon service, containing the Sui client,
///   state manager, and node badge information
/// * `tcp_listener` - A pre-configured TCP listener that the HTTP server will bind to
///
/// # Returns
///
/// * `anyhow::Result<()>` - Ok(()) on successful shutdown, or an error if
///   server initialization or shutdown fails
///
/// # Shutdown Behavior
///
/// The server implements graceful shutdown by:
/// 1. Listening for a Ctrl+C signal
/// 2. Logging shutdown initiation
/// 3. Waiting for existing connections to complete
///
/// # Example
///
/// ```rust,ignore
/// use tokio::net::TcpListener;
/// use tokio::sync::watch;
/// use atoma_daemon::{DaemonState, run_server};
///
/// async fn start_server() -> Result<(), Box<dyn std::error::Error>> {
///     let daemon_state = DaemonState::new(/* ... */);
///     let listener = TcpListener::bind("127.0.0.1:3000").await?;
///     
///     run_server(daemon_state, listener).await
/// }
/// ```
pub async fn run_server(
    daemon_state: DaemonState,
    tcp_listener: TcpListener,
    mut shutdown_receiver: Receiver<bool>,
) -> anyhow::Result<()> {
    let daemon_router = create_router(daemon_state);
    let server = axum::serve(tcp_listener, daemon_router.into_make_service())
        .with_graceful_shutdown(async move {
            shutdown_receiver
                .changed()
                .await
                .expect("Error receiving shutdown signal")
        });
    server.await?;
    Ok(())
}

/// Creates and configures the main router for the Atoma daemon HTTP API.
///
/// # Arguments
/// * `daemon_state` - The shared state container that will be available to all route handlers
///
/// # Returns
/// * `Router` - A configured axum Router instance with all API routes and shared state
///
/// # API Endpoints
///
/// ## Health Check
/// * `GET /health` - Check service health status
///
/// ## Subscription Management
/// * `GET /subscriptions` - Get all subscriptions for registered nodes
/// * `GET /subscriptions/:id` - Get subscriptions for a specific node
/// * `POST /nodes/model-subscribe` - Subscribe a node to a model
/// * `POST /nodes/task-subscribe` - Subscribe a node to a task
/// * `POST /nodes/task-update-subscription` - Updates an existing task subscription
/// * `POST /nodes/task-unsubscribe` - Unsubscribe a node from a task
///
/// ## Task Management
/// * `GET /tasks` - Get all available tasks
///
/// ## Stack Operations
/// * `GET /stacks` - Get all stacks for registered nodes
/// * `GET /stacks/:id` - Get stacks for a specific node
/// * `GET /almost_filled_stacks/:fraction` - Get stacks filled above specified fraction
/// * `GET /almost_filled_stacks/:id/:fraction` - Get node's stacks filled above fraction
/// * `GET /stacks/claimed_stacks` - Get all claimed stacks
/// * `GET /stacks/claimed_stacks/:id` - Get claimed stacks for a specific node
/// * `POST /nodes/try-settle-stacks` - Attempt to settle specified stacks
/// * `POST /submit_stack_settlement_attestations` - Submit attestations for stack settlement
/// * `POST /claim_funds` - Claim funds from completed stacks
///
/// ## Attestation Disputes
/// * `GET /attestation_disputes/against` - Get disputes against registered nodes
/// * `GET /attestation_disputes/against/:id` - Get disputes against a specific node
/// * `GET /attestation_disputes/own` - Get disputes initiated by registered nodes
/// * `GET /attestation_disputes/own/:id` - Get disputes initiated by a specific node
///
/// ## Node Registration
/// * `POST /nodes/register` - Register a new node
///
/// ## API Documentation
/// * `GET /swagger-ui` - Interactive API documentation UI
/// * `GET /api-docs/openapi.json` - OpenAPI specification in JSON format
///
/// # Example
/// ```rust,ignore
/// use atoma_daemon::DaemonState;
///
/// let daemon_state = DaemonState::new(/* ... */);
/// let app = create_router(daemon_state);
///
/// // Start the server with the configured router
/// let listener = TcpListener::bind("0.0.0.0:3000").await?;
/// axum::serve(listener, app.into_make_service())
///     .await?;
/// ```
pub fn create_router(daemon_state: DaemonState) -> Router {
    Router::new()
        .merge(attestation_disputes_router())
        .merge(claimed_stacks_router())
        .merge(nodes_router())
        .merge(stacks_router())
        .merge(subscriptions_router())
        .merge(tasks_router())
        .with_state(daemon_state)
        .route("/health", get(health))
        .merge(openapi_routes())
}

/// Health check endpoint for the daemon.
///
/// # Returns
/// * `StatusCode::OK` - Always returns OK
#[utoipa::path(
    get,
    path = "/health",
    tag = "health",
    responses(
        (status = 200, description = "Service is healthy"),
        (status = 500, description = "Service is unhealthy")
    )
)]
pub async fn health() -> StatusCode {
    StatusCode::OK
}

/// Retrieves all node badges (ObjectID and small ID pairs) from the daemon state.
///
/// # Arguments
/// * `daemon_state` - Reference to the DaemonState containing the node badges
///
/// # Returns
/// * `Vec<(ObjectID, u64)>` - A vector of tuples containing ObjectID and small ID pairs
///
/// # Example Response
/// Returns a vector of tuples where each tuple contains:
/// - ObjectID: The unique identifier of the node badge
/// - u64: The node small ID associated with the node badge
#[instrument(level = "trace", skip_all)]
pub fn get_all_node_badges(daemon_state: &DaemonState) -> Vec<(ObjectID, u64)> {
    daemon_state.node_badges.clone()
}
