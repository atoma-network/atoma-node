use atoma_state::{
    state_manager::AtomaState,
    types::{NodeSubscription, Stack, StackAttestationDispute, StackSettlementTicket, Task},
};
use atoma_sui::client::AtomaSuiClient;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use sui_sdk::types::base_types::ObjectID;
use tokio::{net::TcpListener, signal, sync::RwLock};
use tracing::{error, info, instrument};

use crate::{
    calculate_node_index, compute_committed_stack_proof,
    types::{
        NodeAttestationProofRequest, NodeAttestationProofResponse, NodeClaimFundsRequest,
        NodeClaimFundsResponse, NodeModelSubscriptionRequest, NodeModelSubscriptionResponse,
        NodeRegistrationRequest, NodeRegistrationResponse, NodeTaskSubscriptionRequest,
        NodeTaskSubscriptionResponse, NodeTaskUnsubscriptionRequest,
        NodeTaskUnsubscriptionResponse, NodeTaskUpdateSubscriptionRequest,
        NodeTaskUpdateSubscriptionResponse, NodeTrySettleStacksRequest,
        NodeTrySettleStacksResponse,
    },
    CommittedStackProof,
};

type Result<T> = std::result::Result<T, StatusCode>;

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
/// use atoma_daemon::{DaemonState, run_daemon};
///
/// async fn start_server() -> Result<(), Box<dyn std::error::Error>> {
///     let daemon_state = DaemonState::new(/* ... */);
///     let listener = TcpListener::bind("127.0.0.1:3000").await?;
///     
///     run_daemon(daemon_state, listener).await
/// }
/// ```
pub async fn run_daemon(
    daemon_state: DaemonState,
    tcp_listener: TcpListener,
) -> anyhow::Result<()> {
    let daemon_router = create_daemon_router(daemon_state);
    let shutdown_signal = async {
        signal::ctrl_c()
            .await
            .expect("Failed to parse Ctrl+C signal");
        info!("Shutting down server...");
    };
    let server = axum::serve(tcp_listener, daemon_router.into_make_service())
        .with_graceful_shutdown(shutdown_signal);
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
/// ## Subscription Management
/// * `GET /subscriptions` - Get all subscriptions for registered nodes
/// * `GET /subscriptions/:id` - Get subscriptions for a specific node
/// * `POST /model_subscribe` - Subscribe a node to a model
/// * `POST /task_subscribe` - Subscribe a node to a task
/// * `POST /task_unsubscribe` - Unsubscribe a node from a task
///
/// ## Task Management
/// * `GET /tasks` - Get all available tasks
///
/// ## Stack Operations
/// * `GET /stacks` - Get all stacks for registered nodes
/// * `GET /stacks/:id` - Get stacks for a specific node
/// * `GET /almost_filled_stacks/:fraction` - Get stacks filled above specified fraction
/// * `GET /almost_filled_stacks/:id/:fraction` - Get node's stacks filled above fraction
/// * `GET /claimed_stacks` - Get all claimed stacks
/// * `GET /claimed_stacks/:id` - Get claimed stacks for a specific node
/// * `POST /try_settle_stack_ids` - Attempt to settle specified stacks
/// * `POST /submit_stack_settlement_attestations` - Submit attestations for stack settlement
/// * `POST /claim_funds` - Claim funds from completed stacks
///
/// ## Attestation Disputes
/// * `GET /against_attestation_disputes` - Get disputes against registered nodes
/// * `GET /against_attestation_disputes/:id` - Get disputes against a specific node
/// * `GET /own_attestation_disputes` - Get disputes initiated by registered nodes
/// * `GET /own_attestation_disputes/:id` - Get disputes initiated by a specific node
///
/// ## Node Registration
/// * `POST /register` - Register a new node
///
/// # Example
/// ```rust,ignore
/// use atoma_daemon::DaemonState;
///
/// let daemon_state = DaemonState::new(/* ... */);
/// let app = create_daemon_router(daemon_state);
/// // Start the server with the configured router
/// axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
///     .serve(app.into_make_service())
///     .await?;
/// ```
pub fn create_daemon_router(daemon_state: DaemonState) -> Router {
    Router::new()
        .route("/subscriptions", get(get_all_node_subscriptions))
        .route("/subscriptions/:id", get(get_node_subscriptions))
        .route("/tasks", get(get_all_tasks))
        .route("/stacks", get(get_all_node_stacks))
        .route("/stacks/:id", get(get_node_stacks))
        .route(
            "/almost_filled_stacks/:fraction",
            get(get_all_almost_filled_stacks),
        )
        .route(
            "/almost_filled_stacks/:id/:fraction",
            get(get_node_almost_filled_stacks),
        )
        .route(
            "/against_attestation_disputes",
            get(get_all_against_attestation_disputes),
        )
        .route(
            "/against_attestation_disputes/:id",
            get(get_against_attestation_dispute),
        )
        .route(
            "/own_attestation_disputes",
            get(get_all_own_attestation_disputes),
        )
        .route(
            "/own_attestation_disputes/:id",
            get(get_own_attestation_dispute),
        )
        .route("/claimed_stacks", get(get_all_claimed_stacks))
        .route("/claimed_stacks/:id", get(get_node_claimed_stacks))
        .route("/register", post(submit_node_registration_tx))
        .route("/model_subscribe", post(submit_node_model_subscription_tx))
        .route("/task_subscribe", post(submit_node_task_subscription_tx))
        .route(
            "/task_unsubscribe",
            post(submit_node_task_unsubscription_tx),
        )
        .route(
            "/try_settle_stack_ids",
            post(submit_node_try_settle_stacks_tx),
        )
        .route(
            "/submit_stack_settlement_attestations",
            post(submit_stack_settlement_attestations_tx),
        )
        .route("/claim_funds", post(submit_claim_funds_tx))
        .with_state(daemon_state)
}

/// Retrieves all node subscriptions for the currently registered node badges.
///
/// # Arguments
/// * `daemon_state` - The shared state containing node badges and state manager
///
/// # Returns
/// * `Result<Json<Vec<NodeSubscription>>>` - A JSON response containing a list of node subscriptions
///   - `Ok(Json<Vec<NodeSubscription>>)` - Successfully retrieved subscriptions
///   - `Err(StatusCode::NOT_FOUND)` - No node badges are registered
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve subscriptions from state manager
///
/// # Example Response
/// Returns a JSON array of NodeSubscription objects for all registered nodes
#[instrument(level = "trace", skip_all)]
async fn get_all_node_subscriptions(
    State(daemon_state): State<DaemonState>,
) -> Result<Json<Vec<NodeSubscription>>> {
    let current_node_badges = daemon_state.node_badges;
    if current_node_badges.is_empty() {
        return Err(StatusCode::NOT_FOUND);
    }
    let all_node_subscriptions = daemon_state
        .atoma_state
        .get_all_node_subscriptions(
            &current_node_badges
                .iter()
                .map(|(_, small_id)| *small_id as i64)
                .collect::<Vec<_>>(),
        )
        .await
        .map_err(|_| {
            error!("Failed to get all node subscriptions");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok(Json(all_node_subscriptions))
}

/// Retrieves all subscriptions for a specific node identified by its small ID.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the state manager
/// * `node_small_id` - The small ID of the node whose subscriptions should be retrieved
///
/// # Returns
/// * `Result<Json<Vec<NodeSubscription>>>` - A JSON response containing a list of subscriptions
///   - `Ok(Json<Vec<NodeSubscription>>)` - Successfully retrieved subscriptions
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve subscriptions from state manager
///
/// # Example Response
/// Returns a JSON array of NodeSubscription objects for the specified node, which may include:
/// ```json
/// [
///     {
///         "node_small_id": 123,
///         "model_name": "example_model",
///         "echelon_id": 1,
///         "subscription_time": "2024-03-21T12:00:00Z"
///     }
/// ]
/// ```
#[instrument(level = "trace", skip_all)]
async fn get_node_subscriptions(
    State(daemon_state): State<DaemonState>,
    Path(node_small_id): Path<i64>,
) -> Result<Json<Vec<NodeSubscription>>> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_all_node_subscriptions(&[node_small_id])
            .await
            .map_err(|_| {
                error!("Failed to get node subscriptions");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// Retrieves all tasks from the state manager.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the state manager
///
/// # Returns
/// * `Result<Json<Vec<Task>>>` - A JSON response containing a list of tasks
///   - `Ok(Json<Vec<Task>>)` - Successfully retrieved tasks
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve tasks from state manager
///
/// # Example Response
/// Returns a JSON array of Task objects representing all tasks in the system
#[instrument(level = "trace", skip_all)]
async fn get_all_tasks(State(daemon_state): State<DaemonState>) -> Result<Json<Vec<Task>>> {
    let all_tasks = daemon_state
        .atoma_state
        .get_all_tasks()
        .await
        .map_err(|_| {
            error!("Failed to get all tasks");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok(Json(all_tasks))
}

/// Retrieves all stacks associated with the currently registered node badges.
///
/// # Arguments
/// * `daemon_state` - The shared state containing node badges and state manager
///
/// # Returns
/// * `Result<Json<Vec<Stack>>>` - A JSON response containing a list of stacks
///   - `Ok(Json<Vec<Stack>>)` - Successfully retrieved stacks
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve stacks from state manager
///
/// # Example Response
/// Returns a JSON array of Stack objects for all registered nodes
#[instrument(level = "trace", skip_all)]
async fn get_all_node_stacks(State(daemon_state): State<DaemonState>) -> Result<Json<Vec<Stack>>> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_stacks_by_node_small_ids(
                &daemon_state
                    .node_badges
                    .iter()
                    .map(|(_, small_id)| *small_id as i64)
                    .collect::<Vec<_>>(),
            )
            .await
            .map_err(|_| {
                error!("Failed to get all node stacks");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// Retrieves all stacks for a specific node identified by its small ID.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the state manager
/// * `node_small_id` - The small ID of the node whose stacks should be retrieved
///
/// # Returns
/// * `Result<Json<Vec<Stack>>>` - A JSON response containing a list of stacks
///   - `Ok(Json<Vec<Stack>>)` - Successfully retrieved stacks
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve stacks from state manager
///
/// # Example Response
/// Returns a JSON array of Stack objects for the specified node
#[instrument(level = "trace", skip_all)]
async fn get_node_stacks(
    State(daemon_state): State<DaemonState>,
    Path(node_small_id): Path<i64>,
) -> Result<Json<Vec<Stack>>> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_stack_by_id(node_small_id)
            .await
            .map_err(|_| {
                error!("Failed to get node stack");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// Retrieves all stacks that are filled above a specified fraction threshold for all registered nodes.
///
/// # Arguments
/// * `daemon_state` - The shared state containing node badges and state manager
/// * `fraction` - The fraction threshold (0.0 to 100.0) to filter stacks
///
/// # Returns
/// * `Result<Json<Vec<Stack>>>` - A JSON response containing a list of stacks
///   - `Ok(Json<Vec<Stack>>)` - Successfully retrieved stacks
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve stacks from state manager
///
/// # Example Response
/// Returns a JSON array of Stack objects that are filled above the specified fraction threshold
#[instrument(level = "trace", skip_all)]
async fn get_all_almost_filled_stacks(
    State(daemon_state): State<DaemonState>,
    Path(fraction): Path<f64>,
) -> Result<Json<Vec<Stack>>> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_almost_filled_stacks(
                &daemon_state
                    .node_badges
                    .iter()
                    .map(|(_, small_id)| *small_id as i64)
                    .collect::<Vec<_>>(),
                fraction,
            )
            .await
            .map_err(|_| {
                error!("Failed to get all almost filled stacks");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// Retrieves all stacks that are filled above a specified fraction threshold for a specific node.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the state manager
/// * `node_small_id` - The small ID of the node whose stacks should be retrieved
/// * `fraction` - The fraction threshold (0.0 to 100.0) to filter stacks
///
/// # Returns
/// * `Result<Json<Vec<Stack>>>` - A JSON response containing a list of stacks
///   - `Ok(Json<Vec<Stack>>)` - Successfully retrieved stacks
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve stacks from state manager
///
/// # Example Response
/// Returns a JSON array of Stack objects for the specified node that are filled above the fraction threshold
#[instrument(level = "trace", skip_all)]
async fn get_node_almost_filled_stacks(
    State(daemon_state): State<DaemonState>,
    Path((node_small_id, fraction)): Path<(i64, f64)>,
) -> Result<Json<Vec<Stack>>> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_almost_filled_stacks(&[node_small_id], fraction)
            .await
            .map_err(|_| {
                error!("Failed to get node almost filled stacks");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// Retrieves all attestation disputes against the currently registered nodes.
///
/// # Arguments
/// * `daemon_state` - The shared state containing node badges and state manager
///
/// # Returns
/// * `Result<Json<Vec<StackAttestationDispute>>>` - A JSON response containing a list of attestation disputes
///   - `Ok(Json<Vec<StackAttestationDispute>>)` - Successfully retrieved attestation disputes
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve attestation disputes from state manager
///
/// # Example Response
/// Returns a JSON array of StackAttestationDispute objects where the registered nodes are the defendants
#[instrument(level = "trace", skip_all)]
async fn get_all_against_attestation_disputes(
    State(daemon_state): State<DaemonState>,
) -> Result<Json<Vec<StackAttestationDispute>>> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_against_attestation_disputes(
                &daemon_state
                    .node_badges
                    .iter()
                    .map(|(_, small_id)| *small_id as i64)
                    .collect::<Vec<_>>(),
            )
            .await
            .map_err(|_| {
                error!("Failed to get all attestation disputes");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// Retrieves all attestation disputes against a specific node.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the state manager
/// * `node_small_id` - The small ID of the node whose disputes should be retrieved
///
/// # Returns
/// * `Result<Json<Vec<StackAttestationDispute>>>` - A JSON response containing a list of attestation disputes
///   - `Ok(Json<Vec<StackAttestationDispute>>)` - Successfully retrieved attestation disputes
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve attestation disputes from state manager
///
/// # Example Response
/// Returns a JSON array of StackAttestationDispute objects where the specified node is the defendant
#[instrument(level = "trace", skip_all)]
async fn get_against_attestation_dispute(
    State(daemon_state): State<DaemonState>,
    Path(node_small_id): Path<i64>,
) -> Result<Json<Vec<StackAttestationDispute>>> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_against_attestation_disputes(&[node_small_id])
            .await
            .map_err(|_| {
                error!("Failed to get against attestation dispute");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// Retrieves all attestation disputes initiated by the currently registered nodes.
///
/// # Arguments
/// * `daemon_state` - The shared state containing node badges and state manager
///
/// # Returns
/// * `Result<Json<Vec<StackAttestationDispute>>>` - A JSON response containing a list of attestation disputes
///   - `Ok(Json<Vec<StackAttestationDispute>>)` - Successfully retrieved attestation disputes
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve attestation disputes from state manager
///
/// # Example Response
/// Returns a JSON array of StackAttestationDispute objects where the registered nodes are the plaintiffs
#[instrument(level = "trace", skip_all)]
async fn get_all_own_attestation_disputes(
    State(daemon_state): State<DaemonState>,
) -> Result<Json<Vec<StackAttestationDispute>>> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_own_attestation_disputes(
                &daemon_state
                    .node_badges
                    .iter()
                    .map(|(_, small_id)| *small_id as i64)
                    .collect::<Vec<_>>(),
            )
            .await
            .map_err(|_| {
                error!("Failed to get all own attestation disputes");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// Retrieves all attestation disputes initiated by a specific node.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the state manager
/// * `node_small_id` - The small ID of the node whose initiated disputes should be retrieved
///
/// # Returns
/// * `Result<Json<Vec<StackAttestationDispute>>>` - A JSON response containing a list of attestation disputes
///   - `Ok(Json<Vec<StackAttestationDispute>>)` - Successfully retrieved attestation disputes
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve attestation disputes from state manager
///
/// # Example Response
/// Returns a JSON array of StackAttestationDispute objects where the specified node is the plaintiff
#[instrument(level = "trace", skip_all)]
async fn get_own_attestation_dispute(
    State(daemon_state): State<DaemonState>,
    Path(node_small_id): Path<i64>,
) -> Result<Json<Vec<StackAttestationDispute>>> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_own_attestation_disputes(&[node_small_id])
            .await
            .map_err(|_| {
                error!("Failed to get own attestation dispute");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// Retrieves all claimed stacks for the currently registered node badges.
///
/// # Arguments
/// * `daemon_state` - The shared state containing node badges and state manager
///
/// # Returns
/// * `Result<Json<Vec<StackSettlementTicket>>>` - A JSON response containing a list of claimed stacks
///   - `Ok(Json<Vec<StackSettlementTicket>>)` - Successfully retrieved claimed stacks
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve claimed stacks from state manager
///
/// # Example Response
/// Returns a JSON array of StackSettlementTicket objects for all registered nodes
#[instrument(level = "trace", skip_all)]
async fn get_all_claimed_stacks(
    State(daemon_state): State<DaemonState>,
) -> Result<Json<Vec<StackSettlementTicket>>> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_claimed_stacks(
                &daemon_state
                    .node_badges
                    .iter()
                    .map(|(_, small_id)| *small_id as i64)
                    .collect::<Vec<_>>(),
            )
            .await
            .map_err(|_| {
                error!("Failed to get all claimed stacks");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// Retrieves all claimed stacks for a specific node identified by its small ID.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the state manager
/// * `node_small_id` - The small ID of the node whose claimed stacks should be retrieved
///
/// # Returns
/// * `Result<Json<Vec<StackSettlementTicket>>>` - A JSON response containing a list of claimed stacks
///   - `Ok(Json<Vec<StackSettlementTicket>>)` - Successfully retrieved claimed stacks
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve claimed stacks from state manager
///
/// # Example Response
/// Returns a JSON array of StackSettlementTicket objects for the specified node
#[instrument(level = "trace", skip_all)]
async fn get_node_claimed_stacks(
    State(daemon_state): State<DaemonState>,
    Path(node_small_id): Path<i64>,
) -> Result<Json<Vec<StackSettlementTicket>>> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_claimed_stacks(&[node_small_id])
            .await
            .map_err(|_| {
                error!("Failed to get node claimed stacks");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
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
fn get_all_node_badges(daemon_state: &DaemonState) -> Vec<(ObjectID, u64)> {
    daemon_state.node_badges.clone()
}

/// Submits a node registration transaction.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the client for transaction submission.
/// * `value` - A JSON payload containing the node registration request details.
///
/// # Returns
/// * `Result<Json<NodeRegistrationResponse>>` - A JSON response containing the transaction digest.
///   - `Ok(Json<NodeRegistrationResponse>)` - Successfully submitted the node registration transaction.
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to submit the transaction.
///
/// # Example Request
/// ```json
/// {
///     "gas": "0x123",
///     "gas_budget": 1000,
///     "gas_price": 10
/// }
/// ```
///
/// # Example Response
/// ```json
/// {
///     "tx_digest": "0xabc"
/// }
/// ```
#[instrument(level = "trace", skip_all)]
async fn submit_node_registration_tx(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeRegistrationRequest>,
) -> Result<Json<NodeRegistrationResponse>> {
    let NodeRegistrationRequest {
        gas,
        gas_budget,
        gas_price,
    } = value;
    let mut tx_client = daemon_state.client.write().await;
    let tx_digest = tx_client
        .submit_node_registration_tx(gas, gas_budget, gas_price)
        .await
        .map_err(|_| {
            error!("Failed to submit node registration tx");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    info!("Node registration tx submitted: {}", tx_digest);
    Ok(Json(NodeRegistrationResponse { tx_digest }))
}

/// Submits a node model subscription transaction.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the client for transaction submission.
/// * `value` - A JSON payload containing the node model subscription request details.
///
/// # Returns
/// * `Result<Json<NodeModelSubscriptionResponse>>` - A JSON response containing the transaction digest.
///   - `Ok(Json<NodeModelSubscriptionResponse>)` - Successfully submitted the node model subscription transaction.
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to submit the transaction.
///
/// # Example Request
/// ```json
/// {
///     "model_name": "example_model",
///     "echelon_id": 1,
///     "node_badge_id": "0x123",
///     "gas": "0x456",
///     "gas_budget": 1000,
///     "gas_price": 10
/// }
/// ```
///
/// # Example Response
/// ```json
/// {
///     "tx_digest": "0xabc"
/// }
/// ```
#[instrument(level = "trace", skip_all)]
async fn submit_node_model_subscription_tx(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeModelSubscriptionRequest>,
) -> Result<Json<NodeModelSubscriptionResponse>> {
    let NodeModelSubscriptionRequest {
        model_name,
        echelon_id,
        node_badge_id,
        gas,
        gas_budget,
        gas_price,
    } = value;
    let tx_digest = daemon_state
        .client
        .write()
        .await
        .submit_node_model_subscription_tx(
            &model_name,
            echelon_id,
            node_badge_id,
            gas,
            gas_budget,
            gas_price,
        )
        .await
        .map_err(|_| {
            error!("Failed to submit node model subscription tx");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok(Json(NodeModelSubscriptionResponse { tx_digest }))
}

/// Submits a node task subscription transaction.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the client for transaction submission.
/// * `value` - A JSON payload containing the node task subscription request details.
///
/// # Returns
/// * `Result<Json<NodeTaskSubscriptionResponse>>` - A JSON response containing the transaction digest.
///   - `Ok(Json<NodeTaskSubscriptionResponse>)` - Successfully submitted the node task subscription transaction.
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to submit the transaction.
///
/// # Example Request
/// ```json
/// {
///     "task_small_id": 123,
///     "node_small_id": 456,
///     "price_per_compute_unit": 10,
///     "max_num_compute_units": 100,
///     "gas": "0x789",
///     "gas_budget": 1000,
///     "gas_price": 10
/// }
/// ```
///
/// # Example Response
/// ```json
/// {
///     "tx_digest": "0xabc"
/// }
/// ```
#[instrument(level = "trace", skip_all)]
async fn submit_node_task_subscription_tx(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeTaskSubscriptionRequest>,
) -> Result<Json<NodeTaskSubscriptionResponse>> {
    let NodeTaskSubscriptionRequest {
        task_small_id,
        node_badge_id,
        price_per_compute_unit,
        max_num_compute_units,
        gas,
        gas_budget,
        gas_price,
    } = value;
    let tx_digest = daemon_state
        .client
        .write()
        .await
        .submit_node_task_subscription_tx(
            task_small_id as u64,
            node_badge_id,
            price_per_compute_unit,
            max_num_compute_units,
            gas,
            gas_budget,
            gas_price,
        )
        .await
        .map_err(|_| {
            error!("Failed to submit node task subscription tx");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok(Json(NodeTaskSubscriptionResponse { tx_digest }))
}

#[instrument(level = "trace", skip_all)]
pub async fn submit_update_node_task_subscription_tx(
    State(daemon_state): State<DaemonState>,
    Json(request): Json<NodeTaskUpdateSubscriptionRequest>,
) -> Result<Json<NodeTaskUpdateSubscriptionResponse>> {
    let NodeTaskUpdateSubscriptionRequest {
        task_small_id,
        node_badge_id,
        price_per_compute_unit,
        max_num_compute_units,
        gas,
        gas_budget,
        gas_price,
    } = request;
    let tx_digest = daemon_state
        .client
        .write()
        .await
        .submit_update_node_task_subscription_tx(
            task_small_id as u64,
            node_badge_id,
            price_per_compute_unit,
            max_num_compute_units,
            gas,
            gas_budget,
            gas_price,
        )
        .await
        .map_err(|_| {
            error!("Failed to submit node task update subscription");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok(Json(NodeTaskUpdateSubscriptionResponse { tx_digest }))
}

/// Submits a node task unsubscription transaction.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the client for transaction submission.
/// * `value` - A JSON payload containing the node task unsubscription request details.
///
/// # Returns
/// * `Result<Json<NodeTaskUnsubscriptionResponse>>` - A JSON response containing the transaction digest.
///   - `Ok(Json<NodeTaskUnsubscriptionResponse>)` - Successfully submitted the node task unsubscription transaction.
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to submit the transaction.
///
/// # Example Request
/// ```json
/// {
///     "task_small_id": 123,
///     "node_small_id": 456,
///     "gas": "0x789",
///     "gas_budget": 1000,
///     "gas_price": 10
/// }
/// ```
///
/// # Example Response
/// ```json
/// {
///     "tx_digest": "0xabc"
/// }
/// ```
#[instrument(level = "trace", skip_all)]
async fn submit_node_task_unsubscription_tx(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeTaskUnsubscriptionRequest>,
) -> Result<Json<NodeTaskUnsubscriptionResponse>> {
    let NodeTaskUnsubscriptionRequest {
        task_small_id,
        node_badge_id,
        gas,
        gas_budget,
        gas_price,
    } = value;
    let tx_digest = daemon_state
        .client
        .write()
        .await
        .submit_unsubscribe_node_from_task_tx(
            task_small_id as u64,
            node_badge_id,
            gas,
            gas_budget,
            gas_price,
        )
        .await
        .map_err(|_| {
            error!("Failed to submit node task unsubscription tx");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok(Json(NodeTaskUnsubscriptionResponse { tx_digest }))
}

/// Submits a transaction to attempt settling a stack for a node.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the client and state manager for transaction submission.
/// * `value` - A JSON payload containing the node try settle stack request details.
///
/// # Returns
/// * `Result<Json<NodeTrySettleStacksResponse>>` - A JSON response containing the transaction digests.
///   - `Ok(Json<NodeTrySettleStacksResponse>)` - Successfully submitted the try settle stacks transaction.
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to submit the transaction or retrieve necessary data.
///
/// # Example Request
/// ```json
/// {
///     "stack_small_ids": [123, 456],
///     "num_claimed_compute_units": 50,
///     "node_small_id": 789,
///     "gas": "0x789",
///     "gas_budget": 1000,
///     "gas_price": 10
/// }
/// ```
///
/// # Example Response
/// ```json
/// {
///     "tx_digest": "0xabc"
/// }
/// ```
#[instrument(level = "trace", skip_all)]
async fn submit_node_try_settle_stacks_tx(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeTrySettleStacksRequest>,
) -> Result<Json<NodeTrySettleStacksResponse>> {
    let NodeTrySettleStacksRequest {
        stack_small_ids,
        num_claimed_compute_units,
        node_badge_id,
        gas,
        gas_budget,
        gas_price,
    } = value;

    let total_hashes = daemon_state
        .atoma_state
        .get_all_total_hashes(&stack_small_ids)
        .await
        .map_err(|_| {
            error!("Failed to get stack total hash");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let mut tx_digests: Vec<String> = Vec::with_capacity(stack_small_ids.len());
    for (stack_small_id, total_hash) in stack_small_ids.iter().zip(total_hashes.iter()) {
        let CommittedStackProof {
            root: committed_stack_proof,
            leaf: stack_merkle_leaf,
        } = compute_committed_stack_proof(total_hash, 0)?;

        let tx_digest = daemon_state
            .client
            .write()
            .await
            .submit_try_settle_stack_tx(
                *stack_small_id as u64,
                node_badge_id,
                num_claimed_compute_units,
                committed_stack_proof,
                stack_merkle_leaf,
                gas,
                gas_budget,
                gas_price,
            )
            .await
            .map_err(|_| {
                error!("Failed to submit node try settle stack tx");
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
        tx_digests.push(tx_digest);
    }
    Ok(Json(NodeTrySettleStacksResponse { tx_digests }))
}

/// Submits attestation proof transactions for one or more stacks.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the client and state manager for transaction submission
/// * `value` - A JSON payload containing the node attestation proof request details
///
/// # Returns
/// * `Result<Json<NodeAttestationProofResponse>>` - A JSON response containing the transaction digests
///   - `Ok(Json<NodeAttestationProofResponse>)` - Successfully submitted the attestation proof transactions
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to submit transactions or retrieve necessary data
///
/// # Example Request
/// ```json
/// {
///     "stack_small_ids": [123, 456],
///     "node_small_id": 789,  // Optional: if not provided, uses all registered node badges
///     "gas": "0x789",
///     "gas_budget": 1000,
///     "gas_price": 10
/// }
/// ```
///
/// # Example Response
/// ```json
/// {
///     "tx_digests": ["0xabc", "0xdef"]  // One digest per successful attestation submission
/// }
/// ```
///
/// # Processing Flow
/// 1. Retrieves stack settlement tickets and total hashes for the provided stack IDs
/// 2. Determines which nodes need to submit attestations (single node or all registered nodes)
/// 3. For each stack and attestation node combination:
///    - Computes the committed stack proof
///    - Submits an attestation transaction
///    - Collects the transaction digest
///
/// Note: The attestation node index is offset by 1 since the 0th index is reserved for the original selected node.
#[instrument(level = "trace", skip_all)]
async fn submit_stack_settlement_attestations_tx(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeAttestationProofRequest>,
) -> Result<Json<NodeAttestationProofResponse>> {
    let NodeAttestationProofRequest {
        stack_small_ids,
        node_small_id,
        gas,
        gas_budget,
        gas_price,
    } = value;

    let stack_settlement_tickets = daemon_state
        .atoma_state
        .get_stack_settlement_tickets(&stack_small_ids)
        .await
        .map_err(|_| {
            error!("Failed to get stacks");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let total_hashes = daemon_state
        .atoma_state
        .get_all_total_hashes(&stack_small_ids)
        .await
        .map_err(|_| {
            error!("Failed to get stack total hash");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let node_small_ids = if let Some(node_small_id) = node_small_id {
        vec![node_small_id]
    } else {
        daemon_state
            .node_badges
            .iter()
            .map(|(_, id)| *id as i64)
            .collect::<Vec<i64>>()
    };

    let mut tx_digests = Vec::new();
    for (stack_settlement_ticket, total_hash) in
        stack_settlement_tickets.iter().zip(total_hashes.iter())
    {
        let stack_small_id = stack_settlement_ticket.stack_small_id;
        let attestation_nodes: Vec<i64> = serde_json::from_str(
            &stack_settlement_ticket.requested_attestation_nodes,
        )
        .map_err(|_| {
            error!("Failed to parse attestation nodes");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        let attestation_node_indices = calculate_node_index(&node_small_ids, &attestation_nodes)?;

        for attestation_node_index in attestation_node_indices {
            // NOTE: Need to account for the fact that the 0th index is reserved for the
            // original selected node, so we need to sum 1 to the node index
            let CommittedStackProof {
                root: committed_stack_proof,
                leaf: stack_merkle_leaf,
            } = compute_committed_stack_proof(
                total_hash,
                attestation_node_index.attestation_node_index as u64 + 1,
            )?;

            let node_small_id = node_small_ids[attestation_node_index.node_small_id_index];
            let node_badge_id = daemon_state
                .node_badges
                .iter()
                .find(|(_, ns)| *ns as i64 == node_small_id)
                .map(|(nb, _)| *nb)
                .unwrap();

            let tx_digest = daemon_state
                .client
                .write()
                .await
                .submit_stack_settlement_attestation_tx(
                    stack_small_id as u64,
                    Some(node_badge_id),
                    committed_stack_proof,
                    stack_merkle_leaf,
                    gas,
                    gas_budget,
                    gas_price,
                )
                .await
                .map_err(|_| {
                    error!("Failed to submit node attestation proof tx");
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;
            tx_digests.push(tx_digest);
        }
    }
    Ok(Json(NodeAttestationProofResponse { tx_digests }))
}

/// Submits a transaction to claim funds for completed stacks.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the client for transaction submission
/// * `value` - A JSON payload containing the node claim funds request details
///
/// # Returns
/// * `Result<Json<NodeClaimFundsResponse>>` - A JSON response containing the transaction digest
///   - `Ok(Json<NodeClaimFundsResponse>)` - Successfully submitted the claim funds transaction
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to submit the transaction
///
/// # Example Request
/// ```json
/// {
///     "stack_small_ids": [123, 456],  // IDs of stacks to claim funds from
///     "node_small_id": 789,           // Optional: specific node ID to claim for
///     "gas": "0x789",
///     "gas_budget": 1000,
///     "gas_price": 10
/// }
/// ```
///
/// # Example Response
/// ```json
/// {
///     "tx_digest": "0xabc"  // Transaction digest of the claim funds submission
/// }
/// ```
#[instrument(level = "trace", skip_all)]
async fn submit_claim_funds_tx(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeClaimFundsRequest>,
) -> Result<Json<NodeClaimFundsResponse>> {
    let NodeClaimFundsRequest {
        stack_small_ids,
        node_badge_id,
        gas,
        gas_budget,
        gas_price,
    } = value;

    let tx_digest = daemon_state
        .client
        .write()
        .await
        .submit_claim_funds_tx(
            stack_small_ids.iter().map(|id| *id as u64).collect(),
            node_badge_id,
            gas,
            gas_budget,
            gas_price,
        )
        .await
        .map_err(|_| {
            error!("Failed to submit node claim funds tx");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok(Json(NodeClaimFundsResponse { tx_digest }))
}
