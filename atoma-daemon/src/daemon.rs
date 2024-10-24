use atoma_state::{
    types::{NodeSubscription, Stack, StackAttestationDispute, Task},
    StateManager,
};
use atoma_sui::client::AtomaSuiClient;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use std::sync::Arc;
use sui_sdk::types::base_types::ObjectID;
use tracing::{error, instrument};

type Result<T> = std::result::Result<T, StatusCode>;

#[derive(Clone)]
pub struct DaemonState {
    client: Arc<AtomaSuiClient>,
    state_manager: StateManager,
    node_badges: Vec<(ObjectID, u64)>,
}

pub fn create_daemon_router(daemon_state: DaemonState) -> Router {
    Router::new()
        .route("/subscriptions", get(get_all_node_subscriptions))
        .route("/tasks", get(get_all_tasks))
        .route("/stacks", get(get_all_node_stacks))
        .route("/stacks/:id", get(get_node_stacks))
        .route(
            "/almost_filled_stacks/:percentage",
            get(get_all_almost_filled_stacks),
        )
        .route(
            "/almost_filled_stacks/:id/:percentage",
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
            "/against_attestation_disputes",
            get(get_all_against_attestation_disputes),
        )
        .route(
            "/own_attestation_disputes",
            get(get_all_own_attestation_disputes),
        )
        .route(
            "/own_attestation_disputes/:id",
            get(get_own_attestation_dispute),
        )
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
        .state_manager
        .get_all_node_subscriptions(
            current_node_badges
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
        .state_manager
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
            .state_manager
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
            .state_manager
            .get_stack_by_id(node_small_id)
            .await
            .map_err(|_| {
                error!("Failed to get node stack");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// Retrieves all stacks that are filled above a specified percentage threshold for all registered nodes.
///
/// # Arguments
/// * `daemon_state` - The shared state containing node badges and state manager
/// * `percentage` - The percentage threshold (0.0 to 100.0) to filter stacks
///
/// # Returns
/// * `Result<Json<Vec<Stack>>>` - A JSON response containing a list of stacks
///   - `Ok(Json<Vec<Stack>>)` - Successfully retrieved stacks
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve stacks from state manager
///
/// # Example Response
/// Returns a JSON array of Stack objects that are filled above the specified percentage threshold
#[instrument(level = "trace", skip_all)]
async fn get_all_almost_filled_stacks(
    State(daemon_state): State<DaemonState>,
    Path(percentage): Path<f64>,
) -> Result<Json<Vec<Stack>>> {
    Ok(Json(
        daemon_state
            .state_manager
            .get_almost_filled_stacks(
                &daemon_state
                    .node_badges
                    .iter()
                    .map(|(_, small_id)| *small_id as i64)
                    .collect::<Vec<_>>(),
                percentage,
            )
            .await
            .map_err(|_| {
                error!("Failed to get all almost filled stacks");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// Retrieves all stacks that are filled above a specified percentage threshold for a specific node.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the state manager
/// * `node_small_id` - The small ID of the node whose stacks should be retrieved
/// * `percentage` - The percentage threshold (0.0 to 100.0) to filter stacks
///
/// # Returns
/// * `Result<Json<Vec<Stack>>>` - A JSON response containing a list of stacks
///   - `Ok(Json<Vec<Stack>>)` - Successfully retrieved stacks
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve stacks from state manager
///
/// # Example Response
/// Returns a JSON array of Stack objects for the specified node that are filled above the percentage threshold
#[instrument(level = "trace", skip_all)]
async fn get_node_almost_filled_stacks(
    State(daemon_state): State<DaemonState>,
    Path((node_small_id, percentage)): Path<(i64, f64)>,
) -> Result<Json<Vec<Stack>>> {
    Ok(Json(
        daemon_state
            .state_manager
            .get_almost_filled_stacks(&[node_small_id], percentage)
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
            .state_manager
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
            .state_manager
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
            .state_manager
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
            .state_manager
            .get_own_attestation_disputes(&[node_small_id])
            .await
            .map_err(|_| {
                error!("Failed to get own attestation dispute");
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
