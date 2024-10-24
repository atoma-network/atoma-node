use atoma_state::{
    types::{NodeSubscription, Stack, StackAttestationDispute, StackSettlementTicket, Task},
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
use tokio::sync::RwLock;
use tracing::{error, instrument};

use crate::{
    compute_committed_stack_proof,
    types::{
        NodeModelSubscriptionRequest, NodeModelSubscriptionResponse, NodeRegistrationRequest,
        NodeRegistrationResponse, NodeTaskSubscriptionRequest, NodeTaskSubscriptionResponse,
        NodeTaskUnsubscriptionRequest, NodeTaskUnsubscriptionResponse, NodeTrySettleStackRequest,
        NodeTrySettleStackResponse,
    },
    CommittedStackProof,
};

type Result<T> = std::result::Result<T, StatusCode>;

#[derive(Clone)]
pub struct DaemonState {
    client: Arc<RwLock<AtomaSuiClient>>,
    state_manager: StateManager,
    node_badges: Vec<(ObjectID, u64)>,
}

pub fn create_daemon_router(daemon_state: DaemonState) -> Router {
    Router::new()
        .route("/subscriptions", get(get_all_node_subscriptions))
        .route("/subscriptions/:id", get(get_node_subscriptions))
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
            "/own_attestation_disputes",
            get(get_all_own_attestation_disputes),
        )
        .route(
            "/own_attestation_disputes/:id",
            get(get_own_attestation_dispute),
        )
        .route("/claimed_stacks", get(get_all_claimed_stacks))
        .route("/claimed_stacks/:id", get(get_node_claimed_stacks))
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

#[instrument(level = "trace", skip_all)]
async fn get_node_subscriptions(
    State(daemon_state): State<DaemonState>,
    Path(node_small_id): Path<i64>,
) -> Result<Json<Vec<NodeSubscription>>> {
    Ok(Json(
        daemon_state
            .state_manager
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
            .state_manager
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
            .state_manager
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
        node_small_id,
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
            task_small_id,
            node_small_id,
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
        node_small_id,
        gas,
        gas_budget,
        gas_price,
    } = value;
    let tx_digest = daemon_state
        .client
        .write()
        .await
        .submit_unsubscribe_node_from_task_tx(
            task_small_id,
            node_small_id,
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
/// * `Result<Json<NodeTrySettleStackResponse>>` - A JSON response containing the transaction digest.
///   - `Ok(Json<NodeTrySettleStackResponse>)` - Successfully submitted the try settle stack transaction.
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to submit the transaction or retrieve necessary data.
///
/// # Example Request
/// ```json
/// {
///     "stack_small_id": 123,
///     "num_claimed_compute_units": 50,
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
async fn submit_node_try_settle_stack_tx(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeTrySettleStackRequest>,
) -> Result<Json<NodeTrySettleStackResponse>> {
    let NodeTrySettleStackRequest {
        stack_small_id,
        num_claimed_compute_units,
        node_small_id,
        gas,
        gas_budget,
        gas_price,
    } = value;

    let total_hash = daemon_state
        .state_manager
        .get_stack_total_hash(stack_small_id as i64)
        .await
        .map_err(|_| {
            error!("Failed to get stack total hash");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let CommittedStackProof {
        root: committed_stack_proof,
        leaf: stack_merkle_leaf,
    } = compute_committed_stack_proof(total_hash)?;

    let tx_digest = daemon_state
        .client
        .write()
        .await
        .submit_try_settle_stack_tx(
            stack_small_id,
            node_small_id,
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
    Ok(Json(NodeTrySettleStackResponse { tx_digest }))
}
