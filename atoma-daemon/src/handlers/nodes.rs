use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use tracing::{error, info};
use utoipa::OpenApi;

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
    CommittedStackProof, DaemonState,
};

pub const NODES_PATH: &str = "/nodes";

#[derive(OpenApi)]
#[openapi(
    paths(
        nodes_register,
        nodes_model_subscribe,
        nodes_task_subscribe,
        nodes_task_update_subscription,
        nodes_task_unsubscribe,
        nodes_try_settle_stacks,
        nodes_submit_attestations,
        nodes_claim_funds
    ),
    components(schemas(
        NodeRegistrationRequest,
        NodeRegistrationResponse,
        NodeModelSubscriptionRequest,
        NodeModelSubscriptionResponse,
        NodeTaskSubscriptionRequest,
        NodeTaskSubscriptionResponse,
        NodeTaskUpdateSubscriptionRequest,
        NodeTaskUpdateSubscriptionResponse,
        NodeTaskUnsubscriptionRequest,
        NodeTaskUnsubscriptionResponse,
        NodeTrySettleStacksRequest,
        NodeTrySettleStacksResponse,
        NodeAttestationProofRequest,
        NodeAttestationProofResponse,
        NodeClaimFundsRequest,
        NodeClaimFundsResponse
    ))
)]
pub(crate) struct NodesOpenApi;

/// Router for handling node-related endpoints
///
/// This function sets up the routing for various node-related operations,
/// including registration, model subscription, task subscription, and more.
/// Each route corresponds to a specific operation that nodes can perform
/// within the system.
pub fn nodes_router() -> Router<DaemonState> {
    Router::new()
        .route(&format!("{NODES_PATH}/register"), post(nodes_register))
        .route(
            &format!("{NODES_PATH}/model-subscribe"),
            post(nodes_model_subscribe),
        )
        .route(
            &format!("{NODES_PATH}/task-subscribe"),
            post(nodes_task_subscribe),
        )
        .route(
            &format!("{NODES_PATH}/task-update-subscription"),
            post(nodes_task_update_subscription),
        )
        .route(
            &format!("{NODES_PATH}/task-unsubscribe"),
            post(nodes_task_unsubscribe),
        )
        .route(
            &format!("{NODES_PATH}/try-settle-stacks"),
            post(nodes_try_settle_stacks),
        )
        .route(
            &format!("{NODES_PATH}/submit-attestations"),
            post(nodes_submit_attestations),
        )
        .route(
            &format!("{NODES_PATH}/claim-funds"),
            post(nodes_claim_funds),
        )
}

/// Create node registration transaction
///
/// Registers a new node in the system.
#[utoipa::path(
    post,
    path = "/register",
    request_body = NodeRegistrationRequest,
    responses(
        (status = OK, description = "Node registration successful", body = NodeRegistrationResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Failed to submit registration transaction")
    )
)]
pub async fn nodes_register(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeRegistrationRequest>,
) -> Result<Json<NodeRegistrationResponse>, StatusCode> {
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

/// Create model subscription transaction
///
/// Subscribes a node to a specific model.
#[utoipa::path(
    post,
    path = "/model-subscribe",
    request_body = NodeModelSubscriptionRequest,
    responses(
        (status = OK, description = "Node model subscription successful", body = NodeModelSubscriptionResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Failed to submit model subscription transaction")
    )
)]
pub async fn nodes_model_subscribe(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeModelSubscriptionRequest>,
) -> Result<Json<NodeModelSubscriptionResponse>, StatusCode> {
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

/// Create task subscription transaction
///
/// Subscribes a node to a specific task.
#[utoipa::path(
    post,
    path = "/task-subscribe",
    request_body = NodeTaskSubscriptionRequest,
    responses(
        (status = OK, description = "Node task subscription successful", body = NodeTaskSubscriptionResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Failed to submit task subscription transaction")
    )
)]
pub async fn nodes_task_subscribe(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeTaskSubscriptionRequest>,
) -> Result<Json<NodeTaskSubscriptionResponse>, StatusCode> {
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

/// Modify task subscription
///
/// Updates an existing task subscription for a node.
#[utoipa::path(
    post,
    path = "/task-update-subscription",
    request_body = NodeTaskUpdateSubscriptionRequest,
    responses(
        (status = OK, description = "Node task update subscription successful", body = NodeTaskUpdateSubscriptionResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Failed to submit task update subscription")
    )
)]
pub async fn nodes_task_update_subscription(
    State(daemon_state): State<DaemonState>,
    Json(request): Json<NodeTaskUpdateSubscriptionRequest>,
) -> Result<Json<NodeTaskUpdateSubscriptionResponse>, StatusCode> {
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

//TODO: change to delete

/// Unsubscribes a node from a specific task.
#[utoipa::path(
    post,
    path = "/task-unsubscribe",
    request_body = NodeTaskUnsubscriptionRequest,
    responses(
        (status = OK, description = "Node task unsubscription successful", body = NodeTaskUnsubscriptionResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Failed to submit task unsubscription")
    )
)]
pub async fn nodes_task_unsubscribe(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeTaskUnsubscriptionRequest>,
) -> Result<Json<NodeTaskUnsubscriptionResponse>, StatusCode> {
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

/// Create try settle stacks transaction
///
/// Attempts to settle stacks for a node.
#[utoipa::path(
    post,
    path = "/try-settle-stacks",
    request_body = NodeTrySettleStacksRequest,
    responses(
        (status = OK, description = "Node try settle stacks successful", body = NodeTrySettleStacksResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Failed to submit try settle stacks")
    )
)]
pub async fn nodes_try_settle_stacks(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeTrySettleStacksRequest>,
) -> Result<Json<NodeTrySettleStacksResponse>, StatusCode> {
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

/// Create attestation proof transaction
///
/// Submits attestations for stack settlement.
#[utoipa::path(
    post,
    path = "/submit-attestations",
    request_body = NodeAttestationProofRequest,
    responses(
        (status = OK, description = "Node attestation proof successful", body = NodeAttestationProofResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Failed to submit attestation proof")
    )
)]
pub async fn nodes_submit_attestations(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeAttestationProofRequest>,
) -> Result<Json<NodeAttestationProofResponse>, StatusCode> {
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
                .find_map(|(nb, ns)| {
                    if *ns as i64 == node_small_id {
                        Some(*nb)
                    } else {
                        None
                    }
                })
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

/// Create claim funds transaction
///
/// Claims funds for completed stacks.
#[utoipa::path(
    post,
    path = "/claim-funds",
    request_body = NodeClaimFundsRequest,
    responses(
        (status = OK, description = "Node claim funds successful", body = NodeClaimFundsResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Failed to submit claim funds")
    )
)]
pub async fn nodes_claim_funds(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeClaimFundsRequest>,
) -> Result<Json<NodeClaimFundsResponse>, StatusCode> {
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
