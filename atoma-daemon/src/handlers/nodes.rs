use axum::{
    extract::State,
    http::StatusCode,
    routing::{delete, post},
    Json, Router,
};
use tracing::{error, info};
use utoipa::OpenApi;

use crate::{
    types::{
        NodeAttestationProofRequest, NodeAttestationProofResponse, NodeClaimFundsRequest,
        NodeClaimFundsResponse, NodeClaimStacksFundsRequest, NodeClaimStacksFundsResponse,
        NodeModelSubscriptionRequest, NodeModelSubscriptionResponse, NodeRegistrationRequest,
        NodeRegistrationResponse, NodeTaskSubscriptionRequest, NodeTaskSubscriptionResponse,
        NodeTaskUnsubscriptionRequest, NodeTaskUnsubscriptionResponse,
        NodeTaskUpdateSubscriptionRequest, NodeTaskUpdateSubscriptionResponse,
        NodeTrySettleStacksRequest, NodeTrySettleStacksResponse,
    },
    DaemonState,
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
pub struct NodesOpenApi;

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
            delete(nodes_task_unsubscribe),
        )
        .route(
            &format!("{NODES_PATH}/claim-funds"),
            post(nodes_claim_funds),
        )
}

/// Create node registration transaction
///
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

    let tx_digest = daemon_state
        .client
        .write()
        .await
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
        price_per_one_million_compute_units,
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
            price_per_one_million_compute_units,
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
        price_per_one_million_compute_units,
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
            price_per_one_million_compute_units,
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

/// Delete task subscription
///
/// Unsubscribes a node from a specific task.
#[utoipa::path(
    delete,
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

/// Claims stacks for a node, provided the node is running in confidential compute mode.
#[utoipa::path(
    post,
    path = "/claim-stacks-funds",
    request_body = NodeClaimFundsRequest,
    responses(
        (status = OK, description = "Node claim funds successful", body = NodeClaimFundsResponse),
        (status = INTERNAL_SERVER_ERROR, description = "Failed to submit claim funds")
    )
)]
#[tracing::instrument(level = "info", skip_all)]
pub async fn nodes_claim_stacks(
    State(daemon_state): State<DaemonState>,
    Json(value): Json<NodeClaimStacksFundsRequest>,
) -> Result<Json<NodeClaimStacksFundsResponse>, StatusCode> {
    let NodeClaimStacksFundsRequest {
        stack_small_ids,
        node_badge_id,
        num_claimed_compute_units,
        gas,
        gas_budget,
        gas_price,
    } = value;

    let tx_digest = daemon_state
        .client
        .write()
        .await
        .submit_claim_funds_for_stacks_tx(
            stack_small_ids,
            node_badge_id,
            num_claimed_compute_units,
            gas,
            gas_budget,
            gas_price,
        )
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(NodeClaimStacksFundsResponse { tx_digest }))
}
