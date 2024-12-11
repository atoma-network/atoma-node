use atoma_state::types::NodeSubscription;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use tracing::error;
use utoipa::OpenApi;

use crate::DaemonState;

pub const SUBSCRIPTIONS_PATH: &str = "/subscriptions";

#[derive(OpenApi)]
#[openapi(
    paths(subscriptions_list, subscriptions_get),
    components(schemas(NodeSubscription))
)]
pub(crate) struct SubscriptionsOpenApi;

/// Router for handling subscription-related endpoints
///
/// This function sets up the routing for various subscription-related operations,
/// including listing all subscriptions and retrieving specific subscriptions by ID.
/// Each route corresponds to a specific operation that can be performed on subscriptions
/// within the system.
pub fn subscriptions_router() -> Router<DaemonState> {
    Router::new()
        .route(SUBSCRIPTIONS_PATH, get(subscriptions_list))
        .route(&format!("{SUBSCRIPTIONS_PATH}/:id"), get(subscriptions_get))
}

/// List all subscriptions for currently registered nodes
///
/// Retrieves all node subscriptions for the currently registered node badges.
#[utoipa::path(
    get,
    path = "/",
    responses(
        (status = OK, description = "List of all node subscriptions", body = Vec<NodeSubscription>),
        (status = NOT_FOUND, description = "No node badges registered"),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn subscriptions_list(
    State(daemon_state): State<DaemonState>,
) -> Result<Json<Vec<NodeSubscription>>, StatusCode> {
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

//TODO: this endpoint should be deleted and add filter on subscriptions_list.
/// List all subscriptions for a specific node
///
/// Retrieves all subscriptions for a specific node identified by its small ID.
#[utoipa::path(
    get,
    path = "/{id}",
    params(
        ("id" = i64, Path, description = "The small ID of the node whose subscriptions should be retrieved")
    ),
    responses(
        (status = OK, description = "List of node subscriptions", body = Vec<NodeSubscription>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn subscriptions_get(
    State(daemon_state): State<DaemonState>,
    Path(node_small_id): Path<i64>,
) -> Result<Json<Vec<NodeSubscription>>, StatusCode> {
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
