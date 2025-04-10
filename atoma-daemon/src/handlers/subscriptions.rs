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
#[openapi(paths(subscriptions_nodes_list), components(schemas(NodeSubscription)))]
pub struct SubscriptionsOpenApi;

/// Router for handling subscription-related endpoints
///
/// This function sets up the routing for various subscription-related operations,
/// including listing all subscriptions and retrieving specific subscriptions by ID.
/// Each route corresponds to a specific operation that can be performed on subscriptions
/// within the system.
pub fn subscriptions_router() -> Router<DaemonState> {
    Router::new().route(
        &format!("{SUBSCRIPTIONS_PATH}/nodes/{{node_id}}"),
        get(subscriptions_nodes_list),
    )
}

/// List subscriptions
///
/// Lists all subscriptions for a specific node identified by its small ID.
#[utoipa::path(
    get,
    path = "/nodes/{node_id}",
    params(
        ("node_id" = i64, Path, description = "The small ID of the node whose subscriptions should be retrieved")
    ),
    responses(
        (status = OK, description = "List of node subscriptions", body = Vec<NodeSubscription>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn subscriptions_nodes_list(
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
