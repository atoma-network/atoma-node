use atoma_state::types::StackSettlementTicket;
use axum::{extract::Path, extract::State, http::StatusCode, routing::get, Json, Router};
use tracing::error;
use utoipa::OpenApi;

use crate::DaemonState;

pub const CLAIMED_STACKS_PATH: &str = "/claimed-stacks";

#[derive(OpenApi)]
#[openapi(
    paths(claimed_stacks_nodes_list),
    components(schemas(StackSettlementTicket))
)]
pub struct ClaimedStacksOpenApi;

pub fn claimed_stacks_router() -> Router<DaemonState> {
    Router::new().route(
        &format!("{CLAIMED_STACKS_PATH}/nodes/{{node_id}}"),
        get(claimed_stacks_nodes_list),
    )
}

/// List claimed stacks
///
/// Lists all claimed stacks for a specific node identified by its small ID.
#[utoipa::path(
    get,
    path = "/claimed_stacks/nodes/{node_id}",
    params(
        ("node_id" = i64, Path, description = "Node small ID")
    ),
    responses(
        (status = OK, description = "List of claimed stacks matching the criteria", body = Vec<StackSettlementTicket>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn claimed_stacks_nodes_list(
    State(daemon_state): State<DaemonState>,
    Path(node_id): Path<i64>,
) -> Result<Json<Vec<StackSettlementTicket>>, StatusCode> {
    let node_ids = vec![node_id];

    daemon_state
        .atoma_state
        .get_claimed_stacks(&node_ids)
        .await
        .map(Json)
        .map_err(|_| {
            error!("Failed to get claimed stacks");
            StatusCode::INTERNAL_SERVER_ERROR
        })
}
