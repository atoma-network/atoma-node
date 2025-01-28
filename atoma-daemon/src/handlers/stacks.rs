use atoma_state::types::{Stack, StackSettlementTicket};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use serde::Deserialize;
use tracing::error;
use utoipa::{OpenApi, ToSchema};

use crate::DaemonState;

pub const STACKS_PATH: &str = "/stacks";

#[derive(Deserialize, ToSchema)]
pub struct StackQuery {
    min_fill_fraction: Option<f64>,
}

#[derive(OpenApi)]
#[openapi(
    paths(stacks_nodes_list),
    components(schemas(Stack, StackSettlementTicket, StackQuery))
)]
pub struct StacksOpenApi;

pub fn stacks_router() -> Router<DaemonState> {
    Router::new().route(
        &format!("{STACKS_PATH}/nodes/:node_id"),
        get(stacks_nodes_list),
    )
}

/// List stacks
///
/// Lists all stacks for a specific node identified by its small ID.
#[utoipa::path(
    get,
    path = "/stacks/nodes/{node_id}",
    params(
        ("node_id" = i64, Path, description = "Node small ID"),
        ("min_fill_fraction" = Option<f64>, Query, description = "Optional minimum fill fraction (0.0 to 100.0)")
    ),
    responses(
        (status = OK, description = "List of Stack objects matching the criteria", body = Vec<Stack>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn stacks_nodes_list(
    State(daemon_state): State<DaemonState>,
    Path(node_id): Path<i64>,
    Query(query): Query<StackQuery>,
) -> Result<Json<Vec<Stack>>, StatusCode> {
    let node_ids = vec![node_id];

    let stacks = if let Some(fraction) = query.min_fill_fraction {
        daemon_state
            .atoma_state
            .get_almost_filled_stacks(&node_ids, fraction)
            .await
    } else {
        daemon_state
            .atoma_state
            .get_stacks_by_node_small_ids(&node_ids)
            .await
    };

    stacks.map(Json).map_err(|_| {
        error!("Failed to get stacks");
        StatusCode::INTERNAL_SERVER_ERROR
    })
}
