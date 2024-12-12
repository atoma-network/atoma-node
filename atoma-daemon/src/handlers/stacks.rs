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
    node_id: Option<i64>,
    min_fill_fraction: Option<f64>,
}

#[derive(Deserialize, ToSchema)]
pub struct ClaimedStackQuery {
    node_id: Option<i64>,
}

#[derive(OpenApi)]
#[openapi(
    paths(get_stacks, get_claimed_stacks),
    components(schemas(Stack, StackSettlementTicket, StackQuery, ClaimedStackQuery))
)]
pub(crate) struct StacksOpenApi;

pub fn stacks_router() -> Router<DaemonState> {
    Router::new()
        .route(STACKS_PATH, get(get_stacks))
        .route("/claimed_stacks", get(get_claimed_stacks))
}

/// Get stacks with optional filtering
///
/// Retrieves stacks with optional filtering by node ID and minimum fill fraction.
#[utoipa::path(
    get,
    path = "/stacks",
    params(
        ("node_id" = Option<i64>, Query, description = "Optional node ID to filter by"),
        ("min_fill_fraction" = Option<f64>, Query, description = "Optional minimum fill fraction (0.0 to 100.0)")
    ),
    responses(
        (status = OK, description = "List of Stack objects matching the criteria", body = Vec<Stack>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn get_stacks(
    State(daemon_state): State<DaemonState>,
    Query(query): Query<StackQuery>,
) -> Result<Json<Vec<Stack>>, StatusCode> {
    let node_ids = match query.node_id {
        Some(id) => vec![id],
        None => daemon_state
            .node_badges
            .iter()
            .map(|(_, small_id)| *small_id as i64)
            .collect(),
    };

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

/// Get claimed stacks with optional filtering
#[utoipa::path(
    get,
    path = "/claimed_stacks",
    params(
        ("node_id" = Option<i64>, Query, description = "Optional node ID to filter by")
    ),
    responses(
        (status = OK, description = "List of claimed stacks matching the criteria", body = Vec<StackSettlementTicket>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn get_claimed_stacks(
    State(daemon_state): State<DaemonState>,
    Query(query): Query<ClaimedStackQuery>,
) -> Result<Json<Vec<StackSettlementTicket>>, StatusCode> {
    let node_ids = match query.node_id {
        Some(id) => vec![id],
        None => daemon_state
            .node_badges
            .iter()
            .map(|(_, small_id)| *small_id as i64)
            .collect(),
    };

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
