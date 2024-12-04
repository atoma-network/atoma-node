use atoma_state::types::{Stack, StackSettlementTicket};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use tracing::error;
use utoipa::OpenApi;

use crate::DaemonState;

pub const STACKS_PATH: &str = "/stacks";

#[derive(OpenApi)]
#[openapi(
    paths(stacks_list, stacks_get, claimed_stacks_list, claimed_stacks_get),
    components(schemas(Stack, StackSettlementTicket))
)]
pub(crate) struct StacksOpenApi;

/// Router for handling stack-related endpoints
///
/// This function sets up the routing for various stack-related operations,
/// including listing all stacks, retrieving specific stacks by ID, listing claimed stacks,
/// and retrieving specific claimed stacks by ID. Each route corresponds to a specific
/// operation that can be performed on stacks within the system.
pub fn stacks_router() -> Router<DaemonState> {
    Router::new()
        .route(STACKS_PATH, get(stacks_list))
        .route(&format!("{STACKS_PATH}/:id"), get(stacks_get))
        .route("/claimed_stacks", get(claimed_stacks_list))
        .route("/claimed_stacks/:id", get(claimed_stacks_get))
}

/// Retrieves all stacks associated with the currently registered node badges.
#[utoipa::path(
    get,
    path = "/stacks",
    responses(
        (status = OK, description = "List of all Stack objects for all registered nodes", body = Vec<Stack>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn stacks_list(
    State(daemon_state): State<DaemonState>,
) -> Result<Json<Vec<Stack>>, StatusCode> {
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
#[utoipa::path(
    get,
    path = "/stacks/{id}",
    params(
        ("id" = i64, Path, description = "The small ID of the node whose stacks should be retrieved")
    ),
    responses(
        (status = OK, description = "List of Stack objects for the specified node", body = Vec<Stack>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn stacks_get(
    State(daemon_state): State<DaemonState>,
    Path(node_small_id): Path<i64>,
) -> Result<Json<Vec<Stack>>, StatusCode> {
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

/// Retrieves all claimed stacks for the currently registered node badges.
#[utoipa::path(
    get,
    path = "/claimed_stacks",
    responses(
        (status = OK, description = "List of all StackSettlementTicket objects for all registered nodes", body = Vec<StackSettlementTicket>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn claimed_stacks_list(
    State(daemon_state): State<DaemonState>,
) -> Result<Json<Vec<StackSettlementTicket>>, StatusCode> {
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
#[utoipa::path(
    get,
    path = "/claimed_stacks/{id}",
    params(
        ("id" = i64, Path, description = "The small ID of the node whose claimed stacks should be retrieved")
    ),
    responses(
        (status = OK, description = "List of StackSettlementTicket objects for the specified node", body = Vec<StackSettlementTicket>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn claimed_stacks_get(
    State(daemon_state): State<DaemonState>,
    Path(node_small_id): Path<i64>,
) -> Result<Json<Vec<StackSettlementTicket>>, StatusCode> {
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
