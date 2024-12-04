use atoma_state::types::Stack;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use tracing::error;
use utoipa::OpenApi;

use crate::DaemonState;

pub const ALMOST_FILLED_STACKS_PATH: &str = "/almost_filled_stacks";

#[derive(OpenApi)]
#[openapi(
    paths(almost_filled_stacks_fraction_get, almost_filled_stacks_get),
    components(schemas(Stack))
)]
pub(crate) struct AlmostFilledStacksOpenApi;

/// Router for handling almost filled stacks endpoints
///
/// Creates routes for:
/// - GET /almost_filled_stacks - Get stacks above threshold for all nodes
/// - GET /almost_filled_stacks/:id - Get stacks above threshold for specific node
pub fn almost_filled_stacks_router() -> Router<DaemonState> {
    Router::new()
        .route(
            ALMOST_FILLED_STACKS_PATH,
            get(almost_filled_stacks_fraction_get),
        )
        .route(
            &format!("{ALMOST_FILLED_STACKS_PATH}/:id"),
            get(almost_filled_stacks_get),
        )
}

/// Retrieves all stacks that are filled above a specified fraction threshold for all registered nodes.
#[utoipa::path(
    get,
    path = "/{fraction}",
    params(
        ("fraction" = f64, Path, description = "The fraction threshold (0.0 to 100.0) to filter stacks")
    ),
    responses(
        (status = 200, description = "Stack objects that are filled above the specified fraction threshold", body = Vec<Stack>),
        (status = 500, description = "Internal server error")
    )
)]
pub async fn almost_filled_stacks_fraction_get(
    State(daemon_state): State<DaemonState>,
    Path(fraction): Path<f64>,
) -> Result<Json<Vec<Stack>>, StatusCode> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_almost_filled_stacks(
                &daemon_state
                    .node_badges
                    .iter()
                    .map(|(_, small_id)| *small_id as i64)
                    .collect::<Vec<_>>(),
                fraction,
            )
            .await
            .map_err(|_| {
                error!("Failed to get all almost filled stacks");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// Retrieves all stacks that are filled above a specified fraction threshold for a specific node.
///
/// # Arguments
/// * `daemon_state` - The shared state containing the state manager
/// * `node_small_id` - The small ID of the node whose stacks should be retrieved
/// * `fraction` - The fraction threshold (0.0 to 100.0) to filter stacks
///
/// # Returns
/// * `Result<Json<Vec<Stack>>>` - A JSON response containing a list of stacks
///   - `Ok(Json<Vec<Stack>>)` - Successfully retrieved stacks
///   - `Err(StatusCode::INTERNAL_SERVER_ERROR)` - Failed to retrieve stacks from state manager
///
/// # Example Response
/// Returns a JSON array of Stack objects for the specified node that are filled above the fraction threshold
#[utoipa::path(
    get,
    path = "/{id}/{fraction}",
    params(
        ("id" = i64, Path, description = "The small ID of the node whose stacks should be retrieved"),
        ("fraction" = f64, Path, description = "The fraction threshold (0.0 to 100.0) to filter stacks")
    ),
    responses(
        (status = 200, description = "List of node stacks", body = Vec<Stack>),
        (status = 500, description = "Internal server error")
    )
)]
pub async fn almost_filled_stacks_get(
    State(daemon_state): State<DaemonState>,
    Path((node_small_id, fraction)): Path<(i64, f64)>,
) -> Result<Json<Vec<Stack>>, StatusCode> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_almost_filled_stacks(&[node_small_id], fraction)
            .await
            .map_err(|_| {
                error!("Failed to get node almost filled stacks");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}
