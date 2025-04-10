use atoma_state::types::StackAttestationDispute;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use tracing::error;
use utoipa::OpenApi;

use crate::DaemonState;

pub const ATTESTATION_DISPUTES_PATH: &str = "/attestation_disputes";

#[derive(OpenApi)]
#[openapi(
    paths(
        attestation_disputes_against_nodes_list,
        attestation_disputes_own_nodes_list
    ),
    components(schemas(StackAttestationDispute))
)]
pub struct AttestationDisputesOpenApi;

//TODO: this endpoint can be merged into one (I think) through filters

//TODO: this endpoint can be merged into one (I think) through filters

/// Router for handling attestation disputes endpoints
pub fn attestation_disputes_router() -> Router<DaemonState> {
    Router::new()
        .route(
            &format!("{ATTESTATION_DISPUTES_PATH}/against/nodes/{{node_id}}"),
            get(attestation_disputes_against_nodes_list),
        )
        .route(
            &format!("{ATTESTATION_DISPUTES_PATH}/own/nodes/{{node_id}}"),
            get(attestation_disputes_own_nodes_list),
        )
}

/// List against attestation disputes
///
/// Lists all attestation disputes against a specific node.
#[utoipa::path(
    get,
    path = "/against/nodes/{node_id}",
    params(
        ("node_id" = i64, Path, description = "The small ID of the node whose disputes should be retrieved")
    ),
    responses(
        (status = OK, description = "List of against attestation disputes where the specified node is the defendant", body = Vec<StackAttestationDispute>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn attestation_disputes_against_nodes_list(
    State(daemon_state): State<DaemonState>,
    Path(node_small_id): Path<i64>,
) -> Result<Json<Vec<StackAttestationDispute>>, StatusCode> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_against_attestation_disputes(&[node_small_id])
            .await
            .map_err(|_| {
                error!("Failed to get against attestation dispute");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// List own attestation disputes
///
/// Lists all attestation disputes initiated by a specific node.
#[utoipa::path(
    get,
    path = "/own/nodes/{node_id}",
    params(
        ("node_id" = i64, Path, description = "The small ID of the node whose initiated disputes should be retrieved")
    ),
    responses(
        (status = OK, description = "List of own attestation disputes where the specified node is the plaintiff", body = Vec<StackAttestationDispute>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn attestation_disputes_own_nodes_list(
    State(daemon_state): State<DaemonState>,
    Path(node_small_id): Path<i64>,
) -> Result<Json<Vec<StackAttestationDispute>>, StatusCode> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_own_attestation_disputes(&[node_small_id])
            .await
            .map_err(|_| {
                error!("Failed to get own attestation dispute");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}
