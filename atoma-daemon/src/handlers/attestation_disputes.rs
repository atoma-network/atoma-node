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
        attestation_disputes_against_list,
        attestation_disputes_against_get,
        attestation_disputes_own_list,
        attestation_disputes_own_get
    ),
    components(schemas(StackAttestationDispute))
)]
pub(crate) struct AttestationDisputesOpenApi;

//TODO: this endpoint can be merged into one (I think) through filters

/// Router for handling attestation disputes endpoints
///
/// Creates routes for:
/// - GET /attestation_disputes/against - Get all attestation disputes against the registered nodes
/// - GET /attestation_disputes/against/:id - Get attestation disputes against a specific node
/// - GET /attestation_disputes/own - Get all attestation disputes initiated by the registered nodes
/// - GET /attestation_disputes/own/:id - Get attestation disputes initiated by a specific node
pub fn attestation_disputes_router() -> Router<DaemonState> {
    Router::new()
        .route(
            &format!("{ATTESTATION_DISPUTES_PATH}/against"),
            get(attestation_disputes_against_list),
        )
        .route(
            &format!("{ATTESTATION_DISPUTES_PATH}/against/:id"),
            get(attestation_disputes_against_get),
        )
        .route(
            &format!("{ATTESTATION_DISPUTES_PATH}/own"),
            get(attestation_disputes_own_list),
        )
        .route(
            &format!("{ATTESTATION_DISPUTES_PATH}/own/:id"),
            get(attestation_disputes_own_get),
        )
}

/// List attestation disputes against currently registered nodes
///
/// Retrieves all attestation disputes against the currently registered nodes.
#[utoipa::path(
    get,
    path = "/against",
    responses(
        (status = OK, description = "List of all against attestation disputes where the registered nodes are the defendants", body = Vec<StackAttestationDispute>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn attestation_disputes_against_list(
    State(daemon_state): State<DaemonState>,
) -> Result<Json<Vec<StackAttestationDispute>>, StatusCode> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_against_attestation_disputes(
                &daemon_state
                    .node_badges
                    .iter()
                    .map(|(_, small_id)| *small_id as i64)
                    .collect::<Vec<_>>(),
            )
            .await
            .map_err(|_| {
                error!("Failed to get all attestation disputes");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// List attestation disputes against a specific node
///
/// Retrieves all attestation disputes against a specific node.
#[utoipa::path(
    get,
    path = "/against/{id}",
    params(
        ("id" = i64, Path, description = "The small ID of the node whose disputes should be retrieved")
    ),
    responses(
        (status = OK, description = "List of against attestation disputes where the specified node is the defendant", body = Vec<StackAttestationDispute>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn attestation_disputes_against_get(
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

/// List attestation disputes initiated by currently registered nodes
///
/// Retrieves all attestation disputes initiated by the currently registered nodes.
#[utoipa::path(
    get,
    path = "/own",
    responses(
        (status = OK, description = "List of all own attestation disputes where the registered nodes are the plaintiffs", body = Vec<StackAttestationDispute>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn attestation_disputes_own_list(
    State(daemon_state): State<DaemonState>,
) -> Result<Json<Vec<StackAttestationDispute>>, StatusCode> {
    Ok(Json(
        daemon_state
            .atoma_state
            .get_own_attestation_disputes(
                &daemon_state
                    .node_badges
                    .iter()
                    .map(|(_, small_id)| *small_id as i64)
                    .collect::<Vec<_>>(),
            )
            .await
            .map_err(|_| {
                error!("Failed to get all own attestation disputes");
                StatusCode::INTERNAL_SERVER_ERROR
            })?,
    ))
}

/// List attestation disputes initiated by a specific node
///
/// Retrieves all attestation disputes initiated by a specific node.
#[utoipa::path(
    get,
    path = "/own/{id}",
    params(
        ("id" = i64, Path, description = "The small ID of the node whose initiated disputes should be retrieved")
    ),
    responses(
        (status = OK, description = "List of own attestation disputes where the specified node is the plaintiff", body = Vec<StackAttestationDispute>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn attestation_disputes_own_get(
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
