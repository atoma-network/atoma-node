use atoma_utils::constants;
use axum::{extract::State, Json};
use hyper::HeaderMap;
use tracing::instrument;

use crate::{error::AtomaServiceError, server::AppState};

/// Stop a streamer
///
/// This endpoint is used to stop a streamer.
///
/// # Errors
///
/// Returns a `AtomaServiceError::MissingHeader` if the request ID header is missing.
/// Returns a `AtomaServiceError::InvalidHeader` if the request ID header is invalid.
#[utoipa::path(
    post,
    path = "/v1/stop-streamer",
    responses(
        (status = 200, description = "OK", body = String),
        (status = 400, description = "Bad Request, missing or invalid request ID header")
    )
)]
#[instrument(level = "info", skip_all, fields(path = "/v1/stop-streamer"), err)]
pub async fn stop_streamer_handler(
    State(app_state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<String>, AtomaServiceError> {
    let request_id = headers
        .get(constants::REQUEST_ID)
        .ok_or_else(|| AtomaServiceError::MissingHeader {
            header: constants::REQUEST_ID.to_string(),
            endpoint: "/v1/stop-streamer".to_string(),
        })?
        .to_str()
        .map_err(|_| AtomaServiceError::InvalidHeader {
            message: "Invalid request ID".to_string(),
            endpoint: "/v1/stop-streamer".to_string(),
        })?;

    app_state
        .client_dropped_streamer_connections
        .insert(request_id.to_string());

    Ok(Json("OK".to_string()))
}
