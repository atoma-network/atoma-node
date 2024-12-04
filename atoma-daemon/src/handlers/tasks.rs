use atoma_state::types::Task;
use axum::{extract::State, http::StatusCode, routing::get, Json, Router};
use tracing::error;
use utoipa::OpenApi;

use crate::DaemonState;

pub const TASKS_PATH: &str = "/tasks";

#[derive(OpenApi)]
#[openapi(paths(tasks_list), components(schemas(Task)))]
pub(crate) struct TasksOpenApi;

/// Router for handling task-related endpoints
///
/// This function sets up the routing for various task-related operations,
/// including listing all tasks. Each route corresponds to a specific
/// operation that can be performed on tasks within the system.
pub fn tasks_router() -> Router<DaemonState> {
    Router::new().route(TASKS_PATH, get(tasks_list))
}

/// Retrieves all tasks from the state manager.
#[utoipa::path(
    get,
    path = "/",
    responses(
        (status = OK, description = "List of all tasks", body = Vec<Task>),
        (status = INTERNAL_SERVER_ERROR, description = "Internal server error")
    )
)]
pub async fn tasks_list(
    State(daemon_state): State<DaemonState>,
) -> Result<Json<Vec<Task>>, StatusCode> {
    daemon_state
        .atoma_state
        .get_all_tasks()
        .await
        .map_err(|_| {
            error!("Failed to get all tasks");
            StatusCode::INTERNAL_SERVER_ERROR
        })
        .map(Json)
}
