use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use sqlx::Row;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeComputeRequest {
    pub node_id: u64,
    pub stack_small_id: i64,
    pub requested_num_compute_units: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackLeaderResponse {
    pub can_proceed: bool,
}

pub struct StackLeader {
    db: PgPool,
}

impl StackLeader {
    /// Creates a new `StackLeaderResponse` based on the compute unit request and available capacity
    ///
    /// # Arguments
    /// * `request` - The `NodeComputeRequest` containing requested compute units
    /// * `available_compute_units` - Total remaining compute units available in the stack
    ///
    /// # Returns
    /// Returns a `StackLeaderResponse` indicating whether the request can proceed based on available capacity
    #[must_use]
    pub const fn new(db: PgPool) -> Self {
        Self { db }
    }

    /// Retrieves the available compute units for a specific stack from the database
    ///
    /// # Arguments
    /// * `db` - Database connection or reference
    /// * `stack_small_id` - The unique identifier for the stack
    ///
    /// # Returns
    /// Result containing the available compute units or a database error
    ///
    /// # Errors
    /// Returns a `sqlx::Error` if the database query fails or if the stack is not found
    #[allow(clippy::cast_sign_loss)]
    pub async fn get_stack_available_compute_units(
        &self,
        node_compute_request: &NodeComputeRequest,
    ) -> Result<u64, sqlx::Error> {
        // Query the database for the available compute units for this stack
        let mut tx = self.db.begin().await?;

        let row =
            sqlx::query("SELECT available_compute_units FROM stacks WHERE stack_small_id = $1")
                .bind(node_compute_request.stack_small_id)
                .fetch_one(&mut *tx)
                .await?;

        let compute_units_i64: i64 = row.get("available_compute_units");
        let available_compute_units = compute_units_i64 as u64;
        tx.commit().await?;

        Ok(available_compute_units)
    }

    pub async fn can_proceed(
        &self,
        node_compute_request: &NodeComputeRequest,
    ) -> StackLeaderResponse {
        self.get_stack_available_compute_units(node_compute_request)
            .await
            .map_or_else(
                |_| StackLeaderResponse { can_proceed: false },
                |available_compute_units| StackLeaderResponse {
                    can_proceed: available_compute_units
                        >= node_compute_request.requested_num_compute_units,
                },
            )
    }
}
