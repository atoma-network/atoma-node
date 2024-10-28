pub mod config;
pub mod state_manager;
pub mod types;

pub use config::StateManagerConfig;
use sqlx::Sqlite;
pub use sqlx::SqlitePool;
pub use state_manager::{StateManager, StateManagerError};

/// Builds a query with an IN clause and optional additional conditions
///
/// # Arguments
/// * `base_query` - The base SQL query to build upon
/// * `column` - The column name to use in the IN clause
/// * `values` - The array of values to include in the IN clause
/// * `additional_conditions` - Optional additional WHERE conditions to add after the IN clause
///
/// # Returns
/// A QueryBuilder configured with the IN clause and ready for additional bindings
pub(crate) fn build_query_with_in<'a, T: sqlx::Type<Sqlite> + sqlx::Encode<'a, Sqlite>>(
    base_query: &str,
    column: &str,
    values: &'a [T],
    additional_conditions: Option<&str>,
) -> sqlx::QueryBuilder<'a, Sqlite> {
    let mut builder = sqlx::QueryBuilder::new(base_query);

    if values.is_empty() {
        builder.push(" WHERE 1=0");
        return builder;
    }

    builder.push(" WHERE ");
    builder.push(column);
    builder.push(" IN (");

    // Create placeholders for the IN clause
    let mut separated = builder.separated(", ");
    for value in values {
        separated.push_bind(value);
    }
    separated.push_unseparated(")");

    // Add additional conditions if present
    if let Some(conditions) = additional_conditions {
        builder.push(" AND ");
        builder.push(conditions);
    }

    builder
}
