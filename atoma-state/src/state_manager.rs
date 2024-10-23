use crate::types::{NodeSubscription, Stack, StackAttestationDispute, StackSettlementTicket, Task};

use sqlx::{pool::PoolConnection, Sqlite, SqlitePool};
use sqlx::{FromRow, Row};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, StateManagerError>;

/// StateManager is a wrapper around a SQLite connection pool, responsible for managing the state of the Atoma system.
///
/// It provides an interface to interact with the SQLite database, handling operations
/// related to tasks, node subscriptions, stacks, and various other system components.
pub struct StateManager {
    /// The SQLite connection pool used for database operations.
    pub db: SqlitePool,
}

impl StateManager {
    /// Constructor
    pub fn new(db: SqlitePool) -> Self {
        Self { db }
    }

    /// Creates a new `StateManager` instance from a database URL.
    ///
    /// This method establishes a connection to the SQLite database using the provided URL,
    /// creates all necessary tables in the database, and returns a new `StateManager` instance.
    pub async fn new_from_url(database_url: String) -> Result<Self> {
        let db = SqlitePool::connect(&database_url).await?;
        queries::create_all_tables(&db).await?;
        Ok(Self { db })
    }

    /// Acquires a connection from the SQLite connection pool.
    ///
    /// This method is used to obtain a connection from the pool for database operations.
    ///
    /// # Returns
    ///
    /// - `Result<PoolConnection<Sqlite>>`: A result containing either:
    ///   - `Ok(PoolConnection<Sqlite>)`: A successful connection from the pool.
    ///   - `Err(StateManagerError)`: An error if the connection acquisition fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if the connection pool is unable to
    /// provide a connection, which could happen due to various reasons such as
    /// connection timeouts or pool exhaustion.
    #[tracing::instrument(level = "debug", skip(self), fields(function = "get_connection"))]
    pub async fn get_connection(&self) -> Result<PoolConnection<Sqlite>> {
        let conn = self.db.acquire().await?;
        Ok(conn)
    }

    /// Get a task by its unique identifier.
    ///
    /// This method fetches a task from the database based on the provided `task_small_id`.
    ///
    /// # Arguments
    ///
    /// * `task_small_id` - The unique identifier for the task to be fetched.
    ///
    /// # Returns
    ///
    /// - `Result<Task>`: A result containing either:
    ///   - `Ok(Task)`: The task with the specified `task_id`.
    ///   - `Err(StateManagerError)`: An error if the task is not found or other database operation fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if the database query fails.
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(task_small_id = %task_small_id)
    )]
    pub async fn get_task_by_small_id(&self, task_small_id: i64) -> Result<Task> {
        let task = sqlx::query("SELECT * FROM tasks WHERE task_small_id = ?")
            .bind(task_small_id)
            .fetch_one(&self.db)
            .await?;
        Ok(Task::from_row(&task)?)
    }

    /// Inserts a new task into the database.
    ///
    /// This method takes a `Task` object and inserts its data into the `tasks` table
    /// in the database.
    ///
    /// # Arguments
    ///
    /// * `task` - A `Task` struct containing all the information about the task to be inserted.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(StateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's a constraint violation (e.g., duplicate `task_small_id` or `task_id`).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::{StateManager, Task};
    ///
    /// async fn add_task(state_manager: &mut StateManager, task: Task) -> Result<(), StateManagerError> {
    ///     state_manager.insert_new_task(task).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(task_id = %task.task_small_id)
    )]
    pub async fn insert_new_task(&self, task: Task) -> Result<()> {
        sqlx::query(
            "INSERT INTO tasks (
                task_small_id, task_id, role, model_name, is_deprecated,
                valid_until_epoch, deprecated_at_epoch, optimizations,
                security_level, task_metrics_compute_unit,
                task_metrics_time_unit, task_metrics_value,
                minimum_reputation_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(task.task_small_id)
        .bind(task.task_id)
        .bind(task.role)
        .bind(task.model_name)
        .bind(task.is_deprecated)
        .bind(task.valid_until_epoch)
        .bind(task.deprecated_at_epoch)
        .bind(task.optimizations)
        .bind(task.security_level)
        .bind(task.task_metrics_compute_unit)
        .bind(task.task_metrics_time_unit)
        .bind(task.task_metrics_value)
        .bind(task.minimum_reputation_score)
        .execute(&self.db)
        .await?;
        Ok(())
    }

    /// Deprecates a task in the database based on its small ID.
    ///
    /// This method updates the `is_deprecated` field of a task to `TRUE` in the `tasks` table
    /// using the provided `task_small_id`.
    ///
    /// # Arguments
    ///
    /// * `task_small_id` - The unique small identifier for the task to be deprecated.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(StateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - The task with the specified `task_small_id` doesn't exist.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::StateManager;
    ///
    /// async fn deprecate_task(state_manager: &StateManager, task_small_id: i64) -> Result<(), StateManagerError> {
    ///     state_manager.deprecate_task(task_small_id).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(task_small_id = %task_small_id)
    )]
    pub async fn deprecate_task(&self, task_small_id: i64, epoch: i64) -> Result<()> {
        sqlx::query("UPDATE tasks SET is_deprecated = TRUE, deprecated_at_epoch = ? WHERE task_small_id = ?")
            .bind(epoch)
            .bind(task_small_id)
            .execute(&self.db)
            .await?;
        Ok(())
    }

    /// Retrieves all tasks subscribed to by a specific node.
    ///
    /// This method fetches all tasks from the database that are associated with
    /// the given node through the `node_subscriptions` table.
    ///
    /// # Arguments
    ///
    /// * `node_small_id` - The unique identifier of the node whose subscribed tasks are to be fetched.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<Task>>`: A result containing either:
    ///   - `Ok(Vec<Task>)`: A vector of `Task` objects representing all tasks subscribed to by the node.
    ///   - `Err(StateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into `Task` objects.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::StateManager;
    ///
    /// async fn get_node_tasks(state_manager: &StateManager, node_small_id: i64) -> Result<Vec<Task>, StateManagerError> {
    ///     state_manager.get_subscribed_tasks(node_small_id).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(node_small_id = %node_small_id)
    )]
    pub async fn get_subscribed_tasks(&self, node_small_id: i64) -> Result<Vec<Task>> {
        let tasks = sqlx::query(
            "SELECT tasks.* FROM tasks
            INNER JOIN node_subscriptions ON tasks.task_small_id = node_subscriptions.task_small_id
            WHERE node_subscriptions.node_small_id = ?",
        )
        .bind(node_small_id)
        .fetch_all(&self.db)
        .await?;
        tasks
            .into_iter()
            .map(|task| Task::from_row(&task).map_err(StateManagerError::from))
            .collect()
    }

    /// Checks if a node is subscribed to a specific task.
    ///
    /// This method queries the `node_subscriptions` table to determine if there's
    /// an entry for the given node and task combination.
    ///
    /// # Arguments
    ///
    /// * `node_small_id` - The unique identifier of the node.
    /// * `task_small_id` - The unique identifier of the task.
    ///
    /// # Returns
    ///
    /// - `Result<bool>`: A result containing either:
    ///   - `Ok(true)` if the node is subscribed to the task.
    ///   - `Ok(false)` if the node is not subscribed to the task.
    ///   - `Err(StateManagerError)` if there's a database error.
    ///
    /// # Errors
    ///
    /// This function will return an error if the database query fails to execute.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::StateManager;
    ///
    /// async fn check_subscription(state_manager: &StateManager, node_small_id: i64, task_small_id: i64) -> Result<bool, StateManagerError> {
    ///     state_manager.is_node_subscribed_to_task(node_small_id, task_small_id).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(node_small_id = %node_small_id, task_small_id = %task_small_id)
    )]
    pub async fn is_node_subscribed_to_task(
        &self,
        node_small_id: i64,
        task_small_id: i64,
    ) -> Result<bool> {
        let result = sqlx::query(
            "SELECT COUNT(*) FROM node_subscriptions WHERE node_small_id = ? AND task_small_id = ?",
        )
        .bind(node_small_id)
        .bind(task_small_id)
        .fetch_one(&self.db)
        .await?;
        let count: i64 = result.get(0);
        Ok(count > 0)
    }

    /// Subscribes a node to a task with a specified price per compute unit.
    ///
    /// This method inserts a new entry into the `node_subscriptions` table to
    /// establish a subscription relationship between a node and a task, along
    /// with the specified price per compute unit for the subscription.
    ///
    /// # Arguments
    ///
    /// * `node_small_id` - The unique identifier of the node to be subscribed.
    /// * `task_small_id` - The unique identifier of the task to which the node is subscribing.
    /// * `price_per_compute_unit` - The price per compute unit for the subscription.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(StateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's a constraint violation (e.g., duplicate subscription).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::StateManager;
    ///
    /// async fn subscribe_node_to_task(state_manager: &StateManager, node_small_id: i64, task_small_id: i64, price_per_compute_unit: i64) -> Result<(), StateManagerError> {
    ///     state_manager.subscribe_node_to_task(node_small_id, task_small_id, price_per_compute_unit).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(
            node_small_id = %node_small_id,
            task_small_id = %task_small_id,
            price_per_compute_unit = %price_per_compute_unit,
            max_num_compute_units = %max_num_compute_units
        )
    )]
    pub async fn subscribe_node_to_task(
        &self,
        node_small_id: i64,
        task_small_id: i64,
        price_per_compute_unit: i64,
        max_num_compute_units: i64,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO node_subscriptions 
                (node_small_id, task_small_id, price_per_compute_unit, max_num_compute_units, valid) 
                VALUES (?, ?, ?, ?, TRUE)",
        )
            .bind(node_small_id)
            .bind(task_small_id)
            .bind(price_per_compute_unit)
            .bind(max_num_compute_units)
            .execute(&self.db)
            .await?;
        Ok(())
    }

    /// Retrieves the node subscription associated with a specific task ID.
    ///
    /// This method fetches the node subscription details from the `node_subscriptions` table
    /// based on the provided `task_small_id`.
    ///
    /// # Arguments
    ///
    /// * `task_small_id` - The unique small identifier of the task to retrieve the subscription for.
    ///
    /// # Returns
    ///
    /// - `Result<NodeSubscription>`: A result containing either:
    ///   - `Ok(NodeSubscription)`: A `NodeSubscription` object representing the subscription details.
    ///   - `Err(StateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database row into a `NodeSubscription` object.
    /// - No subscription is found for the given `task_small_id`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::{StateManager, NodeSubscription};
    ///
    /// async fn get_subscription(state_manager: &StateManager, task_small_id: i64) -> Result<NodeSubscription, StateManagerError> {
    ///     state_manager.get_node_subscription_by_task_small_id(task_small_id).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(task_small_id = %task_small_id)
    )]
    pub async fn get_node_subscription_by_task_small_id(
        &self,
        task_small_id: i64,
    ) -> Result<NodeSubscription> {
        let subscription = sqlx::query("SELECT * FROM node_subscriptions WHERE task_small_id = ?")
            .bind(task_small_id)
            .fetch_one(&self.db)
            .await?;
        Ok(NodeSubscription::from_row(&subscription)?)
    }

    /// Updates an existing node subscription to a task with new price and compute unit values.
    ///
    /// This method updates an entry in the `node_subscriptions` table, modifying the
    /// price per compute unit and the maximum number of compute units for an existing
    /// subscription between a node and a task.
    ///
    /// # Arguments
    ///
    /// * `node_small_id` - The unique identifier of the subscribed node.
    /// * `task_small_id` - The unique identifier of the task to which the node is subscribed.
    /// * `price_per_compute_unit` - The new price per compute unit for the subscription.
    /// * `max_num_compute_units` - The new maximum number of compute units for the subscription.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(StateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - The specified node subscription doesn't exist.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::StateManager;
    ///
    /// async fn update_subscription(state_manager: &StateManager) -> Result<(), StateManagerError> {
    ///     state_manager.update_node_subscription(1, 2, 100, 1000).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(
            node_small_id = %node_small_id,
            task_small_id = %task_small_id,
            price_per_compute_unit = %price_per_compute_unit,
            max_num_compute_units = %max_num_compute_units
        )
    )]
    pub async fn update_node_subscription(
        &self,
        node_small_id: i64,
        task_small_id: i64,
        price_per_compute_unit: i64,
        max_num_compute_units: i64,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE node_subscriptions SET price_per_compute_unit = ?, max_num_compute_units = ? WHERE node_small_id = ? AND task_small_id = ?",
        )
            .bind(price_per_compute_unit)
            .bind(max_num_compute_units)
            .bind(node_small_id)
            .bind(task_small_id)
            .execute(&self.db)
            .await?;
        Ok(())
    }

    /// Unsubscribes a node from a task.
    ///
    /// This method updates the `valid` field of the `node_subscriptions` table to `FALSE` for the specified node and task combination.
    ///
    /// # Arguments
    ///
    /// * `node_small_id` - The unique identifier of the node to be unsubscribed.
    /// * `task_small_id` - The unique identifier of the task from which the node is unsubscribed.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(StateManagerError)).
    /// # Errors
    ///
    /// This function will return an error if the database query fails to execute.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::StateManager;
    ///
    /// async fn unsubscribe_node(state_manager: &StateManager, node_small_id: i64, task_small_id: i64) -> Result<(), StateManagerError> {
    ///     state_manager.unsubscribe_node_from_task(node_small_id, task_small_id).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(node_small_id = %node_small_id, task_small_id = %task_small_id)
    )]
    pub async fn unsubscribe_node_from_task(
        &self,
        node_small_id: i64,
        task_small_id: i64,
    ) -> Result<()> {
        sqlx::query("UPDATE node_subscriptions SET valid = FALSE WHERE node_small_id = ? AND task_small_id = ?")
            .bind(node_small_id)
            .bind(task_small_id)
            .execute(&self.db)
            .await?;
        Ok(())
    }

    /// Retrieves the stack associated with a specific stack ID.
    ///
    /// This method fetches the stack details from the `stacks` table based on the provided `stack_id`.
    ///
    /// # Arguments
    ///
    /// * `stack_small_id` - The unique small identifier of the stack to retrieve.
    ///
    /// # Returns
    ///
    /// - `Result<Stack>`: A result containing either:
    ///   - `Ok(Stack)`: A `Stack` object representing the stack.
    ///   - `Err(StateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into a `Stack` object.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::StateManager;
    ///
    /// async fn get_stack(state_manager: &StateManager, stack_small_id: i64) -> Result<Stack, StateManagerError> {  
    ///     state_manager.get_stack(stack_id).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack_small_id)
    )]
    pub async fn get_stack(&self, stack_small_id: i64) -> Result<Stack> {
        let stack = sqlx::query("SELECT * FROM stacks WHERE stack_small_id = ?")
            .bind(stack_small_id)
            .fetch_one(&self.db)
            .await?;
        Ok(Stack::from_row(&stack)?)
    }

    /// Retrieves and updates an available stack with the specified number of compute units.
    ///
    /// This method attempts to reserve a specified number of compute units from a stack
    /// owned by the given public key. It performs this operation atomically using a database
    /// transaction to ensure consistency.
    ///
    /// # Arguments
    ///
    /// * `stack_small_id` - The unique identifier of the stack.
    /// * `public_key` - The public key of the stack owner.
    /// * `num_compute_units` - The number of compute units to reserve.
    ///
    /// # Returns
    ///
    /// - `Result<Option<Stack>>`: A result containing either:
    ///   - `Ok(Some(Stack))`: If the stack was successfully updated, returns the updated stack.
    ///   - `Ok(None)`: If the stack couldn't be updated (e.g., insufficient compute units or stack in settle period).
    ///   - `Err(StateManagerError)`: If there was an error during the database operation.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database transaction fails to begin, execute, or commit.
    /// - There's an issue with the SQL query execution.
    ///
    /// # Details
    ///
    /// The function performs the following steps:
    /// 1. Begins a database transaction.
    /// 2. Attempts to update the stack by increasing the `already_computed_units` field.
    /// 3. If the update is successful (i.e., the stack has enough available compute units),
    ///    it fetches the updated stack information.
    /// 4. Commits the transaction.
    ///
    /// The update will only succeed if:
    /// - The stack belongs to the specified owner (public key).
    /// - The stack has enough remaining compute units.
    /// - The stack is not in the settle period.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::{StateManager, Stack};
    ///
    /// async fn reserve_compute_units(state_manager: &StateManager) -> Result<Option<Stack>, StateManagerError> {
    ///     let stack_small_id = 1;
    ///     let public_key = "owner_public_key";
    ///     let num_compute_units = 100;
    ///
    ///     state_manager.get_available_stack_with_compute_units(stack_small_id, public_key, num_compute_units).await
    /// }
    /// ```
    ///
    /// This function is particularly useful for atomically reserving compute units from a stack,
    /// ensuring that the operation is thread-safe and consistent even under concurrent access.
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(
            stack_small_id = %stack_small_id,
            public_key = %public_key,
            num_compute_units = %num_compute_units
        )
    )]
    pub async fn get_available_stack_with_compute_units(
        &self,
        stack_small_id: i64,
        public_key: &str,
        num_compute_units: i64,
    ) -> Result<Option<Stack>> {
        // First, begin a transaction to ensure atomicity
        let mut transaction = self.db.begin().await?;

        // First try to update the row
        let rows_affected = sqlx::query(
            r#"
            UPDATE stacks
            SET already_computed_units = already_computed_units + ?1
            WHERE stack_small_id = ?2
            AND owner_address = ?3
            AND num_compute_units - already_computed_units >= ?1
            AND in_settle_period = false
            "#,
        )
        .bind(num_compute_units)
        .bind(stack_small_id)
        .bind(public_key)
        .execute(&mut *transaction)
        .await?;

        // If update was successful, get the updated row
        let maybe_stack = if rows_affected.rows_affected() > 0 {
            sqlx::query_as::<_, Stack>(
                r#"
                SELECT * FROM stacks
                WHERE stack_small_id = ?1
                AND owner_address = ?2
                "#,
            )
            .bind(stack_small_id)
            .bind(public_key)
            .fetch_optional(&mut *transaction)
            .await?
        } else {
            None
        };

        // Commit the transaction
        transaction.commit().await?;

        Ok(maybe_stack)
    }

    /// Inserts a new stack into the database.
    ///
    /// This method inserts a new entry into the `stacks` table with the provided stack details.
    ///
    /// # Arguments
    ///
    /// * `stack` - The `Stack` object to be inserted into the database.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(StateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the provided `Stack` object into a database row.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::StateManager;
    ///
    /// async fn insert_stack(state_manager: &StateManager, stack: Stack) -> Result<(), StateManagerError> {
    ///     state_manager.insert_new_stack(stack).await
    /// }   
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack.stack_small_id,
            stack_id = %stack.stack_id,
            task_small_id = %stack.task_small_id,
            selected_node_id = %stack.selected_node_id,
            num_compute_units = %stack.num_compute_units,
            price = %stack.price)
    )]
    pub async fn insert_new_stack(&self, stack: Stack) -> Result<()> {
        sqlx::query(
            "INSERT INTO stacks 
                (owner_address, stack_small_id, stack_id, task_small_id, selected_node_id, num_compute_units, price, already_computed_units, in_settle_period) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
            .bind(stack.owner_address)
            .bind(stack.stack_small_id)
            .bind(stack.stack_id)
            .bind(stack.task_small_id)
            .bind(stack.selected_node_id)
            .bind(stack.num_compute_units)
            .bind(stack.price)
            .bind(stack.already_computed_units)
            .bind(stack.in_settle_period)
            .execute(&self.db)
            .await?;
        Ok(())
    }

    /// Updates the number of compute units already computed for a stack.
    ///
    /// This method updates the `already_computed_units` field in the `stacks` table
    /// for the specified `stack_small_id`.
    ///
    /// # Arguments
    ///
    /// * `stack_small_id` - The unique small identifier of the stack to update.
    /// * `already_computed_units` - The number of compute units already computed.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(StateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into a `Stack` object.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::StateManager;
    ///
    /// async fn update_computed_units(state_manager: &StateManager, stack_small_id: i64, already_computed_units: i64) -> Result<(), StateManagerError> {
    ///     state_manager.update_computed_units_for_stack(stack_small_id, already_computed_units).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack_small_id, already_computed_units = %already_computed_units)
    )]
    pub async fn update_computed_units_for_stack(
        &self,
        stack_small_id: i64,
        already_computed_units: i64,
    ) -> Result<()> {
        sqlx::query("UPDATE stacks SET already_computed_units = ? WHERE stack_small_id = ?")
            .bind(already_computed_units)
            .bind(stack_small_id)
            .execute(&self.db)
            .await?;
        Ok(())
    }

    /// Updates the number of tokens already computed for a stack.
    ///
    /// This method updates the `already_computed_units` field in the `stacks` table
    /// for the specified `stack_small_id`.
    ///
    /// # Arguments
    ///
    /// * `stack_small_id` - The unique small identifier of the stack to update.
    /// * `estimated_total_tokens` - The estimated total number of tokens.
    /// * `total_tokens` - The total number of tokens.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(StateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::StateManager;
    ///
    /// async fn update_stack_num_tokens(state_manager: &StateManager, stack_small_id: i64, estimated_total_tokens: i64, total_tokens: i64) -> Result<(), StateManagerError> {
    ///     state_manager.update_stack_num_tokens(stack_small_id, estimated_total_tokens, total_tokens).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack_small_id, estimated_total_tokens = %estimated_total_tokens, total_tokens = %total_tokens)
    )]
    pub async fn update_stack_num_tokens(
        &self,
        stack_small_id: i64,
        estimated_total_tokens: i64,
        total_tokens: i64,
    ) -> Result<()> {
        let result = sqlx::query(
            "UPDATE stacks 
            SET already_computed_units = already_computed_units - (? - ?) 
            WHERE stack_small_id = ?",
        )
        .bind(estimated_total_tokens)
        .bind(total_tokens)
        .bind(stack_small_id)
        .execute(&self.db)
        .await?;

        if result.rows_affected() == 0 {
            return Err(StateManagerError::StackNotFound);
        }

        Ok(())
    }

    /// Retrieves the stack settlement ticket for a given stack.
    ///
    /// This method fetches the settlement ticket details from the `stack_settlement_tickets` table
    /// based on the provided `stack_small_id`.
    ///
    /// # Arguments
    ///
    /// * `stack_small_id` - The unique small identifier of the stack whose settlement ticket is to be retrieved.
    ///
    /// # Returns
    ///
    /// - `Result<StackSettlementTicket>`: A result containing either:
    ///   - `Ok(StackSettlementTicket)`: A `StackSettlementTicket` object representing the stack settlement ticket.
    ///   - `Err(StateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database row into a `StackSettlementTicket` object.
    /// - No settlement ticket is found for the given `stack_small_id`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::{StateManager, StackSettlementTicket};
    ///
    /// async fn get_settlement_ticket(state_manager: &StateManager, stack_small_id: i64) -> Result<StackSettlementTicket, StateManagerError> {
    ///     state_manager.get_stack_settlement_ticket(stack_small_id).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack_small_id)
    )]
    pub async fn get_stack_settlement_ticket(
        &self,
        stack_small_id: i64,
    ) -> Result<StackSettlementTicket> {
        let stack_settlement_ticket =
            sqlx::query("SELECT * FROM stack_settlement_tickets WHERE stack_small_id = ?")
                .bind(stack_small_id)
                .fetch_one(&self.db)
                .await?;
        Ok(StackSettlementTicket::from_row(&stack_settlement_ticket)?)
    }

    /// Inserts a new stack settlement ticket into the database.
    ///
    /// This method inserts a new entry into the `stack_settlement_tickets` table with the provided stack settlement ticket details.
    ///
    /// # Arguments
    ///
    /// * `stack_settlement_ticket` - The `StackSettlementTicket` object to be inserted into the database.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(StateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the provided `StackSettlementTicket` object into a database row.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::{StateManager, StackSettlementTicket};
    ///
    /// async fn insert_settlement_ticket(state_manager: &StateManager, stack_settlement_ticket: StackSettlementTicket) -> Result<(), StateManagerError> {
    ///     state_manager.insert_new_stack_settlement_ticket(stack_settlement_ticket).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack_settlement_ticket.stack_small_id,
            selected_node_id = %stack_settlement_ticket.selected_node_id,
            num_claimed_compute_units = %stack_settlement_ticket.num_claimed_compute_units,
            requested_attestation_nodes = %stack_settlement_ticket.requested_attestation_nodes)
    )]
    pub async fn insert_new_stack_settlement_ticket(
        &self,
        stack_settlement_ticket: StackSettlementTicket,
    ) -> Result<()> {
        let mut tx = self.db.begin().await?;
        sqlx::query(
            "INSERT INTO stack_settlement_tickets 
                (
                    stack_small_id, 
                    selected_node_id, 
                    num_claimed_compute_units, 
                    requested_attestation_nodes, 
                    committed_stack_proofs, 
                    stack_merkle_leaves, 
                    dispute_settled_at_epoch, 
                    already_attested_nodes, 
                    is_in_dispute, 
                    user_refund_amount, 
                    is_claimed) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(stack_settlement_ticket.stack_small_id)
        .bind(stack_settlement_ticket.selected_node_id)
        .bind(stack_settlement_ticket.num_claimed_compute_units)
        .bind(stack_settlement_ticket.requested_attestation_nodes)
        .bind(stack_settlement_ticket.committed_stack_proofs)
        .bind(stack_settlement_ticket.stack_merkle_leaves)
        .bind(stack_settlement_ticket.dispute_settled_at_epoch)
        .bind(stack_settlement_ticket.already_attested_nodes)
        .bind(stack_settlement_ticket.is_in_dispute)
        .bind(stack_settlement_ticket.user_refund_amount)
        .bind(stack_settlement_ticket.is_claimed)
        .execute(&mut *tx)
        .await?;

        // Also update the stack to set in_settle_period to true
        sqlx::query("UPDATE stacks SET in_settle_period = true WHERE stack_small_id = ?")
            .bind(stack_settlement_ticket.stack_small_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Updates a stack settlement ticket with attestation commitments.
    ///
    /// This method updates the `stack_settlement_tickets` table with new attestation information
    /// for a specific stack. It updates the committed stack proof, stack Merkle leaf, and adds
    /// a new attestation node to the list of already attested nodes.
    ///
    /// # Arguments
    ///
    /// * `stack_small_id` - The unique small identifier of the stack to update.
    /// * `committed_stack_proof` - The new committed stack proof as a byte vector.
    /// * `stack_merkle_leaf` - The new stack Merkle leaf as a byte vector.
    /// * `attestation_node_id` - The ID of the node providing the attestation.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(StateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - The specified stack settlement ticket doesn't exist.
    /// - The attestation node is not found in the list of requested attestation nodes.
    /// - The provided Merkle leaf has an invalid length.
    /// - There's an issue updating the JSON array of already attested nodes.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::StateManager;
    ///
    /// async fn update_settlement_ticket(state_manager: &StateManager) -> Result<(), StateManagerError> {
    ///     let stack_small_id = 1;
    ///     let committed_stack_proof = vec![1, 2, 3, 4];
    ///     let stack_merkle_leaf = vec![5, 6, 7, 8];
    ///     let attestation_node_id = 42;
    ///
    ///     state_manager.update_stack_settlement_ticket_with_attestation_commitments(
    ///         stack_small_id,
    ///         committed_stack_proof,
    ///         stack_merkle_leaf,
    ///         attestation_node_id
    ///     ).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack_small_id,
            attestation_node_id = %attestation_node_id)
    )]
    pub async fn update_stack_settlement_ticket_with_attestation_commitments(
        &self,
        stack_small_id: i64,
        committed_stack_proof: Vec<u8>,
        stack_merkle_leaf: Vec<u8>,
        attestation_node_id: i64,
    ) -> Result<()> {
        let mut tx = self.db.begin().await?;

        let row = sqlx::query(
            "SELECT committed_stack_proofs, stack_merkle_leaves, requested_attestation_nodes 
             FROM stack_settlement_tickets 
             WHERE stack_small_id = ?",
        )
        .bind(stack_small_id)
        .fetch_one(&mut *tx)
        .await?;

        let mut committed_stack_proofs: Vec<u8> = row.get("committed_stack_proofs");
        let mut current_merkle_leaves: Vec<u8> = row.get("stack_merkle_leaves");
        let requested_nodes: String = row.get("requested_attestation_nodes");
        let requested_nodes: Vec<i64> = serde_json::from_str(&requested_nodes)?;

        // Find the index of the attestation_node_id
        let index = requested_nodes
            .iter()
            .position(|&id| id == attestation_node_id)
            .ok_or_else(|| StateManagerError::AttestationNodeNotFound(attestation_node_id))?;

        // Update the corresponding 32-byte range in the stack_merkle_leaves
        let start = (index + 1) * 32;
        let end = start + 32;
        if end > current_merkle_leaves.len() {
            return Err(StateManagerError::InvalidMerkleLeafLength);
        }
        if end > committed_stack_proofs.len() {
            return Err(StateManagerError::InvalidCommittedStackProofLength);
        }

        current_merkle_leaves[start..end].copy_from_slice(&stack_merkle_leaf[..32]);
        committed_stack_proofs[start..end].copy_from_slice(&committed_stack_proof[..32]);
        sqlx::query(
            "UPDATE stack_settlement_tickets 
             SET committed_stack_proofs = ?,
                 stack_merkle_leaves = ?, 
                 already_attested_nodes = json_insert(already_attested_nodes, '$[#]', ?)
             WHERE stack_small_id = ?",
        )
        .bind(committed_stack_proofs)
        .bind(current_merkle_leaves)
        .bind(attestation_node_id)
        .bind(stack_small_id)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        Ok(())
    }

    /// Settles a stack settlement ticket by updating the dispute settled at epoch.
    ///
    /// This method updates the `stack_settlement_tickets` table, setting the `dispute_settled_at_epoch` field.
    ///
    /// # Arguments
    ///
    /// * `stack_small_id` - The unique small identifier of the stack settlement ticket to update.
    /// * `dispute_settled_at_epoch` - The epoch at which the dispute was settled.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(StateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - The specified stack settlement ticket doesn't exist.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::StateManager;
    ///
    /// async fn settle_ticket(state_manager: &StateManager) -> Result<(), StateManagerError> {
    ///     let stack_small_id = 1;
    ///     let dispute_settled_at_epoch = 10;
    ///
    ///     state_manager.settle_stack_settlement_ticket(stack_small_id, dispute_settled_at_epoch).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack_small_id,
            dispute_settled_at_epoch = %dispute_settled_at_epoch)
    )]
    pub async fn settle_stack_settlement_ticket(
        &self,
        stack_small_id: i64,
        dispute_settled_at_epoch: i64,
    ) -> Result<()> {
        sqlx::query("UPDATE stack_settlement_tickets SET dispute_settled_at_epoch = ? WHERE stack_small_id = ?")
            .bind(dispute_settled_at_epoch)
            .bind(stack_small_id)
            .execute(&self.db)
            .await?;
        Ok(())
    }

    /// Updates a stack settlement ticket to mark it as claimed and set the user refund amount.
    ///
    /// This method updates the `stack_settlement_tickets` table, setting the `is_claimed` flag to true
    /// and updating the `user_refund_amount` for a specific stack settlement ticket.
    ///
    /// # Arguments
    ///
    /// * `stack_small_id` - The unique small identifier of the stack settlement ticket to update.
    /// * `user_refund_amount` - The amount to be refunded to the user.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(StateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - The specified stack settlement ticket doesn't exist.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::StateManager;
    ///
    /// async fn claim_settlement_ticket(state_manager: &StateManager) -> Result<(), StateManagerError> {
    ///     let stack_small_id = 1;
    ///     let user_refund_amount = 1000;
    ///
    ///     state_manager.update_stack_settlement_ticket_with_claim(stack_small_id, user_refund_amount).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack_small_id,
            user_refund_amount = %user_refund_amount)
    )]
    pub async fn update_stack_settlement_ticket_with_claim(
        &self,
        stack_small_id: i64,
        user_refund_amount: i64,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE stack_settlement_tickets 
                SET user_refund_amount = ?,
                    is_claimed = true
                WHERE stack_small_id = ?",
        )
        .bind(user_refund_amount)
        .bind(stack_small_id)
        .execute(&self.db)
        .await?;

        Ok(())
    }

    /// Retrieves all stack attestation disputes for a given stack and attestation node.
    ///
    /// This method fetches all disputes from the `stack_attestation_disputes` table
    /// that match the provided `stack_small_id` and `attestation_node_id`.
    ///
    /// # Arguments
    ///
    /// * `stack_small_id` - The unique small identifier of the stack.
    /// * `attestation_node_id` - The ID of the attestation node.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<StackAttestationDispute>>`: A result containing either:
    ///   - `Ok(Vec<StackAttestationDispute>)`: A vector of `StackAttestationDispute` objects representing all disputes found.
    ///   - `Err(StateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into `StackAttestationDispute` objects.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::{StateManager, StackAttestationDispute};
    ///
    /// async fn get_disputes(state_manager: &StateManager) -> Result<Vec<StackAttestationDispute>, StateManagerError> {
    ///     let stack_small_id = 1;
    ///     let attestation_node_id = 42;
    ///     state_manager.get_stack_attestation_disputes(stack_small_id, attestation_node_id).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack_small_id,
            attestation_node_id = %attestation_node_id)
    )]
    pub async fn get_stack_attestation_disputes(
        &self,
        stack_small_id: i64,
        attestation_node_id: i64,
    ) -> Result<Vec<StackAttestationDispute>> {
        let disputes = sqlx::query(
            "SELECT * FROM stack_attestation_disputes 
                WHERE stack_small_id = ? AND attestation_node_id = ?",
        )
        .bind(stack_small_id)
        .bind(attestation_node_id)
        .fetch_all(&self.db)
        .await?;

        disputes
            .into_iter()
            .map(|row| StackAttestationDispute::from_row(&row).map_err(StateManagerError::from))
            .collect()
    }

    /// Inserts a new stack attestation dispute into the database.
    ///
    /// This method adds a new entry to the `stack_attestation_disputes` table with the provided dispute information.
    ///
    /// # Arguments
    ///
    /// * `stack_attestation_dispute` - A `StackAttestationDispute` struct containing all the information about the dispute to be inserted.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(StateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's a constraint violation (e.g., duplicate primary key).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use atoma_node::atoma_state::{StateManager, StackAttestationDispute};
    ///
    /// async fn add_dispute(state_manager: &StateManager, dispute: StackAttestationDispute) -> Result<(), StateManagerError> {
    ///     state_manager.insert_stack_attestation_dispute(dispute).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack_attestation_dispute.stack_small_id,
            attestation_node_id = %stack_attestation_dispute.attestation_node_id,
            original_node_id = %stack_attestation_dispute.original_node_id)
    )]
    pub async fn insert_stack_attestation_dispute(
        &self,
        stack_attestation_dispute: StackAttestationDispute,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO stack_attestation_disputes 
                (stack_small_id, attestation_commitment, attestation_node_id, original_node_id, original_commitment) 
                VALUES (?, ?, ?, ?, ?)",
        )
            .bind(stack_attestation_dispute.stack_small_id)
            .bind(stack_attestation_dispute.attestation_commitment)
            .bind(stack_attestation_dispute.attestation_node_id)
            .bind(stack_attestation_dispute.original_node_id)
            .bind(stack_attestation_dispute.original_commitment)
            .execute(&self.db)
            .await?;

        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum StateManagerError {
    #[error("Failed to connect to the database: {0}")]
    DatabaseConnectionError(#[from] sqlx::Error),
    #[error("Stack not found")]
    StackNotFound,
    #[error("Attestation node not found: {0}")]
    AttestationNodeNotFound(i64),
    #[error("Invalid Merkle leaf length")]
    InvalidMerkleLeafLength,
    #[error("Invalid committed stack proof length")]
    InvalidCommittedStackProofLength,
    #[error("Failed to parse JSON: {0}")]
    JsonParseError(#[from] serde_json::Error),
}

pub(crate) mod queries {
    use sqlx::Pool;

    use super::*;

    /// Generates the SQL query to create the `tasks` table.
    ///
    /// This table stores information about tasks in the system.
    ///
    /// # Table Structure
    /// - `task_small_id`: INTEGER PRIMARY KEY - The unique identifier for the task.
    /// - `task_id`: TEXT UNIQUE NOT NULL - A unique text identifier for the task.
    /// - `role`: INTEGER NOT NULL - The role associated with the task.
    /// - `model_name`: TEXT - The name of the model used for the task (nullable).
    /// - `is_deprecated`: BOOLEAN NOT NULL - Indicates whether the task is deprecated.
    /// - `valid_until_epoch`: INTEGER - The epoch until which the task is valid (nullable).
    /// - `deprecated_at_epoch`: INTEGER - The epoch when the task was deprecated (nullable).
    /// - `optimizations`: TEXT NOT NULL - A string representing task optimizations.
    /// - `security_level`: INTEGER NOT NULL - The security level of the task.
    /// - `task_metrics_compute_unit`: INTEGER NOT NULL - The compute unit metric for the task.
    /// - `task_metrics_time_unit`: INTEGER - The time unit metric for the task (nullable).
    /// - `task_metrics_value`: INTEGER - The value metric for the task (nullable).
    /// - `minimum_reputation_score`: INTEGER - The minimum reputation score required (nullable).
    ///
    /// # Primary Key
    /// The table uses `task_small_id` as the primary key.
    ///
    /// # Unique Constraint
    /// The `task_id` field has a unique constraint to ensure no duplicate task IDs.
    ///
    /// # Returns
    /// A `String` containing the SQL query to create the `tasks` table.
    pub(crate) fn create_tasks_table_query() -> String {
        "CREATE TABLE IF NOT EXISTS tasks (
                task_small_id INTEGER PRIMARY KEY,
                task_id TEXT UNIQUE NOT NULL,
                role INTEGER NOT NULL,
                model_name TEXT,
                is_deprecated BOOLEAN NOT NULL,
                valid_until_epoch INTEGER,
                deprecated_at_epoch INTEGER,
                optimizations TEXT NOT NULL,
                security_level INTEGER NOT NULL,
                task_metrics_compute_unit INTEGER NOT NULL,
                task_metrics_time_unit INTEGER,
                task_metrics_value INTEGER,
                minimum_reputation_score INTEGER
            )"
        .to_string()
    }

    /// Generates the SQL query to create the `node_subscriptions` table.
    ///
    /// This table stores information about node subscriptions to tasks.
    ///
    /// # Table Structure
    /// - `task_small_id`: INTEGER NOT NULL - The ID of the task being subscribed to.
    /// - `node_small_id`: INTEGER NOT NULL - The ID of the node subscribing to the task.
    /// - `price_per_compute_unit`: INTEGER NOT NULL - The price per compute unit for this subscription.
    ///
    /// # Primary Key
    /// The table uses a composite primary key of (task_small_id, node_small_id).
    ///
    /// # Foreign Key
    /// - `task_small_id` references the `tasks` table.
    ///
    /// # Returns
    /// A `String` containing the SQL query to create the `node_subscriptions` table.
    pub(crate) fn subscribed_tasks_query() -> String {
        "CREATE TABLE IF NOT EXISTS node_subscriptions (
            task_small_id INTEGER NOT NULL,
            node_small_id INTEGER NOT NULL,
            price_per_compute_unit INTEGER NOT NULL,
            max_num_compute_units INTEGER NOT NULL,
            valid BOOLEAN NOT NULL,
            PRIMARY KEY (task_small_id, node_small_id),
            FOREIGN KEY (task_small_id) REFERENCES tasks (task_small_id)
        );
        CREATE INDEX IF NOT EXISTS retrieval_index ON node_subscriptions (task_small_id, node_small_id);"
        .to_string()
    }

    /// Generates the SQL query to create the `stacks` table.
    ///
    /// This table stores information about stacks in the system.
    ///
    /// # Table Structure
    /// - `stack_small_id`: INTEGER PRIMARY KEY - The unique identifier for the stack.
    /// - `stack_id`: TEXT UNIQUE NOT NULL - A unique text identifier for the stack.
    /// - `selected_node_id`: INTEGER NOT NULL - The ID of the node selected for this stack.
    /// - `num_compute_units`: INTEGER NOT NULL - The number of compute units allocated to this stack.
    /// - `price`: INTEGER NOT NULL - The price associated with this stack.
    ///
    /// # Primary Key
    /// The table uses `stack_small_id` as the primary key.
    ///
    /// # Foreign Keys
    /// - `selected_node_id` references the `node_subscriptions` table.
    /// - `task_small_id` references the `tasks` table.
    /// # Unique Constraint
    /// The `stack_id` field has a unique constraint to ensure no duplicate stack IDs.
    ///
    /// # Returns
    /// A `String` containing the SQL query to create the `stacks` table.
    pub(crate) fn stacks() -> String {
        "CREATE TABLE IF NOT EXISTS stacks (
                stack_small_id INTEGER PRIMARY KEY,
                owner_address TEXT NOT NULL,
                stack_id TEXT UNIQUE NOT NULL,
                task_small_id INTEGER NOT NULL,
                selected_node_id INTEGER NOT NULL,
                num_compute_units INTEGER NOT NULL,
                price INTEGER NOT NULL,
                already_computed_units INTEGER NOT NULL,
                in_settle_period BOOLEAN NOT NULL,
                FOREIGN KEY (selected_node_id, task_small_id) REFERENCES node_subscriptions (node_small_id, task_small_id)
            );
            CREATE INDEX IF NOT EXISTS owner_address_index ON stacks (owner_address);
            CREATE INDEX IF NOT EXISTS stack_small_id_index ON stacks (stack_small_id);"
        .to_string()
    }

    /// Generates the SQL query to create the `stack_settlement_tickets` table.
    ///
    /// This table stores information about settlement tickets for stacks.
    ///
    /// # Table Structure
    /// - `stack_small_id`: INTEGER PRIMARY KEY - The unique identifier for the stack.
    /// - `selected_node_id`: INTEGER NOT NULL - The ID of the node selected for settlement.
    /// - `num_claimed_compute_units`: INTEGER NOT NULL - The number of compute units claimed.
    /// - `requested_attestation_nodes`: TEXT NOT NULL - A list of nodes requested for attestation.
    /// - `committed_stack_proofs`: BLOB NOT NULL - The committed proofs for the stack settlement.
    /// - `stack_merkle_leaves`: BLOB NOT NULL - The Merkle leaves for the stack.
    /// - `dispute_settled_at_epoch`: INTEGER - The epoch when the dispute was settled (nullable).
    /// - `already_attested_nodes`: TEXT NOT NULL - A list of nodes that have already attested.
    /// - `is_in_dispute`: BOOLEAN NOT NULL - Indicates whether the settlement is in dispute.
    /// - `user_refund_amount`: INTEGER NOT NULL - The amount to be refunded to the user.
    /// - `is_claimed`: BOOLEAN NOT NULL - Indicates whether the settlement has been claimed.
    ///
    /// # Primary Key
    /// The table uses `stack_small_id` as the primary key.
    ///
    /// # Returns
    /// A `String` containing the SQL query to create the `stack_settlement_tickets` table.
    ///
    /// # Foreign Key
    /// - `stack_small_id` references the `stacks` table.
    ///
    /// # Returns
    /// A `String` containing the SQL query to create the `stack_settlement_tickets` table.
    pub(crate) fn stack_settlement_tickets() -> String {
        "CREATE TABLE IF NOT EXISTS stack_settlement_tickets (
            stack_small_id INTEGER PRIMARY KEY,
            selected_node_id INTEGER NOT NULL,
            num_claimed_compute_units INTEGER NOT NULL,
            requested_attestation_nodes TEXT NOT NULL,
            committed_stack_proofs BLOB NOT NULL,
            stack_merkle_leaves BLOB NOT NULL,
            dispute_settled_at_epoch INTEGER,
            already_attested_nodes TEXT NOT NULL,
            is_in_dispute BOOLEAN NOT NULL,
            user_refund_amount INTEGER NOT NULL,
            is_claimed BOOLEAN NOT NULL
        );
        CREATE INDEX IF NOT EXISTS stack_small_id_index ON stack_settlement_tickets (stack_small_id);"
        .to_string()
    }

    /// Generates the SQL query to create the `stack_attestation_disputes` table.
    ///
    /// This table stores information about disputes related to stack attestations.
    ///
    /// # Table Structure
    /// - `stack_small_id`: INTEGER NOT NULL - The ID of the stack involved in the dispute.
    /// - `attestation_commitment`: BLOB NOT NULL - The commitment provided by the attesting node.
    /// - `attestation_node_id`: INTEGER NOT NULL - The ID of the node providing the attestation.
    /// - `original_node_id`: INTEGER NOT NULL - The ID of the original node involved in the dispute.
    /// - `original_commitment`: BLOB NOT NULL - The original commitment that is being disputed.
    ///
    /// # Primary Key
    /// The table uses a composite primary key of (stack_small_id, attestation_node_id).
    ///
    /// # Foreign Key
    /// - `stack_small_id` references the `stacks` table.
    ///
    /// # Returns
    /// A `String` containing the SQL query to create the `stack_attestation_disputes` table.
    pub(crate) fn stack_attestation_disputes() -> String {
        "CREATE TABLE IF NOT EXISTS stack_attestation_disputes (
                stack_small_id INTEGER NOT NULL,
                attestation_commitment BLOB NOT NULL,
                attestation_node_id INTEGER NOT NULL,
                original_node_id INTEGER NOT NULL,
                original_commitment BLOB NOT NULL,
                PRIMARY KEY (stack_small_id, attestation_node_id),
                FOREIGN KEY (stack_small_id) REFERENCES stacks (stack_small_id)
            )"
        .to_string()
    }

    /// Creates all the necessary tables in the database.
    ///
    /// This function executes SQL queries to create the following tables:
    /// - tasks
    /// - node_subscriptions
    /// - stacks
    /// - stack_settlement_tickets
    /// - stack_attestation_disputes
    ///
    /// # Arguments
    ///
    /// * `db` - A reference to the SQLite database pool.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if all tables are created successfully, or an error if any operation fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if any of the SQL queries fail to execute.
    /// Possible reasons for failure include:
    /// - Database connection issues
    /// - Insufficient permissions
    /// - Syntax errors in the SQL queries
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use sqlx::SqlitePool;
    /// use atoma_node::atoma_state::queries;
    ///
    /// async fn setup_database(pool: &SqlitePool) -> Result<(), Box<dyn std::error::Error>> {
    ///     queries::create_all_tables(pool).await?;
    ///     Ok(())
    /// }
    /// ```
    pub(crate) async fn create_all_tables(db: &Pool<Sqlite>) -> Result<()> {
        sqlx::query("PRAGMA foreign_keys = ON;").execute(db).await?;

        sqlx::query(&create_tasks_table_query()).execute(db).await?;
        sqlx::query(&subscribed_tasks_query()).execute(db).await?;
        sqlx::query(&stacks()).execute(db).await?;
        sqlx::query(&stack_settlement_tickets()).execute(db).await?;
        sqlx::query(&stack_attestation_disputes())
            .execute(db)
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup_test_db() -> (StateManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        std::fs::create_dir_all(temp_dir.path()).unwrap();
        std::fs::File::create(&db_path).unwrap();
        let database_url = format!("sqlite:{}", db_path.to_str().unwrap());
        let state_manager = StateManager::new_from_url(database_url).await.unwrap();
        (state_manager, temp_dir)
    }

    #[tokio::test]
    async fn test_insert_and_get_task() {
        let (state_manager, temp_dir) = setup_test_db().await;

        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            optimizations: "opt1".to_string(),
            security_level: 1,
            task_metrics_compute_unit: 10,
            task_metrics_time_unit: Some(5),
            task_metrics_value: Some(100),
            minimum_reputation_score: Some(50),
        };

        state_manager.insert_new_task(task.clone()).await.unwrap();
        let retrieved_task = state_manager.get_task_by_small_id(1).await.unwrap();
        assert_eq!(task, retrieved_task);

        std::fs::remove_dir_all(temp_dir.path()).unwrap();
    }

    #[tokio::test]
    async fn test_deprecate_task() {
        let (state_manager, temp_dir) = setup_test_db().await;

        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            optimizations: "opt1".to_string(),
            security_level: 1,
            task_metrics_compute_unit: 10,
            task_metrics_time_unit: Some(5),
            task_metrics_value: Some(100),
            minimum_reputation_score: Some(50),
        };

        state_manager.insert_new_task(task).await.unwrap();
        state_manager.deprecate_task(1, 100).await.unwrap();

        let deprecated_task = state_manager.get_task_by_small_id(1).await.unwrap();
        assert!(deprecated_task.is_deprecated);

        std::fs::remove_dir_all(temp_dir.path()).unwrap();
    }

    #[tokio::test]
    async fn test_get_subscribed_tasks() {
        let (state_manager, temp_dir) = setup_test_db().await;

        // Insert two tasks
        let task1 = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            optimizations: "opt1".to_string(),
            security_level: 1,
            task_metrics_compute_unit: 10,
            task_metrics_time_unit: Some(5),
            task_metrics_value: Some(100),
            minimum_reputation_score: Some(50),
        };
        let task2 = Task {
            task_small_id: 2,
            task_id: "task2".to_string(),
            role: 1,
            model_name: Some("model2".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            optimizations: "opt2".to_string(),
            security_level: 1,
            task_metrics_compute_unit: 10,
            task_metrics_time_unit: Some(5),
            task_metrics_value: Some(100),
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task1.clone()).await.unwrap();
        state_manager.insert_new_task(task2).await.unwrap();

        // Subscribe a node to task1
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();

        let subscribed_tasks = state_manager.get_subscribed_tasks(1).await.unwrap();
        assert_eq!(subscribed_tasks.len(), 1);
        assert_eq!(subscribed_tasks[0], task1);

        std::fs::remove_dir_all(temp_dir.path()).unwrap();
    }

    #[tokio::test]
    async fn test_update_node_subscription() {
        let (state_manager, temp_dir) = setup_test_db().await;

        // Insert a task
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            optimizations: "opt1".to_string(),
            security_level: 1,
            task_metrics_compute_unit: 10,
            task_metrics_time_unit: Some(5),
            task_metrics_value: Some(100),
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();

        // Subscribe a node to the task
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();

        let subscription = state_manager
            .get_node_subscription_by_task_small_id(1)
            .await
            .unwrap();
        assert_eq!(subscription.price_per_compute_unit, 100);
        assert_eq!(subscription.max_num_compute_units, 1000);

        // Update the subscription
        state_manager
            .update_node_subscription(1, 1, 200, 2000)
            .await
            .unwrap();

        // Verify the update
        // You'll need to implement a method to get subscription details)
        let subscription = state_manager
            .get_node_subscription_by_task_small_id(1)
            .await
            .unwrap();
        assert_eq!(subscription.price_per_compute_unit, 200);
        assert_eq!(subscription.max_num_compute_units, 2000);

        std::fs::remove_dir_all(temp_dir.path()).unwrap();
    }

    #[tokio::test]
    async fn test_insert_and_get_stack() {
        let (state_manager, temp_dir) = setup_test_db().await;

        // Insert a task and subscribe a node
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            optimizations: "opt1".to_string(),
            security_level: 1,
            task_metrics_compute_unit: 10,
            task_metrics_time_unit: Some(5),
            task_metrics_value: Some(100),
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();

        // Insert a stack
        let stack = Stack {
            owner_address: "0x123".to_string(),
            stack_small_id: 1,
            stack_id: "stack1".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 10,
            price: 1000,
            already_computed_units: 0,
            in_settle_period: false,
        };
        state_manager.insert_new_stack(stack.clone()).await.unwrap();

        // Get the stack and verify
        let retrieved_stack = state_manager.get_stack(1).await.unwrap();
        assert_eq!(stack, retrieved_stack);

        std::fs::remove_dir_all(temp_dir.path()).unwrap();
    }

    #[tokio::test]
    async fn test_update_computed_units() {
        let (state_manager, temp_dir) = setup_test_db().await;

        // Insert a task, subscribe a node, and create a stack
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            optimizations: "opt1".to_string(),
            security_level: 1,
            task_metrics_compute_unit: 10,
            task_metrics_time_unit: Some(5),
            task_metrics_value: Some(100),
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();
        let stack = Stack {
            owner_address: "0x123".to_string(),
            stack_small_id: 1,
            stack_id: "stack1".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 10,
            price: 1000,
            already_computed_units: 0,
            in_settle_period: false,
        };
        state_manager.insert_new_stack(stack).await.unwrap();

        // Update computed units

        state_manager
            .update_computed_units_for_stack(1, 15)
            .await
            .unwrap();

        // Verify the update
        let updated_stack = state_manager.get_stack(1).await.unwrap();
        assert_eq!(updated_stack.already_computed_units, 15);

        std::fs::remove_dir_all(temp_dir.path()).unwrap();
    }

    #[tokio::test]
    async fn test_insert_and_get_stack_settlement_ticket() {
        let (state_manager, temp_dir) = setup_test_db().await;

        // Insert a task, subscribe a node, and create a stack
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            optimizations: "opt1".to_string(),
            security_level: 1,
            task_metrics_compute_unit: 10,
            task_metrics_time_unit: Some(5),
            task_metrics_value: Some(100),
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();
        let stack = Stack {
            owner_address: "0x123".to_string(),
            stack_small_id: 1,
            stack_id: "stack1".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 10,
            price: 1000,
            already_computed_units: 0,
            in_settle_period: false,
        };
        state_manager.insert_new_stack(stack).await.unwrap();

        // Insert a stack settlement ticket
        let ticket = StackSettlementTicket {
            stack_small_id: 1,
            selected_node_id: 1,
            num_claimed_compute_units: 10,
            requested_attestation_nodes: "node1,node2".to_string(),
            committed_stack_proofs: vec![1, 2, 3],
            stack_merkle_leaves: vec![4, 5, 6],
            dispute_settled_at_epoch: None,
            already_attested_nodes: "[]".to_string(),
            is_in_dispute: false,
            user_refund_amount: 0,
            is_claimed: false,
        };
        state_manager
            .insert_new_stack_settlement_ticket(ticket.clone())
            .await
            .unwrap();

        // Verify the insertion (you'll need to implement a method to get a settlement ticket)
        let retrieved_ticket = state_manager.get_stack_settlement_ticket(1).await.unwrap();
        assert_eq!(ticket, retrieved_ticket);

        std::fs::remove_dir_all(temp_dir.path()).unwrap();
    }

    #[tokio::test]
    async fn test_update_stack_settlement_ticket() {
        let (state_manager, temp_dir) = setup_test_db().await;

        // Setup: Insert task, subscription, stack, and initial settlement ticket
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            optimizations: "opt1".to_string(),
            security_level: 1,
            task_metrics_compute_unit: 10,
            task_metrics_time_unit: Some(5),
            task_metrics_value: Some(100),
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();
        let stack = Stack {
            owner_address: "0x123".to_string(),
            stack_small_id: 1,
            stack_id: "stack1".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 10,
            price: 1000,
            already_computed_units: 0,
            in_settle_period: false,
        };
        state_manager.insert_new_stack(stack).await.unwrap();
        let initial_ticket = StackSettlementTicket {
            stack_small_id: 1,
            selected_node_id: 1,
            num_claimed_compute_units: 5,
            requested_attestation_nodes: "[1,2]".to_string(),
            committed_stack_proofs: vec![0; 96],
            stack_merkle_leaves: vec![0; 96],
            dispute_settled_at_epoch: None,
            already_attested_nodes: "[]".to_string(),
            is_in_dispute: false,
            user_refund_amount: 0,
            is_claimed: false,
        };
        state_manager
            .insert_new_stack_settlement_ticket(initial_ticket)
            .await
            .unwrap();

        // Update the settlement ticket with attestation commitments
        let committed_stack_proof = vec![2; 32];
        let stack_merkle_leaf = vec![2; 32];
        let attestation_node_id = 2;
        state_manager
            .update_stack_settlement_ticket_with_attestation_commitments(
                1,
                committed_stack_proof.clone(),
                stack_merkle_leaf.clone(),
                attestation_node_id,
            )
            .await
            .unwrap();

        // Verify the update
        let updated_ticket = state_manager.get_stack_settlement_ticket(1).await.unwrap();
        assert_eq!(
            updated_ticket.committed_stack_proofs[64..96],
            committed_stack_proof[0..32]
        );
        assert_eq!(
            updated_ticket.stack_merkle_leaves[64..96],
            stack_merkle_leaf[0..32]
        );
        assert_eq!(updated_ticket.committed_stack_proofs[0..64], vec![0; 64]);
        assert_eq!(updated_ticket.stack_merkle_leaves[0..64], vec![0; 64]);
        assert_eq!(updated_ticket.already_attested_nodes, "[2]");

        // Update the settlement ticket with a claim
        let user_refund_amount = 500;
        state_manager
            .update_stack_settlement_ticket_with_claim(1, user_refund_amount)
            .await
            .unwrap();

        // Verify the claim update
        let claimed_ticket = state_manager.get_stack_settlement_ticket(1).await.unwrap();
        assert_eq!(claimed_ticket.user_refund_amount, user_refund_amount);
        assert!(claimed_ticket.is_claimed);

        std::fs::remove_dir_all(temp_dir.path()).unwrap();
    }

    #[tokio::test]
    async fn test_stack_attestation_disputes() {
        let (state_manager, temp_dir) = setup_test_db().await;

        // Setup: Insert task, subscription, stack, and settlement ticket
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            optimizations: "opt1".to_string(),
            security_level: 1,
            task_metrics_compute_unit: 10,
            task_metrics_time_unit: Some(5),
            task_metrics_value: Some(100),
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();
        let stack = Stack {
            owner_address: "0x123".to_string(),
            stack_small_id: 1,
            stack_id: "stack1".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 10,
            price: 1000,
            already_computed_units: 0,
            in_settle_period: false,
        };
        state_manager.insert_new_stack(stack).await.unwrap();
        let ticket = StackSettlementTicket {
            stack_small_id: 1,
            selected_node_id: 1,
            num_claimed_compute_units: 10,
            requested_attestation_nodes: "2,3".to_string(),
            committed_stack_proofs: vec![1, 2, 3],
            stack_merkle_leaves: vec![4, 5, 6],
            dispute_settled_at_epoch: None,
            already_attested_nodes: "2".to_string(),
            is_in_dispute: false,
            user_refund_amount: 0,
            is_claimed: false,
        };
        state_manager
            .insert_new_stack_settlement_ticket(ticket)
            .await
            .unwrap();

        // Insert an attestation dispute
        let dispute = StackAttestationDispute {
            stack_small_id: 1,
            attestation_commitment: vec![7, 8, 9],
            attestation_node_id: 3,
            original_node_id: 1,
            original_commitment: vec![1, 2, 3],
        };
        state_manager
            .insert_stack_attestation_dispute(dispute.clone())
            .await
            .unwrap();

        // Verify the dispute was inserted
        let retrieved_disputes = state_manager
            .get_stack_attestation_disputes(1, 3)
            .await
            .unwrap();
        assert_eq!(retrieved_disputes.len(), 1);
        assert_eq!(retrieved_disputes[0], dispute);

        // Try to insert a duplicate dispute (should fail)
        let result = state_manager
            .insert_stack_attestation_dispute(dispute.clone())
            .await;
        assert!(result.is_err());

        // Insert another dispute for the same stack but different attestation node
        let another_dispute = StackAttestationDispute {
            stack_small_id: 1,
            attestation_commitment: vec![10, 11, 12],
            attestation_node_id: 2,
            original_node_id: 1,
            original_commitment: vec![1, 2, 3],
        };
        state_manager
            .insert_stack_attestation_dispute(another_dispute)
            .await
            .unwrap();

        // Verify both disputes are present
        let disputes = state_manager
            .get_stack_attestation_disputes(1, 2)
            .await
            .unwrap();
        assert_eq!(disputes.len(), 1);
        let disputes = state_manager
            .get_stack_attestation_disputes(1, 2)
            .await
            .unwrap();
        assert_eq!(disputes.len(), 1);

        std::fs::remove_dir_all(temp_dir.path()).unwrap();
    }

    #[tokio::test]
    async fn test_get_available_stack_with_compute_units() {
        let (state_manager, temp_dir) = setup_test_db().await;

        // Setup: Insert a task and subscribe a node
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            optimizations: "opt1".to_string(),
            security_level: 1,
            task_metrics_compute_unit: 10,
            task_metrics_time_unit: Some(5),
            task_metrics_value: Some(100),
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();

        // Test case 1: Stack with sufficient compute units
        let stack1 = Stack {
            owner_address: "0x123".to_string(),
            stack_small_id: 1,
            stack_id: "stack1".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 100,
            price: 1000,
            already_computed_units: 0,
            in_settle_period: false,
        };
        state_manager
            .insert_new_stack(stack1.clone())
            .await
            .unwrap();

        let result = state_manager
            .get_available_stack_with_compute_units(1, "0x123", 50)
            .await
            .unwrap();
        assert!(result.is_some());
        let updated_stack = result.unwrap();
        assert_eq!(updated_stack.already_computed_units, 50);

        // Test case 2: Stack with insufficient compute units
        let result = state_manager
            .get_available_stack_with_compute_units(1, "0x123", 60)
            .await
            .unwrap();
        assert!(result.is_none());

        // Test case 3: Stack in settle period
        let mut stack2 = stack1.clone();
        stack2.stack_small_id = 2;
        stack2.stack_id = "stack2".to_string();
        stack2.in_settle_period = true;
        state_manager.insert_new_stack(stack2).await.unwrap();

        let result = state_manager
            .get_available_stack_with_compute_units(2, "0x123", 50)
            .await
            .unwrap();
        assert!(result.is_none());

        // Test case 4: Stack with different owner
        let mut stack3 = stack1.clone();
        stack3.stack_small_id = 3;
        stack3.stack_id = "stack3".to_string();
        stack3.owner_address = "0x456".to_string();
        state_manager.insert_new_stack(stack3).await.unwrap();

        let result = state_manager
            .get_available_stack_with_compute_units(3, "0x123", 50)
            .await
            .unwrap();
        assert!(result.is_none());

        // Test case 5: Non-existent stack
        let result = state_manager
            .get_available_stack_with_compute_units(999, "0x123", 50)
            .await
            .unwrap();
        assert!(result.is_none());

        // Test case 6: Exact number of compute units available
        let mut stack4 = stack1.clone();
        stack4.stack_small_id = 4;
        stack4.stack_id = "stack4".to_string();
        stack4.num_compute_units = 100;
        stack4.already_computed_units = 50;
        state_manager.insert_new_stack(stack4).await.unwrap();

        let result = state_manager
            .get_available_stack_with_compute_units(4, "0x123", 50)
            .await
            .unwrap();
        assert!(result.is_some());
        let updated_stack = result.unwrap();
        assert_eq!(updated_stack.already_computed_units, 100);

        // Test case 7: Attempt to use more compute units than available
        let result = state_manager
            .get_available_stack_with_compute_units(4, "0x123", 1)
            .await
            .unwrap();
        assert!(result.is_none());

        std::fs::remove_dir_all(temp_dir.path()).unwrap();
    }
}
