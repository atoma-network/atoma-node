use crate::build_query_with_in;
use crate::handlers::{handle_atoma_event, handle_p2p_event, handle_state_manager_event};
use crate::types::{
    AtomaAtomaStateManagerEvent, Node, NodeSubscription, Stack, StackAttestationDispute,
    StackSettlementTicket, Task,
};

use atoma_p2p::types::AtomaP2pEvent;
use atoma_sui::events::AtomaEvent;
use flume::Receiver as FlumeReceiver;
use sqlx::PgPool;
use sqlx::{FromRow, Row};
use thiserror::Error;
use tokio::sync::oneshot;
use tokio::sync::watch::Receiver;

pub(crate) type Result<T> = std::result::Result<T, AtomaStateManagerError>;

/// AtomaStateManager is a wrapper around a Postgres connection pool, responsible for managing the state of the Atoma system.
///
/// It provides an interface to interact with the Postgres database, handling operations
/// related to tasks, node subscriptions, stacks, and various other system components.
pub struct AtomaStateManager {
    /// The Postgres connection pool used for database operations.
    pub state: AtomaState,
    /// Receiver channel from the SuiEventSubscriber
    pub event_subscriber_receiver: FlumeReceiver<AtomaEvent>,
    /// Atoma service receiver
    pub state_manager_receiver: FlumeReceiver<AtomaAtomaStateManagerEvent>,
    /// Atoma p2p service receiver
    pub p2p_service_receiver: FlumeReceiver<(AtomaP2pEvent, Option<oneshot::Sender<bool>>)>,
}

impl AtomaStateManager {
    /// Constructor
    #[must_use]
    pub const fn new(
        db: PgPool,
        event_subscriber_receiver: FlumeReceiver<AtomaEvent>,
        state_manager_receiver: FlumeReceiver<AtomaAtomaStateManagerEvent>,
        p2p_service_receiver: FlumeReceiver<(AtomaP2pEvent, Option<oneshot::Sender<bool>>)>,
    ) -> Self {
        Self {
            state: AtomaState::new(db),
            event_subscriber_receiver,
            state_manager_receiver,
            p2p_service_receiver,
        }
    }

    /// Creates a new `AtomaStateManager` instance from a database URL.
    ///
    /// This method establishes a connection to the Postgres database using the provided URL,
    /// creates all necessary tables in the database, and returns a new `AtomaStateManager` instance.
    ///
    /// # Arguments
    /// * `database_url` - The URL of the PostgreSQL database to connect to
    /// * `event_subscriber_receiver` - Channel receiver for Atoma events
    /// * `state_manager_receiver` - Channel receiver for state manager events
    ///
    /// # Returns
    /// A new state manager instance
    ///
    /// # Errors
    /// Returns `AtomaStateManagerError` if:
    /// - Failed to connect to database
    /// - Failed to create connection pool
    /// - Failed to run database migrations
    pub async fn new_from_url(
        database_url: &str,
        event_subscriber_receiver: FlumeReceiver<AtomaEvent>,
        state_manager_receiver: FlumeReceiver<AtomaAtomaStateManagerEvent>,
        p2p_service_receiver: FlumeReceiver<(AtomaP2pEvent, Option<oneshot::Sender<bool>>)>,
    ) -> Result<Self> {
        // Create connection options with create_if_missing enabled
        let db = PgPool::connect(database_url).await?;
        sqlx::migrate!("./src/migrations").run(&db).await?;
        Ok(Self {
            state: AtomaState::new(db),
            event_subscriber_receiver,
            state_manager_receiver,
            p2p_service_receiver,
        })
    }

    /// Runs the state manager, listening for events from the event subscriber and state manager receivers.
    ///
    /// This method continuously processes incoming events from the event subscriber and state manager receivers
    /// until a shutdown signal is received. It uses asynchronous select to handle multiple event sources concurrently.
    ///
    /// # Arguments
    ///
    /// * `shutdown_signal` - A `Receiver<bool>` that signals when the state manager should shut down.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - An error occurs while handling events from the event subscriber or state manager receivers.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn start_state_manager(state_manager: AtomaStateManager) -> Result<(), AtomaStateManagerError> {
    ///     let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    ///     state_manager.run(shutdown_rx).await
    /// }
    /// ```
    #[tracing::instrument(level = "trace", skip_all)]
    pub async fn run(self, mut shutdown_signal: Receiver<bool>) -> Result<()> {
        loop {
            tokio::select! {
                atoma_event = self.event_subscriber_receiver.recv_async() => {
                    match atoma_event {
                        Ok(atoma_event) => {
                            tracing::trace!(
                                target = "atoma-state-manager",
                                event = "event_subscriber_receiver",
                                "Event received from event subscriber receiver"
                            );
                            handle_atoma_event(atoma_event, &self).await?;
                        }
                        Err(e) => {
                            tracing::error!(
                                target = "atoma-state-manager",
                                event = "event_subscriber_receiver_error",
                                error = %e,
                                "All event subscriber senders have been dropped, terminating the state manager running process"
                            );
                            break;
                        }
                    }
                }
                state_manager_event = self.state_manager_receiver.recv_async() => {
                    match state_manager_event {
                        Ok(state_manager_event) => {
                            match handle_state_manager_event(&self, state_manager_event).await {
                                Ok(()) => continue,
                                Err(e) => {
                                    tracing::error!(
                                        target = "atoma-state-manager",
                                        event = "state_manager_event_error",
                                        error = %e,
                                        "Error handling state manager event"
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                target = "atoma-state-manager",
                                event = "state_manager_receiver_error",
                                error = %e,
                                "All state manager senders have been dropped, we will not be able to handle any more events from the Atoma node inference service"
                            );
                            // NOTE: We continue the loop, as the inference service might be shutting down,
                            // but we want to keep the state manager running
                            // for event synchronization with the Atoma Network protocol.
                            continue;
                        }
                    }
                }
                p2p_event = self.p2p_service_receiver.recv_async() => {
                    match p2p_event {
                        Ok((p2p_event, Some(sender))) => {
                            handle_p2p_event(&self, p2p_event, sender).await?;
                        }
                        Ok((_, None)) => {
                            // NOTE: Atoma nodes do not need to register public URLs of other peer nodes, and this event is unreachable
                            // from the Atoma state manager service
                            unreachable!("Atoma nodes do not need to register public URLs of other peer nodes");
                        }
                        Err(e) => {
                            tracing::error!(
                                target = "atoma-state-manager",
                                event = "p2p_service_receiver_error",
                                error = %e,
                                "All p2p service senders have been dropped, terminating the state manager running process"
                            );
                            // NOTE: We continue the loop, as the inference service might be shutting down,
                            // but we want to keep the state manager running
                            // for event synchronization with the Atoma Network protocol.
                            continue;
                        }
                    }
                }
                shutdown_signal_changed = shutdown_signal.changed() => {
                    match shutdown_signal_changed {
                        Ok(()) => {
                            if *shutdown_signal.borrow() {
                                tracing::trace!(
                                    target = "atoma-state-manager",
                                    event = "shutdown_signal",
                                    "Shutdown signal received, shutting down"
                                );
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                target = "atoma-state-manager",
                                event = "shutdown_signal_error",
                                error = %e,
                                "Shutdown signal channel closed"
                            );
                            // NOTE: We want to break here as well, since no one can signal shutdown anymore
                            break;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// AtomaState is a wrapper around a Postgres connection pool, responsible for managing the state of the Atoma system.
#[derive(Clone)]
pub struct AtomaState {
    /// The Postgres connection pool used for database operations.
    pub db: PgPool,
}

impl AtomaState {
    /// Constructor
    ///
    /// # Arguments
    /// * `db` - The Postgres connection pool
    ///
    /// # Returns
    /// A new `AtomaState` instance
    #[must_use]
    pub const fn new(db: PgPool) -> Self {
        Self { db }
    }

    /// Creates a new `AtomaState` instance from a database URL.
    ///
    /// # Arguments
    /// * `database_url` - The URL of the PostgreSQL database to connect to
    ///
    /// # Returns
    /// A new state instance wrapped in `Result`
    ///
    /// # Errors
    /// Returns `AtomaStateManagerError` if:
    /// - Failed to connect to database
    /// - Failed to run database migrations
    pub async fn new_from_url(database_url: &str) -> Result<Self> {
        let db = PgPool::connect(database_url).await?;
        sqlx::migrate!("./src/migrations").run(&db).await?;
        Ok(Self { db })
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
    ///   - `Err(AtomaStateManagerError)`: An error if the task is not found or other database operation fails.
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
        let task = sqlx::query("SELECT * FROM tasks WHERE task_small_id = $1")
            .bind(task_small_id)
            .fetch_one(&self.db)
            .await?;
        Ok(Task::from_row(&task)?)
    }

    /// Retrieves all tasks from the database.
    ///
    /// This method fetches all task records from the `tasks` table in the database.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<Task>>`: A result containing either:
    ///   - `Ok(Vec<Task>)`: A vector of all `Task` objects in the database.
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into `Task` objects.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn list_all_tasks(state_manager: &AtomaStateManager) -> Result<Vec<Task>, AtomaStateManagerError> {
    ///     state_manager.get_all_tasks().await
    /// }
    /// ```
    #[tracing::instrument(level = "trace", skip_all, fields(function = "get_all_tasks"))]
    pub async fn get_all_tasks(&self) -> Result<Vec<Task>> {
        let tasks = sqlx::query("SELECT * FROM tasks")
            .fetch_all(&self.db)
            .await?;
        tasks
            .into_iter()
            .map(|task| Task::from_row(&task).map_err(AtomaStateManagerError::from))
            .collect()
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
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's a constraint violation (e.g., duplicate `task_small_id` or `task_id`).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, Task};
    ///
    /// async fn add_task(state_manager: &mut AtomaStateManager, task: Task) -> Result<(), AtomaStateManagerError> {
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
                valid_until_epoch, security_level, minimum_reputation_score
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (task_small_id) DO NOTHING",
        )
        .bind(task.task_small_id)
        .bind(task.task_id)
        .bind(task.role)
        .bind(task.model_name)
        .bind(task.is_deprecated)
        .bind(task.valid_until_epoch)
        .bind(task.security_level)
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
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - The task with the specified `task_small_id` doesn't exist.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn deprecate_task(state_manager: &AtomaStateManager, task_small_id: i64) -> Result<(), AtomaStateManagerError> {
    ///     state_manager.deprecate_task(task_small_id).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(task_small_id = %task_small_id)
    )]
    pub async fn deprecate_task(&self, task_small_id: i64, epoch: i64) -> Result<()> {
        sqlx::query("UPDATE tasks SET is_deprecated = TRUE, deprecated_at_epoch = $1 WHERE task_small_id = $2")
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
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into `Task` objects.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn get_node_tasks(state_manager: &AtomaStateManager, node_small_id: i64) -> Result<Vec<Task>, AtomaStateManagerError> {
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
            WHERE node_subscriptions.node_small_id = $1",
        )
        .bind(node_small_id)
        .fetch_all(&self.db)
        .await?;
        tasks
            .into_iter()
            .map(|task| Task::from_row(&task).map_err(AtomaStateManagerError::from))
            .collect()
    }

    /// Retrieves all node subscriptions for a given set of node IDs.
    ///
    /// This method fetches all subscription records from the `node_subscriptions` table
    /// that match any of the provided node IDs.
    ///
    /// # Arguments
    ///
    /// * `node_small_ids` - A slice of node IDs to fetch subscriptions for.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<NodeSubscription>>`: A result containing either:
    ///   - `Ok(Vec<NodeSubscription>)`: A vector of `NodeSubscription` objects representing all found subscriptions.
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into `NodeSubscription` objects.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, NodeSubscription};
    ///
    /// async fn get_subscriptions(state_manager: &AtomaStateManager) -> Result<Vec<NodeSubscription>, AtomaStateManagerError> {
    ///     let node_ids = vec![1, 2, 3];
    ///     state_manager.get_all_node_subscriptions(&node_ids).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(node_small_ids = ?node_small_ids)
    )]
    pub async fn get_all_node_subscriptions(
        &self,
        node_small_ids: &[i64],
    ) -> Result<Vec<NodeSubscription>> {
        let mut query_builder = build_query_with_in(
            "SELECT * FROM node_subscriptions",
            "node_small_id",
            node_small_ids,
            None,
        );

        let subscriptions = query_builder.build().fetch_all(&self.db).await?;

        subscriptions
            .into_iter()
            .map(|subscription| {
                NodeSubscription::from_row(&subscription).map_err(AtomaStateManagerError::from)
            })
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
    ///   - `Err(AtomaStateManagerError)` if there's a database error.
    ///
    /// # Errors
    ///
    /// This function will return an error if the database query fails to execute.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn check_subscription(state_manager: &AtomaStateManager, node_small_id: i64, task_small_id: i64) -> Result<bool, AtomaStateManagerError> {
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
            "SELECT COUNT(*) FROM node_subscriptions WHERE node_small_id = $1 AND task_small_id = $2",
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
    /// * `price_per_one_million_compute_units` - The price per compute unit for the subscription.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's a constraint violation (e.g., duplicate subscription).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn subscribe_node_to_task(state_manager: &AtomaStateManager, node_small_id: i64, task_small_id: i64, price_per_one_million_compute_units: i64) -> Result<(), AtomaStateManagerError> {
    ///     state_manager.subscribe_node_to_task(node_small_id, task_small_id, price_per_one_million_compute_units).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(
            node_small_id = %node_small_id,
            task_small_id = %task_small_id,
            price_per_one_million_compute_units = %price_per_one_million_compute_units,
            max_num_compute_units = %max_num_compute_units
        )
    )]
    pub async fn subscribe_node_to_task(
        &self,
        node_small_id: i64,
        task_small_id: i64,
        price_per_one_million_compute_units: i64,
        max_num_compute_units: i64,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO node_subscriptions
                (node_small_id, task_small_id, price_per_one_million_compute_units, max_num_compute_units, valid)
                VALUES ($1, $2, $3, $4, TRUE)
                ON CONFLICT (task_small_id, node_small_id) DO NOTHING",
        )
            .bind(node_small_id)
            .bind(task_small_id)
            .bind(price_per_one_million_compute_units)
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
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
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
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, NodeSubscription};
    ///
    /// async fn get_subscription(state_manager: &AtomaStateManager, task_small_id: i64) -> Result<NodeSubscription, AtomaStateManagerError> {
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
        let subscription = sqlx::query("SELECT * FROM node_subscriptions WHERE task_small_id = $1")
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
    /// * `price_per_one_million_compute_units` - The new price per compute unit for the subscription.
    /// * `max_num_compute_units` - The new maximum number of compute units for the subscription.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - The specified node subscription doesn't exist.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn update_subscription(state_manager: &AtomaStateManager) -> Result<(), AtomaStateManagerError> {
    ///     state_manager.update_node_subscription(1, 2, 100, 1000).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(
            node_small_id = %node_small_id,
            task_small_id = %task_small_id,
            price_per_one_million_compute_units = %price_per_one_million_compute_units,
            max_num_compute_units = %max_num_compute_units
        )
    )]
    pub async fn update_node_subscription(
        &self,
        node_small_id: i64,
        task_small_id: i64,
        price_per_one_million_compute_units: i64,
        max_num_compute_units: i64,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE node_subscriptions SET price_per_one_million_compute_units = $1, max_num_compute_units = $2 WHERE node_small_id = $3 AND task_small_id = $4",
        )
            .bind(price_per_one_million_compute_units)
            .bind(max_num_compute_units)
            .bind(node_small_id)
            .bind(task_small_id)
            .execute(&self.db)
            .await?;
        Ok(())
    }

    /// Inserts a new node registration event into the database.
    ///
    /// This method records a new node registration by inserting the node's details into the `nodes` table.
    /// Each record consists of a small ID (used for internal references), a node ID (unique identifier),
    /// and the node's Sui blockchain address.
    ///
    /// # Arguments
    ///
    /// * `node_small_id` - An internal identifier for the node, used for efficient database operations
    /// * `node_id` - The unique identifier string for the node
    /// * `node_sui_address` - The Sui blockchain address associated with the node
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError))
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute
    /// - There's a constraint violation (e.g., duplicate node_small_id or node_id)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn register_node(state_manager: &AtomaStateManager) -> Result<(), AtomaStateManagerError> {
    ///     state_manager.insert_node_registration_event(
    ///         1,                              // node_small_id
    ///         "node_123".to_string(),        // node_id
    ///         "0x123...abc".to_string()      // node_sui_address
    ///     ).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(node_small_id = %node_small_id, node_id = %node_id, node_sui_address = %node_sui_address)
    )]
    pub async fn insert_node_registration_event(
        &self,
        node_small_id: i64,
        node_id: String,
        node_sui_address: String,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO nodes (node_small_id, node_id, node_sui_address) VALUES ($1, $2, $3)",
        )
        .bind(node_small_id)
        .bind(node_id)
        .bind(node_sui_address)
        .execute(&self.db)
        .await?;
        Ok(())
    }

    /// Retrieves a node from the database by its small ID.
    ///
    /// This method fetches a node record from the `nodes` table using the provided `node_small_id`.
    ///
    /// # Arguments
    ///
    /// * `node_small_id` - The unique small identifier of the node to retrieve.
    ///
    /// # Returns
    ///
    /// - `Result<Node>`: A result containing either:
    ///   - `Ok(Node)`: A `Node` object representing the requested node.
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - The specified node doesn't exist.
    /// - There's an issue converting the database row into a `Node` object.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, Node};
    ///
    /// async fn get_node(state_manager: &AtomaStateManager, node_small_id: i64) -> Result<Node, AtomaStateManagerError> {
    ///     state_manager.get_node_by_small_id(node_small_id).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(node_small_id = %node_small_id)
    )]
    pub async fn get_node_by_small_id(&self, node_small_id: i64) -> Result<Node> {
        let node = sqlx::query("SELECT * FROM nodes WHERE node_small_id = $1")
            .bind(node_small_id)
            .fetch_one(&self.db)
            .await?;
        Ok(Node::from_row(&node)?)
    }

    /// Verifies the ownership of a node's small ID by checking the Sui address.
    ///
    /// This method fetches the node's small ID from the database and checks if the provided Sui address
    /// matches the address stored in the database.
    ///
    /// # Arguments
    ///
    /// * `node_small_id` - The small ID of the node to verify.
    /// * `sui_address` - The Sui address to verify against the node's small ID.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if the database query fails to execute.
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(node_small_id = %node_small_id, sui_address = %sui_address)
    )]
    pub async fn verify_node_small_id_ownership(
        &self,
        node_small_id: u64,
        sui_address: String,
    ) -> Result<()> {
        let exists = sqlx::query(
            "SELECT EXISTS(SELECT 1 FROM nodes WHERE node_small_id = $1 AND node_sui_address = $2)",
        )
        .bind(node_small_id as i64)
        .bind(sui_address)
        .fetch_one(&self.db)
        .await?
        .get::<bool, _>(0);

        if !exists {
            return Err(AtomaStateManagerError::NodeSmallIdOwnershipVerificationFailed);
        }

        Ok(())
    }

    /// Unsubscribes a node from a task.
    ///
    /// This method deletes the subscription from the `node_subscriptions` table for the specified node and task combination.
    ///
    /// # Arguments
    ///
    /// * `node_small_id` - The unique identifier of the node to be unsubscribed.
    /// * `task_small_id` - The unique identifier of the task from which the node is unsubscribed.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    /// # Errors
    ///
    /// This function will return an error if the database query fails to execute.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn unsubscribe_node(state_manager: &AtomaStateManager, node_small_id: i64, task_small_id: i64) -> Result<(), AtomaStateManagerError> {
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
        sqlx::query(
            "DELETE FROM node_subscriptions WHERE node_small_id = $1 AND task_small_id = $2",
        )
        .bind(node_small_id)
        .bind(task_small_id)
        .execute(&self.db)
        .await?;
        Ok(())
    }

    /// Inserts or updates a node's public key rotation record in the database.
    ///
    /// This method inserts a new entry into the `node_public_key_rotations` table with the provided
    /// epoch, node ID, public key bytes, and TEE remote attestation bytes. If a record with the same
    /// `node_small_id` already exists, it updates the existing record with the new values.
    ///
    /// # Arguments
    ///
    /// * `epoch` - The epoch at which the public key rotation is recorded.
    /// * `node_small_id` - The unique identifier of the node.
    /// * `public_key_bytes` - A vector of bytes representing the node's public key.
    /// * `tee_remote_attestation_bytes` - A vector of bytes representing the TEE remote attestation.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if the database query fails to execute.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn rotate_node_key(state_manager: &AtomaStateManager, epoch: u64, node_id: u64, public_key: Vec<u8>, attestation: Vec<u8>) -> Result<(), AtomaStateManagerError> {
    ///     state_manager.insert_node_public_key_rotation(epoch, node_id, public_key, attestation).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(epoch = %epoch, node_small_id = %node_small_id)
    )]
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_node_public_key_rotation(
        &self,
        epoch: u64,
        key_rotation_counter: u64,
        node_small_id: u64,
        public_key_bytes: Vec<u8>,
        evidence_data_bytes: Vec<u8>,
        device_type: u16,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO node_public_key_rotations (
                epoch, 
                key_rotation_counter, 
                node_small_id, 
                public_key_bytes, 
                evidence_data_bytes,
                device_type
            ) 
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (node_small_id, device_type)
            DO UPDATE SET
                epoch = EXCLUDED.epoch,
                key_rotation_counter = EXCLUDED.key_rotation_counter,
                public_key_bytes = EXCLUDED.public_key_bytes,
                evidence_data_bytes = EXCLUDED.evidence_data_bytes
            ",
        )
        .bind(epoch as i64)
        .bind(key_rotation_counter as i64)
        .bind(node_small_id as i64)
        .bind(public_key_bytes)
        .bind(evidence_data_bytes)
        .bind(i32::from(device_type))
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
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into a `Stack` object.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn get_stack(state_manager: &AtomaStateManager, stack_small_id: i64) -> Result<Stack, AtomaStateManagerError> {
    ///     state_manager.get_stack(stack_id).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack_small_id)
    )]
    pub async fn get_stack(&self, stack_small_id: i64) -> Result<Stack> {
        let stack = sqlx::query("SELECT * FROM stacks WHERE stack_small_id = $1")
            .bind(stack_small_id)
            .fetch_one(&self.db)
            .await?;
        Ok(Stack::from_row(&stack)?)
    }

    /// Retrieves multiple stacks from the database by their small IDs.
    ///
    /// This method efficiently fetches multiple stack records from the `stacks` table in a single query
    /// by using an IN clause with the provided stack IDs.
    ///
    /// # Arguments
    ///
    /// * `stack_small_ids` - A slice of stack IDs to retrieve from the database.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<Stack>>`: A result containing either:
    ///   - `Ok(Vec<Stack>)`: A vector of `Stack` objects corresponding to the requested IDs.
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into `Stack` objects.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, Stack};
    ///
    /// async fn get_multiple_stacks(state_manager: &AtomaStateManager) -> Result<Vec<Stack>, AtomaStateManagerError> {
    ///     let stack_ids = &[1, 2, 3]; // IDs of stacks to retrieve
    ///     state_manager.get_stacks(stack_ids).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_ids = ?stack_small_ids)
    )]
    pub async fn get_stacks(&self, stack_small_ids: &[i64]) -> Result<Vec<Stack>> {
        let mut query_builder = build_query_with_in(
            "SELECT * FROM stacks",
            "stack_small_id",
            stack_small_ids,
            None,
        );
        let stacks = query_builder.build().fetch_all(&self.db).await?;
        stacks
            .into_iter()
            .map(|stack| Stack::from_row(&stack).map_err(AtomaStateManagerError::from))
            .collect()
    }

    /// Retrieves all stacks associated with the given node IDs.
    ///
    /// This method fetches all stack records from the database that are associated with any
    /// of the provided node IDs in the `node_small_ids` array.
    ///
    /// # Arguments
    ///
    /// * `node_small_ids` - A slice of node IDs to fetch stacks for.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<Stack>>`: A result containing either:
    ///   - `Ok(Vec<Stack>)`: A vector of `Stack` objects representing all stacks found.
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into `Stack` objects.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, Stack};
    ///
    /// async fn get_stacks(state_manager: &AtomaStateManager) -> Result<Vec<Stack>, AtomaStateManagerError> {
    ///     let node_ids = &[1, 2, 3];
    ///     state_manager.get_all_stacks(node_ids).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(node_small_ids = ?node_small_ids)
    )]
    pub async fn get_stacks_by_node_small_ids(&self, node_small_ids: &[i64]) -> Result<Vec<Stack>> {
        let mut query_builder = build_query_with_in(
            "SELECT * FROM stacks",
            "selected_node_id",
            node_small_ids,
            None,
        );

        let stacks = query_builder.build().fetch_all(&self.db).await?;

        stacks
            .into_iter()
            .map(|stack| Stack::from_row(&stack).map_err(AtomaStateManagerError::from))
            .collect()
    }

    /// Retrieves all stacks associated with a specific node ID.
    ///
    /// This method fetches all stack records from the `stacks` table that are associated
    /// with the provided `node_small_id`.
    ///
    /// # Arguments
    ///
    /// * `node_small_id` - The unique identifier of the node whose stacks should be retrieved.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<Stack>>`: A result containing either:
    ///   - `Ok(Vec<Stack>)`: A vector of `Stack` objects associated with the given node ID.
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into `Stack` objects.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, Stack};
    ///
    /// async fn get_node_stacks(state_manager: &AtomaStateManager, node_small_id: i64) -> Result<Vec<Stack>, AtomaStateManagerError> {
    ///     state_manager.get_stack_by_id(node_small_id).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(node_small_id = %node_small_id)
    )]
    pub async fn get_stack_by_id(&self, node_small_id: i64) -> Result<Vec<Stack>> {
        let stacks = sqlx::query("SELECT * FROM stacks WHERE selected_node_id = $1")
            .bind(node_small_id)
            .fetch_all(&self.db)
            .await?;
        stacks
            .into_iter()
            .map(|stack| Stack::from_row(&stack).map_err(AtomaStateManagerError::from))
            .collect()
    }

    /// Retrieves stacks that are almost filled beyond a specified fraction threshold.
    ///
    /// This method fetches all stacks from the database where:
    /// 1. The stack belongs to one of the specified nodes (`node_small_ids`)
    /// 2. The number of already computed units exceeds the specified fraction of total compute units
    ///    (i.e., `already_computed_units > num_compute_units * fraction`)
    ///
    /// # Arguments
    ///
    /// * `node_small_ids` - A slice of node IDs to check for almost filled stacks.
    /// * `fraction` - A floating-point value between 0 and 1 representing the threshold fraction.
    ///                 For example, 0.8 means stacks that are more than 80% filled.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<Stack>>`: A result containing either:
    ///   - `Ok(Vec<Stack>)`: A vector of `Stack` objects that meet the filling criteria.
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into `Stack` objects.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn get_filled_stacks(state_manager: &AtomaStateManager) -> Result<Vec<Stack>, AtomaStateManagerError> {
    ///     let node_ids = &[1, 2, 3];  // Check stacks for these nodes
    ///     let threshold = 0.8;        // Look for stacks that are 80% or more filled
    ///
    ///     state_manager.get_almost_filled_stacks(node_ids, threshold).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(node_small_ids = ?node_small_ids, fraction = %fraction)
    )]
    pub async fn get_almost_filled_stacks(
        &self,
        node_small_ids: &[i64],
        fraction: f64,
    ) -> Result<Vec<Stack>> {
        Ok(sqlx::query_as::<_, Stack>(
            r"
            SELECT *
            FROM stacks
            WHERE selected_node_id = ANY($1)
            AND CASE
                WHEN num_compute_units = 0 THEN true
                ELSE (already_computed_units::float / num_compute_units::float) > $2
            END
            ",
        )
        .bind(node_small_ids)
        .bind(fraction)
        .fetch_all(&self.db)
        .await?)
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
    ///   - `Err(AtomaStateManagerError)`: If there was an error during the database operation.
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
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, Stack};
    ///
    /// async fn reserve_compute_units(state_manager: &AtomaStateManager) -> Result<Option<Stack>, AtomaStateManagerError> {
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
            sui_address = %sui_address,
            num_compute_units = %num_compute_units
        )
    )]
    pub async fn get_available_stack_with_compute_units(
        &self,
        stack_small_id: i64,
        sui_address: &str,
        num_compute_units: i64,
    ) -> Result<Option<Stack>> {
        // Single query that updates and returns the modified row
        let maybe_stack = sqlx::query_as::<_, Stack>(
            r"
            UPDATE stacks
            SET already_computed_units = already_computed_units + $1
            WHERE stack_small_id = $2
            AND owner_address = $3
            AND num_compute_units - already_computed_units >= $1
            AND in_settle_period = false
            RETURNING *
            ",
        )
        .bind(num_compute_units)
        .bind(stack_small_id)
        .bind(sui_address)
        .fetch_optional(&self.db)
        .await?;

        Ok(maybe_stack)
    }

    /// Inserts a new stack into the database if it doesn't already exist.
    ///
    /// This method attempts to insert a new entry into the `stacks` table with the provided stack details.
    /// If a stack with the same `stack_small_id` already exists, the operation will silently succeed without
    /// modifying the existing record. This idempotent behavior is relevant as a `Stack` might be inserted
    /// multiple times, if a request is sent to the Atoma service, while the event subscriber is still processing
    /// the events from previous transactions.
    ///
    /// # Arguments
    ///
    /// * `stack` - The `Stack` object to be inserted into the database.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the provided `Stack` object into a database row.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn insert_stack(state_manager: &AtomaStateManager, stack: Stack) -> Result<(), AtomaStateManagerError> {
    ///     // If a stack with the same stack_small_id exists, this will succeed without modifying it
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
            price = %stack.price_per_one_million_compute_units)
    )]
    pub async fn insert_new_stack(&self, stack: Stack) -> Result<()> {
        sqlx::query(
            "INSERT INTO stacks
                (owner_address, stack_small_id, stack_id, task_small_id, selected_node_id, num_compute_units, price_per_one_million_compute_units, already_computed_units, in_settle_period, total_hash, num_total_messages)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (stack_small_id) DO UPDATE
            SET already_computed_units = stacks.already_computed_units + $8
            WHERE stacks.stack_small_id = $2;",
        )
            .bind(stack.owner_address)
            .bind(stack.stack_small_id)
            .bind(stack.stack_id)
            .bind(stack.task_small_id)
            .bind(stack.selected_node_id)
            .bind(stack.num_compute_units)
            .bind(stack.price_per_one_million_compute_units)
            .bind(stack.already_computed_units)
            .bind(stack.in_settle_period)
            .bind(stack.total_hash)
            .bind(stack.num_total_messages)
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
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into a `Stack` object.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn update_computed_units(state_manager: &AtomaStateManager, stack_small_id: i64, already_computed_units: i64) -> Result<(), AtomaStateManagerError> {
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
        sqlx::query("UPDATE stacks SET already_computed_units = $1 WHERE stack_small_id = $2")
            .bind(already_computed_units)
            .bind(stack_small_id)
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
    /// * `estimated_total_compute_units` - The estimated total number of compute units.
    /// * `total_compute_units` - The total number of compute units.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn update_stack_num_compute_units(state_manager: &AtomaStateManager, stack_small_id: i64, estimated_total_compute_units: i64, total_compute_units: i64) -> Result<(), AtomaStateManagerError> {
    ///     state_manager.update_stack_num_compute_units(stack_small_id, estimated_total_compute_units, total_compute_units).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack_small_id, estimated_total_compute_units = %estimated_total_compute_units, total_compute_units = %total_compute_units)
    )]
    pub async fn update_stack_num_compute_units(
        &self,
        stack_small_id: i64,
        estimated_total_compute_units: i64,
        total_compute_units: i64,
    ) -> Result<()> {
        let result = sqlx::query(
            "UPDATE stacks
            SET already_computed_units = already_computed_units - ($1 - $2)
            WHERE stack_small_id = $3",
        )
        .bind(estimated_total_compute_units)
        .bind(total_compute_units)
        .bind(stack_small_id)
        .execute(&self.db)
        .await?;

        if result.rows_affected() == 0 {
            return Err(AtomaStateManagerError::StackNotFound);
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
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
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
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, StackSettlementTicket};
    ///
    /// async fn get_settlement_ticket(state_manager: &AtomaStateManager, stack_small_id: i64) -> Result<StackSettlementTicket, AtomaStateManagerError> {
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
            sqlx::query("SELECT * FROM stack_settlement_tickets WHERE stack_small_id = $1")
                .bind(stack_small_id)
                .fetch_one(&self.db)
                .await?;
        Ok(StackSettlementTicket::from_row(&stack_settlement_ticket)?)
    }

    /// Retrieves multiple stack settlement tickets from the database.
    ///
    /// This method fetches multiple settlement tickets from the `stack_settlement_tickets` table
    /// based on the provided `stack_small_ids`.
    ///
    /// # Arguments
    ///
    /// * `stack_small_ids` - A slice of stack IDs whose settlement tickets should be retrieved.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<StackSettlementTicket>>`: A result containing a vector of `StackSettlementTicket` objects.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, StackSettlementTicket};
    ///
    /// async fn get_settlement_tickets(state_manager: &AtomaStateManager, stack_small_ids: &[i64]) -> Result<Vec<StackSettlementTicket>, AtomaStateManagerError> {
    ///     state_manager.get_stack_settlement_tickets(stack_small_ids).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_ids = ?stack_small_ids)
    )]
    pub async fn get_stack_settlement_tickets(
        &self,
        stack_small_ids: &[i64],
    ) -> Result<Vec<StackSettlementTicket>> {
        let mut query_builder = build_query_with_in(
            "SELECT * FROM stack_settlement_tickets",
            "stack_small_id",
            stack_small_ids,
            None,
        );

        let stack_settlement_tickets = query_builder.build().fetch_all(&self.db).await?;

        stack_settlement_tickets
            .into_iter()
            .map(|row| StackSettlementTicket::from_row(&row).map_err(AtomaStateManagerError::from))
            .collect()
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
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the provided `StackSettlementTicket` object into a database row.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, StackSettlementTicket};
    ///
    /// async fn insert_settlement_ticket(state_manager: &AtomaStateManager, stack_settlement_ticket: StackSettlementTicket) -> Result<(), AtomaStateManagerError> {
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
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (stack_small_id, selected_node_id) DO NOTHING",
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
        sqlx::query("UPDATE stacks SET in_settle_period = true WHERE stack_small_id = $1")
            .bind(stack_settlement_ticket.stack_small_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Updates the total hash and increments the total number of messages for a stack.
    ///
    /// This method updates the `total_hash` field in the `stacks` table by appending a new hash
    /// to the existing hash and increments the `num_total_messages` field by 1 for the specified `stack_small_id`.
    ///
    /// # Arguments
    ///
    /// * `stack_small_id` - The unique small identifier of the stack to update.
    /// * `new_hash` - A 32-byte array representing the new hash to append to the existing total hash.
    ///
    /// # Returns
    ///
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database transaction fails to begin, execute, or commit.
    /// - The specified stack is not found.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn update_hash(state_manager: &AtomaStateManager) -> Result<(), AtomaStateManagerError> {
    ///     let stack_small_id = 1;
    ///     let new_hash = [0u8; 32]; // Example hash
    ///
    ///     state_manager.update_stack_total_hash(stack_small_id, new_hash).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack_small_id, new_hash = ?new_hash)
    )]
    pub async fn update_stack_total_hash(
        &self,
        stack_small_id: i64,
        new_hash: [u8; 32],
    ) -> Result<()> {
        let rows_affected = sqlx::query(
            "UPDATE stacks
            SET total_hash = total_hash || $1,
                num_total_messages = num_total_messages + 1
            WHERE stack_small_id = $2",
        )
        .bind(&new_hash[..])
        .bind(stack_small_id)
        .execute(&self.db)
        .await?
        .rows_affected();

        if rows_affected == 0 {
            return Err(AtomaStateManagerError::StackNotFound);
        }

        Ok(())
    }

    /// Retrieves the total hash for a specific stack.
    ///
    /// This method fetches the `total_hash` field from the `stacks` table for the given `stack_small_id`.
    ///
    /// # Arguments
    ///
    /// * `stack_small_id` - The unique small identifier of the stack whose total hash is to be retrieved.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<u8>>`: A result containing either:
    ///   - `Ok(Vec<u8>)`: A byte vector representing the total hash of the stack.
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn get_total_hash(state_manager: &AtomaStateManager, stack_small_id: i64) -> Result<Vec<u8>, AtomaStateManagerError> {
    ///     state_manager.get_stack_total_hash(stack_small_id).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_id = %stack_small_id)
    )]
    pub async fn get_stack_total_hash(&self, stack_small_id: i64) -> Result<Vec<u8>> {
        let total_hash = sqlx::query_scalar::<_, Vec<u8>>(
            "SELECT total_hash FROM stacks WHERE stack_small_id = $1",
        )
        .bind(stack_small_id)
        .fetch_one(&self.db)
        .await?;
        Ok(total_hash)
    }

    /// Retrieves the total hashes for multiple stacks in a single query.
    ///
    /// This method efficiently fetches the `total_hash` field from the `stacks` table for all
    /// provided stack IDs in a single database query.
    ///
    /// # Arguments
    ///
    /// * `stack_small_ids` - A slice of stack IDs whose total hashes should be retrieved.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<Vec<u8>>>`: A result containing either:
    ///   - `Ok(Vec<Vec<u8>>)`: A vector of byte vectors, where each inner vector represents
    ///     the total hash of a stack. The order corresponds to the order of results returned
    ///     by the database query.
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue retrieving the hash data from the result rows.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn get_hashes(state_manager: &AtomaStateManager) -> Result<Vec<Vec<u8>>, AtomaStateManagerError> {
    ///     let stack_ids = &[1, 2, 3]; // IDs of stacks to fetch hashes for
    ///     state_manager.get_all_total_hashes(stack_ids).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(stack_small_ids = ?stack_small_ids)
    )]
    pub async fn get_all_total_hashes(&self, stack_small_ids: &[i64]) -> Result<Vec<Vec<u8>>> {
        let mut query_builder = build_query_with_in(
            "SELECT total_hash FROM stacks",
            "stack_small_id",
            stack_small_ids,
            None,
        );

        Ok(query_builder
            .build()
            .fetch_all(&self.db)
            .await?
            .iter()
            .map(|row| row.get("total_hash"))
            .collect())
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
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
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
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn update_settlement_ticket(state_manager: &AtomaStateManager) -> Result<(), AtomaStateManagerError> {
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

        // First query remains the same - get existing data
        let row = sqlx::query(
            "SELECT committed_stack_proofs, stack_merkle_leaves, requested_attestation_nodes, already_attested_nodes
             FROM stack_settlement_tickets
             WHERE stack_small_id = $1",
        )
        .bind(stack_small_id)
        .fetch_one(&mut *tx)
        .await?;

        let mut committed_stack_proofs: Vec<u8> = row.get("committed_stack_proofs");
        let mut current_merkle_leaves: Vec<u8> = row.get("stack_merkle_leaves");
        let requested_nodes: String = row.get("requested_attestation_nodes");
        let requested_nodes: Vec<i64> = serde_json::from_str(&requested_nodes)?;

        // Get current attested nodes and handle null case
        let already_attested: Option<String> = row.get("already_attested_nodes");
        let mut attested_nodes: Vec<i64> = already_attested
            .map(|nodes| serde_json::from_str(&nodes).unwrap_or_default())
            .unwrap_or_default();

        // Find the index and update binary data as before
        let index = requested_nodes
            .iter()
            .position(|&id| id == attestation_node_id)
            .ok_or_else(|| AtomaStateManagerError::AttestationNodeNotFound(attestation_node_id))?;

        let start = (index + 1) * 32;
        let end = start + 32;
        if end > current_merkle_leaves.len() {
            return Err(AtomaStateManagerError::InvalidMerkleLeafLength);
        }
        if end > committed_stack_proofs.len() {
            return Err(AtomaStateManagerError::InvalidCommittedStackProofLength);
        }

        current_merkle_leaves[start..end].copy_from_slice(&stack_merkle_leaf[..32]);
        committed_stack_proofs[start..end].copy_from_slice(&committed_stack_proof[..32]);

        // Update attested nodes list
        attested_nodes.push(attestation_node_id);
        let attested_nodes_json = serde_json::to_string(&attested_nodes)?;

        // Simplified update query
        sqlx::query(
            "UPDATE stack_settlement_tickets
            SET committed_stack_proofs = $1,
                stack_merkle_leaves = $2,
                already_attested_nodes = $3
            WHERE stack_small_id = $4",
        )
        .bind(committed_stack_proofs)
        .bind(current_merkle_leaves)
        .bind(attested_nodes_json)
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
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - The specified stack settlement ticket doesn't exist.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn settle_ticket(state_manager: &AtomaStateManager) -> Result<(), AtomaStateManagerError> {
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
        sqlx::query("UPDATE stack_settlement_tickets SET dispute_settled_at_epoch = $1 WHERE stack_small_id = $2")
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
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - The specified stack settlement ticket doesn't exist.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::AtomaStateManager;
    ///
    /// async fn claim_settlement_ticket(state_manager: &AtomaStateManager) -> Result<(), AtomaStateManagerError> {
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
                SET user_refund_amount = $1,
                    is_claimed = true
                WHERE stack_small_id = $2",
        )
        .bind(user_refund_amount)
        .bind(stack_small_id)
        .execute(&self.db)
        .await?;

        Ok(())
    }

    /// Retrieves all stacks that have been claimed for the specified node IDs.
    ///
    /// This method fetches all stack records from the `stacks` table where the `selected_node_id`
    /// matches any of the provided node IDs and the stack is marked as claimed (`is_claimed = true`).
    ///
    /// # Arguments
    ///
    /// * `node_small_ids` - A slice of node IDs to fetch claimed stacks for.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<Stack>>`: A result containing either:
    ///   - `Ok(Vec<Stack>)`: A vector of `Stack` objects representing all claimed stacks found.
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into `Stack` objects.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, Stack};
    ///
    /// async fn get_claimed_stacks(state_manager: &AtomaStateManager) -> Result<Vec<Stack>, AtomaStateManagerError> {
    ///     let node_ids = &[1, 2, 3]; // IDs of nodes to fetch claimed stacks for
    ///     state_manager.get_claimed_stacks(node_ids).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(node_small_ids = ?node_small_ids)
    )]
    pub async fn get_claimed_stacks(
        &self,
        node_small_ids: &[i64],
    ) -> Result<Vec<StackSettlementTicket>> {
        let mut query_builder = build_query_with_in(
            "SELECT * FROM stack_settlement_tickets",
            "selected_node_id",
            node_small_ids,
            Some("is_claimed = true"),
        );

        let stacks = query_builder.build().fetch_all(&self.db).await?;

        stacks
            .into_iter()
            .map(|row| StackSettlementTicket::from_row(&row).map_err(AtomaStateManagerError::from))
            .collect()
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
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into `StackAttestationDispute` objects.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, StackAttestationDispute};
    ///
    /// async fn get_disputes(state_manager: &AtomaStateManager) -> Result<Vec<StackAttestationDispute>, AtomaStateManagerError> {
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
                WHERE stack_small_id = $1 AND attestation_node_id = $2",
        )
        .bind(stack_small_id)
        .bind(attestation_node_id)
        .fetch_all(&self.db)
        .await?;

        disputes
            .into_iter()
            .map(|row| {
                StackAttestationDispute::from_row(&row).map_err(AtomaStateManagerError::from)
            })
            .collect()
    }

    /// Retrieves all attestation disputes filed against the specified nodes.
    ///
    /// This method fetches all disputes from the `stack_attestation_disputes` table where the
    /// specified nodes are the original nodes being disputed against (i.e., where they are
    /// listed as `original_node_id`).
    ///
    /// # Arguments
    ///
    /// * `node_small_ids` - A slice of node IDs to check for disputes against.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<StackAttestationDispute>>`: A result containing either:
    ///   - `Ok(Vec<StackAttestationDispute>)`: A vector of all disputes found against the specified nodes.
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into `StackAttestationDispute` objects.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, StackAttestationDispute};
    ///
    /// async fn get_disputes(state_manager: &AtomaStateManager) -> Result<Vec<StackAttestationDispute>, AtomaStateManagerError> {
    ///     let node_ids = &[1, 2, 3]; // IDs of nodes to check for disputes against
    ///     state_manager.get_against_attestation_disputes(node_ids).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(node_small_ids = ?node_small_ids)
    )]
    pub async fn get_against_attestation_disputes(
        &self,
        node_small_ids: &[i64],
    ) -> Result<Vec<StackAttestationDispute>> {
        let mut query_builder = build_query_with_in(
            "SELECT * FROM stack_attestation_disputes",
            "original_node_id",
            node_small_ids,
            None,
        );

        let disputes = query_builder.build().fetch_all(&self.db).await?;

        disputes
            .into_iter()
            .map(|row| {
                StackAttestationDispute::from_row(&row).map_err(AtomaStateManagerError::from)
            })
            .collect()
    }

    /// Retrieves all attestation disputes where the specified nodes are the attestation providers.
    ///
    /// This method fetches all disputes from the `stack_attestation_disputes` table where the
    /// specified nodes are the attestation providers (i.e., where they are listed as `attestation_node_id`).
    ///
    /// # Arguments
    ///
    /// * `node_small_ids` - A slice of node IDs to check for disputes where they are attestation providers.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<StackAttestationDispute>>`: A result containing either:
    ///   - `Ok(Vec<StackAttestationDispute>)`: A vector of all disputes where the specified nodes are attestation providers.
    ///   - `Err(AtomaStateManagerError)`: An error if the database query fails or if there's an issue parsing the results.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's an issue converting the database rows into `StackAttestationDispute` objects.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, StackAttestationDispute};
    ///
    /// async fn get_disputes(state_manager: &AtomaStateManager) -> Result<Vec<StackAttestationDispute>, AtomaStateManagerError> {
    ///     let node_ids = &[1, 2, 3]; // IDs of nodes to check for disputes as attestation providers
    ///     state_manager.get_own_attestation_disputes(node_ids).await
    /// }
    /// ```
    #[tracing::instrument(
        level = "trace",
        skip_all,
        fields(node_small_ids = ?node_small_ids)
    )]
    pub async fn get_own_attestation_disputes(
        &self,
        node_small_ids: &[i64],
    ) -> Result<Vec<StackAttestationDispute>> {
        let mut query_builder = build_query_with_in(
            "SELECT * FROM stack_attestation_disputes",
            "attestation_node_id",
            node_small_ids,
            None,
        );

        let disputes = query_builder.build().fetch_all(&self.db).await?;

        disputes
            .into_iter()
            .map(|row| {
                StackAttestationDispute::from_row(&row).map_err(AtomaStateManagerError::from)
            })
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
    /// - `Result<()>`: A result indicating success (Ok(())) or failure (Err(AtomaStateManagerError)).
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The database query fails to execute.
    /// - There's a constraint violation (e.g., duplicate primary key).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atoma_node::atoma_state::{AtomaStateManager, StackAttestationDispute};
    ///
    /// async fn add_dispute(state_manager: &AtomaStateManager, dispute: StackAttestationDispute) -> Result<(), AtomaStateManagerError> {
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
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (stack_small_id, attestation_node_id) DO NOTHING",
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
pub enum AtomaStateManagerError {
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
    #[error("Failed to retrieve existing total hash for stack: `{0}`")]
    FailedToRetrieveExistingTotalHash(i64),
    #[error("Failed to send result to channel")]
    ChannelSendError,
    #[error("Failed to create database directory")]
    FailedToCreateDatabaseDirectory(#[from] std::io::Error),
    #[error("Failed to migrate database")]
    FailedToMigrateDatabase(#[from] sqlx::migrate::MigrateError),
    #[error("Node small ID ownership verification failed")]
    NodeSmallIdOwnershipVerificationFailed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use atoma_utils::test::POSTGRES_TEST_DB_URL;

    async fn setup_test_db() -> AtomaState {
        AtomaState::new_from_url(POSTGRES_TEST_DB_URL)
            .await
            .unwrap()
    }

    async fn truncate_tables(db: &sqlx::PgPool) {
        // List all your tables here
        sqlx::query(
            "TRUNCATE TABLE
                tasks,
                node_subscriptions,
                stacks,
                stack_settlement_tickets,
                stack_attestation_disputes,
                node_public_key_rotations
            CASCADE",
        )
        .execute(db)
        .await
        .expect("Failed to truncate tables");
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_insert_and_get_task() {
        let state_manager = setup_test_db().await;

        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
            minimum_reputation_score: Some(50),
        };

        state_manager.insert_new_task(task.clone()).await.unwrap();
        let retrieved_task = state_manager.get_task_by_small_id(1).await.unwrap();
        assert_eq!(task, retrieved_task);

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_deprecate_task() {
        let state_manager = setup_test_db().await;

        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
            minimum_reputation_score: Some(50),
        };

        state_manager.insert_new_task(task).await.unwrap();
        state_manager.deprecate_task(1, 100).await.unwrap();

        let deprecated_task = state_manager.get_task_by_small_id(1).await.unwrap();
        assert!(deprecated_task.is_deprecated);

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_subscribed_tasks() {
        let state_manager = setup_test_db().await;

        // Insert two tasks
        let task1 = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
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
            security_level: 1,
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

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_insert_node_registration_event() {
        let state_manager = setup_test_db().await;

        // Test data
        let node_small_id = 42;
        let node_id = "test_node_123".to_string();
        let node_sui_address = "0xabc...def".to_string();

        // Test successful insertion
        let result = state_manager
            .insert_node_registration_event(
                node_small_id,
                node_id.clone(),
                node_sui_address.clone(),
            )
            .await;
        assert!(result.is_ok(), "Failed to insert node registration event");

        let node = state_manager
            .get_node_by_small_id(node_small_id)
            .await
            .unwrap();
        assert_eq!(node.node_small_id, node_small_id);
        assert_eq!(node.node_id, node_id);
        assert_eq!(node.node_sui_address, node_sui_address);

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_verify_node_small_id_ownership() {
        let state_manager = setup_test_db().await;

        // Setup: Create nodes with different addresses
        let nodes = vec![
            (1, "0x123abc".to_string()),
            (2, "0x456def".to_string()),
            (3, "0x789ghi".to_string()),
        ];

        for (node_small_id, node_sui_address) in nodes.clone() {
            sqlx::query(
                "INSERT INTO nodes (node_small_id, node_id, node_sui_address) VALUES ($1, $2, $3)",
            )
            .bind(node_small_id)
            .bind(format!("node{node_small_id}"))
            .bind(node_sui_address.clone())
            .execute(&state_manager.db)
            .await
            .unwrap();
        }

        // Test 1: Verify correct ownership
        let result = state_manager
            .verify_node_small_id_ownership(1, "0x123abc".to_string())
            .await;
        assert!(
            result.is_ok(),
            "Valid ownership verification should succeed"
        );

        // Test 2: Verify incorrect address for existing node
        let result = state_manager
            .verify_node_small_id_ownership(1, "0x456def".to_string())
            .await;
        assert!(
            result.is_err(),
            "Verification with wrong address should fail"
        );
        assert!(
            matches!(
                result.unwrap_err(),
                AtomaStateManagerError::NodeSmallIdOwnershipVerificationFailed
            ),
            "Should return NodeSmallIdOwnershipVerificationFailed error"
        );

        // Test 3: Verify non-existent node
        let result = state_manager
            .verify_node_small_id_ownership(999, "0x999xyz".to_string())
            .await;
        assert!(
            result.is_err(),
            "Verification of non-existent node should fail"
        );
        assert!(
            matches!(
                result.unwrap_err(),
                AtomaStateManagerError::NodeSmallIdOwnershipVerificationFailed
            ),
            "Should return NodeSmallIdOwnershipVerificationFailed error"
        );

        // Test 4: Verify with empty address
        let result = state_manager
            .verify_node_small_id_ownership(1, String::new())
            .await;
        assert!(
            result.is_err(),
            "Verification with empty address should fail"
        );
        assert!(
            matches!(
                result.unwrap_err(),
                AtomaStateManagerError::NodeSmallIdOwnershipVerificationFailed
            ),
            "Should return NodeSmallIdOwnershipVerificationFailed error"
        );

        // Test 5: Verify case sensitivity
        let result = state_manager
            .verify_node_small_id_ownership(1, "0x123ABC".to_string())
            .await;
        assert!(result.is_err(), "Verification should be case sensitive");
        assert!(
            matches!(
                result.unwrap_err(),
                AtomaStateManagerError::NodeSmallIdOwnershipVerificationFailed
            ),
            "Should return NodeSmallIdOwnershipVerificationFailed error"
        );

        // Test 6: Verify with leading/trailing whitespace
        let result = state_manager
            .verify_node_small_id_ownership(1, " 0x123abc ".to_string())
            .await;
        assert!(result.is_err(), "Verification with whitespace should fail");
        assert!(
            matches!(
                result.unwrap_err(),
                AtomaStateManagerError::NodeSmallIdOwnershipVerificationFailed
            ),
            "Should return NodeSmallIdOwnershipVerificationFailed error"
        );

        // Test 7: Verify multiple nodes in sequence
        for (node_small_id, node_address) in nodes {
            let result = state_manager
                .verify_node_small_id_ownership(u64::try_from(node_small_id).unwrap(), node_address)
                .await;
            assert!(
                result.is_ok(),
                "Valid ownership verification should succeed for all valid nodes"
            );
        }

        // Test 8: Verify with maximum u64 value
        let result = state_manager
            .verify_node_small_id_ownership(u64::MAX, "0xffffff".to_string())
            .await;
        assert!(
            result.is_err(),
            "Verification with maximum u64 value should fail"
        );
        assert!(
            matches!(
                result.unwrap_err(),
                AtomaStateManagerError::NodeSmallIdOwnershipVerificationFailed
            ),
            "Should return NodeSmallIdOwnershipVerificationFailed error"
        );

        // Test 9: Verify node zero
        let result = state_manager
            .verify_node_small_id_ownership(0, "0x000000".to_string())
            .await;
        assert!(
            result.is_err(),
            "Verification with node_small_id 0 should fail"
        );
        assert!(
            matches!(
                result.unwrap_err(),
                AtomaStateManagerError::NodeSmallIdOwnershipVerificationFailed
            ),
            "Should return NodeSmallIdOwnershipVerificationFailed error"
        );

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_update_node_subscription() {
        let state_manager = setup_test_db().await;

        // Insert a task
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
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
        assert_eq!(subscription.price_per_one_million_compute_units, 100);
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
        assert_eq!(subscription.price_per_one_million_compute_units, 200);
        assert_eq!(subscription.max_num_compute_units, 2000);

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_insert_and_get_stack() {
        let state_manager = setup_test_db().await;

        // Insert a task and subscribe a node
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
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
            price_per_one_million_compute_units: 1000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![],
            num_total_messages: 0,
        };
        state_manager.insert_new_stack(stack.clone()).await.unwrap();

        // Get the stack and verify
        let retrieved_stack = state_manager.get_stack(1).await.unwrap();
        assert_eq!(stack, retrieved_stack);

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_insert_new_stack_compute_units() {
        let state_manager = setup_test_db().await;

        // Setup: Create a task and subscribe a node
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();

        // Test case 1: Initial insert
        let initial_stack = Stack {
            owner_address: "owner1".to_string(),
            stack_small_id: 1,
            stack_id: "stack1".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 100,
            price_per_one_million_compute_units: 1000,
            already_computed_units: 30,
            in_settle_period: false,
            total_hash: vec![0; 32],
            num_total_messages: 0,
        };
        state_manager.insert_new_stack(initial_stack).await.unwrap();

        // Verify initial insert
        let stack = state_manager.get_stack(1).await.unwrap();
        assert_eq!(stack.already_computed_units, 30);

        // Test case 2: Valid update (within limits)
        let update_stack = Stack {
            already_computed_units: 40,
            ..stack
        };
        state_manager.insert_new_stack(update_stack).await.unwrap();

        // Verify successful update
        let updated_stack = state_manager.get_stack(1).await.unwrap();
        assert_eq!(updated_stack.already_computed_units, 70); // 30 + 40

        // Test case 3: Invalid update (exceeds limit)
        let invalid_stack = Stack {
            already_computed_units: 50, // Would exceed 100 (70 + 50 > 100)
            ..updated_stack
        };
        let result = state_manager.insert_new_stack(invalid_stack).await;

        // Verify error is returned
        assert!(result.is_err());

        // Verify original value remains unchanged
        let final_stack = state_manager.get_stack(1).await.unwrap();
        assert_eq!(final_stack.already_computed_units, 70);

        // Test case 4: Exact limit update
        let exact_limit_stack = Stack {
            already_computed_units: 30, // Exactly reaches limit (70 + 30 = 100)
            ..final_stack
        };
        state_manager
            .insert_new_stack(exact_limit_stack)
            .await
            .unwrap();

        // Verify successful update to exact limit
        let limit_stack = state_manager.get_stack(1).await.unwrap();
        assert_eq!(limit_stack.already_computed_units, 100);

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_insert_new_stack_idempotent() {
        let state_manager = setup_test_db().await;

        // Setup: Create a task and subscribe a node
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();

        // Create a test stack
        let stack = Stack {
            owner_address: "0x123".to_string(),
            stack_small_id: 1,
            stack_id: "stack1".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 10,
            price_per_one_million_compute_units: 1000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![],
            num_total_messages: 0,
        };

        // Test case 1: First insertion should succeed
        state_manager.insert_new_stack(stack.clone()).await.unwrap();
        let retrieved_stack = state_manager.get_stack(1).await.unwrap();
        assert_eq!(stack, retrieved_stack);

        // Test case 2: Second insertion of the same stack should not modify anything
        state_manager.insert_new_stack(stack.clone()).await.unwrap();
        let retrieved_stack_after_second_insert = state_manager.get_stack(1).await.unwrap();
        assert_eq!(stack, retrieved_stack_after_second_insert);

        // Test case 3: Verify no duplicate entries were created
        let all_stacks = state_manager.get_stack_by_id(1).await.unwrap();
        assert_eq!(all_stacks.len(), 1);
        assert_eq!(all_stacks[0], stack);

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_update_computed_units() {
        let state_manager = setup_test_db().await;

        // Insert a task, subscribe a node, and create a stack
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
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
            num_compute_units: 15,
            price_per_one_million_compute_units: 1000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![],
            num_total_messages: 0,
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

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_insert_and_get_stack_settlement_ticket() {
        let state_manager = setup_test_db().await;

        // Insert a task, subscribe a node, and create a stack
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
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
            price_per_one_million_compute_units: 1000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![],
            num_total_messages: 0,
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

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_update_stack_settlement_ticket() {
        let state_manager = setup_test_db().await;

        // Setup: Insert task, subscription, stack, and initial settlement ticket
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
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
            price_per_one_million_compute_units: 1000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![],
            num_total_messages: 0,
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

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_stack_attestation_disputes() {
        let state_manager = setup_test_db().await;

        // Setup: Insert task, subscription, stack, and settlement ticket
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
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
            price_per_one_million_compute_units: 1000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![],
            num_total_messages: 0,
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

        // Inserting a duplicate dispute should be idempotent
        let result = state_manager
            .insert_stack_attestation_dispute(dispute.clone())
            .await;
        assert!(result.is_ok());
        let retrieved_disputes = state_manager
            .get_stack_attestation_disputes(1, 3)
            .await
            .unwrap();
        assert_eq!(retrieved_disputes.len(), 1);
        assert_eq!(retrieved_disputes[0], dispute);

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

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_available_stack_with_compute_units() {
        let state_manager = setup_test_db().await;

        // Setup: Insert a task and subscribe a node
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
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
            price_per_one_million_compute_units: 1000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![],
            num_total_messages: 0,
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

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_update_stack_total_hash() {
        let state_manager = setup_test_db().await;

        // Setup: Insert necessary task and subscription
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
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
            price_per_one_million_compute_units: 1000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![],
            num_total_messages: 0,
        };
        state_manager.insert_new_stack(stack).await.unwrap();

        // Update the total hash
        let new_hash = [42u8; 32];
        state_manager
            .update_stack_total_hash(1, new_hash)
            .await
            .unwrap();

        // Verify the update
        let updated_stack = state_manager.get_stack(1).await.unwrap();
        assert_eq!(updated_stack.total_hash.len(), 32);
        assert_eq!(updated_stack.total_hash, new_hash);
        assert_eq!(updated_stack.num_total_messages, 1);

        // Update the total hash again
        let new_hash = [84; 32];
        state_manager
            .update_stack_total_hash(1, new_hash)
            .await
            .unwrap();

        // Verify the update
        let updated_stack = state_manager.get_stack(1).await.unwrap();
        assert_eq!(updated_stack.total_hash.len(), 64);
        assert_eq!(updated_stack.total_hash[0..32], [42u8; 32]);
        assert_eq!(updated_stack.total_hash[32..64], [84u8; 32]);
        assert_eq!(updated_stack.num_total_messages, 2);

        // Test updating non-existent stack
        let result = state_manager.update_stack_total_hash(999, new_hash).await;
        assert!(matches!(result, Err(AtomaStateManagerError::StackNotFound)));

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_all_node_subscriptions() {
        let state_manager = setup_test_db().await;

        // Insert a task first (required for foreign key constraint)
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();

        // Create multiple node subscriptions
        let subscriptions = vec![
            (1, 100, 1000), // node_id: 1, price: 100, max_units: 1000
            (2, 200, 2000), // node_id: 2, price: 200, max_units: 2000
            (3, 300, 3000), // node_id: 3, price: 300, max_units: 3000
        ];

        // Insert the subscriptions
        for (node_id, price, max_units) in subscriptions {
            state_manager
                .subscribe_node_to_task(node_id, 1, price, max_units)
                .await
                .unwrap();
        }

        // Test 1: Get subscriptions for all nodes
        let all_node_ids = vec![1, 2, 3];
        let result = state_manager
            .get_all_node_subscriptions(&all_node_ids)
            .await
            .unwrap();
        assert_eq!(result.len(), 3);
        assert!(result
            .iter()
            .any(|s| s.node_small_id == 1 && s.price_per_one_million_compute_units == 100));
        assert!(result
            .iter()
            .any(|s| s.node_small_id == 2 && s.price_per_one_million_compute_units == 200));
        assert!(result
            .iter()
            .any(|s| s.node_small_id == 3 && s.price_per_one_million_compute_units == 300));

        // Test 2: Get subscriptions for subset of nodes
        let subset_node_ids = vec![1, 3];
        let result = state_manager
            .get_all_node_subscriptions(&subset_node_ids)
            .await
            .unwrap();
        assert_eq!(result.len(), 2);
        assert!(result
            .iter()
            .any(|s| s.node_small_id == 1 && s.price_per_one_million_compute_units == 100));
        assert!(result
            .iter()
            .any(|s| s.node_small_id == 3 && s.price_per_one_million_compute_units == 300));
        assert!(!result.iter().any(|s| s.node_small_id == 2));

        // Test 3: Get subscriptions for non-existent nodes
        let non_existent_ids = vec![99, 100];
        let result = state_manager
            .get_all_node_subscriptions(&non_existent_ids)
            .await
            .unwrap();
        assert_eq!(result.len(), 0);

        // Test 4: Get subscriptions with empty input
        let empty_ids: Vec<i64> = vec![];
        let result = state_manager
            .get_all_node_subscriptions(&empty_ids)
            .await
            .unwrap();
        assert_eq!(result.len(), 0);

        // Test 5: Verify subscription details
        let single_node_id = vec![1];
        let result = state_manager
            .get_all_node_subscriptions(&single_node_id)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        let subscription = &result[0];
        assert_eq!(subscription.node_small_id, 1);
        assert_eq!(subscription.task_small_id, 1);
        assert_eq!(subscription.price_per_one_million_compute_units, 100);
        assert_eq!(subscription.max_num_compute_units, 1000);
        assert!(subscription.valid);

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_stacks_by_node_small_ids_comprehensive() {
        let state_manager = setup_test_db().await;

        // Setup: Insert a task first (required for foreign key constraint)
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();

        // Setup: Subscribe nodes to the task
        for node_id in 1..=3 {
            state_manager
                .subscribe_node_to_task(node_id, 1, 100, 1000)
                .await
                .unwrap();
        }

        // Test 1: Empty database - no stacks
        let result = state_manager
            .get_stacks_by_node_small_ids(&[1, 2])
            .await
            .unwrap();
        assert!(result.is_empty(), "Expected empty result for no stacks");

        // Setup: Create multiple stacks for different nodes
        let stacks = vec![
            Stack {
                owner_address: "owner1".to_string(),
                stack_small_id: 1,
                stack_id: "stack1".to_string(),
                task_small_id: 1,
                selected_node_id: 1,
                num_compute_units: 100,
                price_per_one_million_compute_units: 1000,
                already_computed_units: 0,
                in_settle_period: false,
                total_hash: vec![1, 2, 3],
                num_total_messages: 1,
            },
            Stack {
                owner_address: "owner2".to_string(),
                stack_small_id: 2,
                stack_id: "stack2".to_string(),
                task_small_id: 1,
                selected_node_id: 1,
                num_compute_units: 200,
                price_per_one_million_compute_units: 2000,
                already_computed_units: 50,
                in_settle_period: true,
                total_hash: vec![4, 5, 6],
                num_total_messages: 2,
            },
            Stack {
                owner_address: "owner3".to_string(),
                stack_small_id: 3,
                stack_id: "stack3".to_string(),
                task_small_id: 1,
                selected_node_id: 2,
                num_compute_units: 300,
                price_per_one_million_compute_units: 3000,
                already_computed_units: 100,
                in_settle_period: false,
                total_hash: vec![7, 8, 9],
                num_total_messages: 3,
            },
        ];

        // Insert all stacks
        for stack in stacks {
            state_manager.insert_new_stack(stack).await.unwrap();
        }

        // Test 2: Get stacks for single node with multiple stacks
        let result = state_manager
            .get_stacks_by_node_small_ids(&[1])
            .await
            .unwrap();
        assert_eq!(result.len(), 2, "Node 1 should have 2 stacks");
        assert!(
            result.iter().all(|s| s.selected_node_id == 1),
            "All stacks should belong to node 1"
        );
        assert!(
            result
                .iter()
                .any(|s| s.stack_small_id == 1 && s.price_per_one_million_compute_units == 1000),
            "Should find stack 1"
        );
        assert!(
            result
                .iter()
                .any(|s| s.stack_small_id == 2 && s.price_per_one_million_compute_units == 2000),
            "Should find stack 2"
        );

        // Test 3: Get stacks for multiple nodes
        let result = state_manager
            .get_stacks_by_node_small_ids(&[1, 2])
            .await
            .unwrap();
        assert_eq!(
            result.len(),
            3,
            "Should find 3 total stacks for nodes 1 and 2"
        );
        assert_eq!(
            result.iter().filter(|s| s.selected_node_id == 1).count(),
            2,
            "Node 1 should have 2 stacks"
        );
        assert_eq!(
            result.iter().filter(|s| s.selected_node_id == 2).count(),
            1,
            "Node 2 should have 1 stack"
        );

        // Test 4: Get stacks with mix of existing and non-existing nodes
        let result = state_manager
            .get_stacks_by_node_small_ids(&[1, 99])
            .await
            .unwrap();
        assert_eq!(
            result.len(),
            2,
            "Should only find stacks for existing node 1"
        );
        assert!(
            result.iter().all(|s| s.selected_node_id == 1),
            "All stacks should belong to node 1"
        );

        // Test 5: Get stacks with all non-existing nodes
        let result = state_manager
            .get_stacks_by_node_small_ids(&[98, 99])
            .await
            .unwrap();
        assert!(
            result.is_empty(),
            "Should find no stacks for non-existing nodes"
        );

        // Test 6: Get stacks with empty input
        let result = state_manager
            .get_stacks_by_node_small_ids(&[])
            .await
            .unwrap();
        assert!(
            result.is_empty(),
            "Should return empty result for empty input"
        );

        // Test 7: Verify stack details are correct
        let result = state_manager
            .get_stacks_by_node_small_ids(&[2])
            .await
            .unwrap();
        assert_eq!(result.len(), 1, "Node 2 should have 1 stack");
        let stack = &result[0];
        assert_eq!(stack.stack_small_id, 3);
        assert_eq!(stack.owner_address, "owner3");
        assert_eq!(stack.stack_id, "stack3");
        assert_eq!(stack.task_small_id, 1);
        assert_eq!(stack.selected_node_id, 2);
        assert_eq!(stack.num_compute_units, 300);
        assert_eq!(stack.price_per_one_million_compute_units, 3000);
        assert_eq!(stack.already_computed_units, 100);
        assert!(!stack.in_settle_period);
        assert_eq!(stack.total_hash, vec![7, 8, 9]);
        assert_eq!(stack.num_total_messages, 3);

        // Test 8: Verify stacks with different states (in_settle_period)
        let result = state_manager
            .get_stacks_by_node_small_ids(&[1])
            .await
            .unwrap();
        assert!(
            result.iter().any(|s| s.in_settle_period),
            "Should find stack in settle period"
        );
        assert!(
            result.iter().any(|s| !s.in_settle_period),
            "Should find stack not in settle period"
        );

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_stack_by_id() {
        let state_manager = setup_test_db().await;

        // First create a task and subscription since they're required foreign keys
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();

        // Create node subscription
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();

        // Create test stacks
        let stack1 = Stack {
            stack_small_id: 1,
            owner_address: "owner1".to_string(),
            stack_id: "stack1".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 100,
            price_per_one_million_compute_units: 1000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![0; 32],
            num_total_messages: 0,
        };

        let stack2 = Stack {
            stack_small_id: 2,
            owner_address: "owner2".to_string(),
            stack_id: "stack2".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 200,
            price_per_one_million_compute_units: 2000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![0; 32],
            num_total_messages: 0,
        };

        // Insert test stacks
        state_manager
            .insert_new_stack(stack1.clone())
            .await
            .unwrap();
        state_manager
            .insert_new_stack(stack2.clone())
            .await
            .unwrap();

        // Test retrieving stacks for node_id 1
        let retrieved_stacks = state_manager.get_stack_by_id(1).await.unwrap();
        assert_eq!(retrieved_stacks.len(), 2);
        assert_eq!(retrieved_stacks[0], stack1);
        assert_eq!(retrieved_stacks[1], stack2);

        // Test retrieving stacks for non-existent node_id
        let empty_stacks = state_manager.get_stack_by_id(999).await.unwrap();
        assert!(empty_stacks.is_empty());

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_stack_by_id_with_multiple_nodes() {
        let state_manager = setup_test_db().await;

        // Create task
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();

        // Create node subscriptions for two different nodes
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();
        state_manager
            .subscribe_node_to_task(2, 1, 100, 1000)
            .await
            .unwrap();

        // Create stacks for different nodes
        let stack1 = Stack {
            stack_small_id: 1,
            owner_address: "owner1".to_string(),
            stack_id: "stack1".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 100,
            price_per_one_million_compute_units: 1000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![0; 32],
            num_total_messages: 0,
        };

        let stack2 = Stack {
            stack_small_id: 2,
            owner_address: "owner2".to_string(),
            stack_id: "stack2".to_string(),
            task_small_id: 1,
            selected_node_id: 2,
            num_compute_units: 200,
            price_per_one_million_compute_units: 2000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![0; 32],
            num_total_messages: 0,
        };

        state_manager
            .insert_new_stack(stack1.clone())
            .await
            .unwrap();
        state_manager
            .insert_new_stack(stack2.clone())
            .await
            .unwrap();

        // Test retrieving stacks for each node
        let node1_stacks = state_manager.get_stack_by_id(1).await.unwrap();
        assert_eq!(node1_stacks.len(), 1);
        assert_eq!(node1_stacks[0], stack1);

        let node2_stacks = state_manager.get_stack_by_id(2).await.unwrap();
        assert_eq!(node2_stacks.len(), 1);
        assert_eq!(node2_stacks[0], stack2);

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_almost_filled_stacks() {
        let state_manager = setup_test_db().await;

        // Setup: Create a task and node subscription first
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();

        // Subscribe nodes to the task
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();
        state_manager
            .subscribe_node_to_task(2, 1, 100, 1000)
            .await
            .unwrap();
        state_manager
            .subscribe_node_to_task(3, 1, 100, 1000)
            .await
            .unwrap();

        // Insert test stacks with various fill levels
        let test_stacks = vec![
            // Stack 90% filled for node 1
            Stack {
                stack_small_id: 1,
                owner_address: "owner1".to_string(),
                stack_id: "stack1".to_string(),
                task_small_id: 1,
                selected_node_id: 1,
                num_compute_units: 100,
                already_computed_units: 90,
                price_per_one_million_compute_units: 1000,
                in_settle_period: false,
                total_hash: vec![0; 32],
                num_total_messages: 0,
            },
            // Stack 50% filled for node 1
            Stack {
                stack_small_id: 2,
                owner_address: "owner2".to_string(),
                stack_id: "stack2".to_string(),
                task_small_id: 1,
                selected_node_id: 1,
                num_compute_units: 100,
                already_computed_units: 50,
                price_per_one_million_compute_units: 1000,
                in_settle_period: false,
                total_hash: vec![0; 32],
                num_total_messages: 0,
            },
            // Stack 95% filled for node 2
            Stack {
                stack_small_id: 3,
                owner_address: "owner3".to_string(),
                stack_id: "stack3".to_string(),
                task_small_id: 1,
                selected_node_id: 2,
                num_compute_units: 100,
                already_computed_units: 95,
                price_per_one_million_compute_units: 1000,
                in_settle_period: false,
                total_hash: vec![0; 32],
                num_total_messages: 0,
            },
            // Stack 100% filled for node 3
            Stack {
                stack_small_id: 4,
                owner_address: "owner4".to_string(),
                stack_id: "stack4".to_string(),
                task_small_id: 1,
                selected_node_id: 3,
                num_compute_units: 100,
                already_computed_units: 100,
                price_per_one_million_compute_units: 1000,
                in_settle_period: false,
                total_hash: vec![0; 32],
                num_total_messages: 0,
            },
        ];

        for stack in test_stacks {
            state_manager.insert_new_stack(stack).await.unwrap();
        }

        // Test case 1: Get stacks that are more than 80% filled
        let filled_stacks = state_manager
            .get_almost_filled_stacks(&[1, 2, 3], 0.8)
            .await
            .unwrap();
        assert_eq!(filled_stacks.len(), 3);
        assert!(filled_stacks
            .iter()
            .all(|s| { (s.already_computed_units as f64 / s.num_compute_units as f64) > 0.8 }));
        assert!(filled_stacks
            .iter()
            .all(|s| { [1, 2, 3].contains(&s.selected_node_id) }));

        // Test case 2: Get stacks that are more than 90% filled
        let very_filled_stacks = state_manager
            .get_almost_filled_stacks(&[1, 2, 3], 0.9)
            .await
            .unwrap();
        assert_eq!(very_filled_stacks.len(), 2);
        assert!(very_filled_stacks
            .iter()
            .all(|s| { (s.already_computed_units as f64 / s.num_compute_units as f64) > 0.9 }));
        assert!(very_filled_stacks
            .iter()
            .all(|s| { [2, 3].contains(&s.selected_node_id) }));

        // Test case 3: Check specific node
        let node1_stacks = state_manager
            .get_almost_filled_stacks(&[1], 0.8)
            .await
            .unwrap();
        assert_eq!(node1_stacks.len(), 1);
        assert_eq!(node1_stacks[0].selected_node_id, 1);

        // Test case 4: Check with threshold that matches no stacks
        let no_stacks = state_manager
            .get_almost_filled_stacks(&[1, 2, 3], 1.1)
            .await
            .unwrap();
        assert_eq!(no_stacks.len(), 0);

        // Test case 5: Check with empty node list
        let empty_nodes = state_manager
            .get_almost_filled_stacks(&[], 0.8)
            .await
            .unwrap();
        assert_eq!(empty_nodes.len(), 0);

        // Test case 6: Check multiple specific nodes
        let specific_nodes_stacks = state_manager
            .get_almost_filled_stacks(&[1, 2], 0.8)
            .await
            .unwrap();
        assert_eq!(specific_nodes_stacks.len(), 2);
        assert!(specific_nodes_stacks
            .iter()
            .all(|s| [1, 2].contains(&s.selected_node_id)));
        assert!(specific_nodes_stacks
            .iter()
            .all(|s| { (s.already_computed_units as f64 / s.num_compute_units as f64) > 0.8 }));

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_almost_filled_stacks_edge_cases() {
        let state_manager = setup_test_db().await;

        // Setup: Create a task and node subscription
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();

        // Test case 1: Stack with 0 compute units
        let zero_stack = Stack {
            stack_small_id: 1,
            owner_address: "owner1".to_string(),
            stack_id: "stack1".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 0,
            already_computed_units: 0,
            price_per_one_million_compute_units: 1000,
            in_settle_period: false,
            total_hash: vec![0; 32],
            num_total_messages: 0,
        };
        state_manager.insert_new_stack(zero_stack).await.unwrap();

        // Test case 2: Stack with very large numbers
        let large_stack = Stack {
            stack_small_id: 2,
            owner_address: "owner2".to_string(),
            stack_id: "stack2".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: i64::MAX,
            already_computed_units: i64::MAX / 2 + i64::MAX / 4,
            price_per_one_million_compute_units: 1000,
            in_settle_period: false,
            total_hash: vec![0; 32],
            num_total_messages: 0,
        };
        state_manager.insert_new_stack(large_stack).await.unwrap();

        // Test with various thresholds
        let test_cases = vec![
            (0.0, 2),  // Should return both stacks
            (0.5, 2),  // Should return both stacks
            (0.74, 2), // Should return both stacks
            (0.75, 1), // Should return the empty stack
            (0.99, 1), // Should return the empty stack
        ];

        for (threshold, expected_count) in test_cases {
            let stacks = state_manager
                .get_almost_filled_stacks(&[1], threshold)
                .await
                .unwrap();
            assert_eq!(
                stacks.len(),
                expected_count,
                "Failed for threshold {}: expected {} stacks, got {}",
                threshold,
                expected_count,
                stacks.len()
            );
        }

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_against_attestation_disputes() {
        let state_manager = setup_test_db().await;

        // Setup: Insert task, subscription, stack, and disputes
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
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
            price_per_one_million_compute_units: 1000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![],
            num_total_messages: 0,
        };
        state_manager.insert_new_stack(stack).await.unwrap();

        // Insert disputes
        let dispute1 = StackAttestationDispute {
            stack_small_id: 1,
            attestation_commitment: vec![1, 2, 3],
            attestation_node_id: 2,
            original_node_id: 1,
            original_commitment: vec![4, 5, 6],
        };
        let dispute2 = StackAttestationDispute {
            stack_small_id: 1,
            attestation_commitment: vec![7, 8, 9],
            attestation_node_id: 3,
            original_node_id: 1,
            original_commitment: vec![10, 11, 12],
        };
        state_manager
            .insert_stack_attestation_dispute(dispute1.clone())
            .await
            .unwrap();
        state_manager
            .insert_stack_attestation_dispute(dispute2.clone())
            .await
            .unwrap();

        // Test: Retrieve disputes against original_node_id = 1
        let node_ids = &[1];
        let disputes = state_manager
            .get_against_attestation_disputes(node_ids)
            .await
            .unwrap();
        assert_eq!(disputes.len(), 2);
        assert!(disputes.contains(&dispute1));
        assert!(disputes.contains(&dispute2));

        // Test: Retrieve disputes against a non-existent node
        let node_ids = &[999];
        let disputes = state_manager
            .get_against_attestation_disputes(node_ids)
            .await
            .unwrap();
        assert!(disputes.is_empty());

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_own_attestation_disputes() {
        let state_manager = setup_test_db().await;

        // Setup: Insert task, subscription, stack, and disputes
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
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
            price_per_one_million_compute_units: 1000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: vec![],
            num_total_messages: 0,
        };
        state_manager.insert_new_stack(stack).await.unwrap();

        // Insert disputes
        let dispute1 = StackAttestationDispute {
            stack_small_id: 1,
            attestation_commitment: vec![1, 2, 3],
            attestation_node_id: 2,
            original_node_id: 1,
            original_commitment: vec![4, 5, 6],
        };
        let dispute2 = StackAttestationDispute {
            stack_small_id: 1,
            attestation_commitment: vec![7, 8, 9],
            attestation_node_id: 3,
            original_node_id: 1,
            original_commitment: vec![10, 11, 12],
        };
        state_manager
            .insert_stack_attestation_dispute(dispute1.clone())
            .await
            .unwrap();
        state_manager
            .insert_stack_attestation_dispute(dispute2.clone())
            .await
            .unwrap();

        // Test: Retrieve disputes for attestation_node_id = 2
        let node_ids = &[2];
        let disputes = state_manager
            .get_own_attestation_disputes(node_ids)
            .await
            .unwrap();
        assert_eq!(disputes.len(), 1);
        assert_eq!(disputes[0], dispute1);

        // Test: Retrieve disputes for attestation_node_id = 3
        let node_ids = &[3];
        let disputes = state_manager
            .get_own_attestation_disputes(node_ids)
            .await
            .unwrap();
        assert_eq!(disputes.len(), 1);
        assert_eq!(disputes[0], dispute2);

        // Test: Retrieve disputes for multiple attestation_node_ids
        let node_ids = &[2, 3];
        let disputes = state_manager
            .get_own_attestation_disputes(node_ids)
            .await
            .unwrap();
        assert_eq!(disputes.len(), 2);
        assert!(disputes.contains(&dispute1));
        assert!(disputes.contains(&dispute2));

        // Test: Retrieve disputes for a non-existent attestation_node_id
        let node_ids = &[999];
        let disputes = state_manager
            .get_own_attestation_disputes(node_ids)
            .await
            .unwrap();
        assert!(disputes.is_empty());

        // Test: Retrieve disputes with an empty node list
        let node_ids: &[i64] = &[];
        let disputes = state_manager
            .get_own_attestation_disputes(node_ids)
            .await
            .unwrap();
        assert!(disputes.is_empty());

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_claimed_stacks() {
        let state_manager = setup_test_db().await;

        // Setup: Insert a task and subscribe nodes
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();
        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();
        state_manager
            .subscribe_node_to_task(2, 1, 100, 1000)
            .await
            .unwrap();

        // Insert stacks with different claimed statuses
        let stacks = vec![
            Stack {
                owner_address: "owner1".to_string(),
                stack_small_id: 1,
                stack_id: "stack1".to_string(),
                task_small_id: 1,
                selected_node_id: 1,
                num_compute_units: 100,
                price_per_one_million_compute_units: 1000,
                already_computed_units: 0,
                in_settle_period: false,
                total_hash: vec![0; 32],
                num_total_messages: 0,
            },
            Stack {
                owner_address: "owner2".to_string(),
                stack_small_id: 2,
                stack_id: "stack2".to_string(),
                task_small_id: 1,
                selected_node_id: 1,
                num_compute_units: 200,
                price_per_one_million_compute_units: 2000,
                already_computed_units: 50,
                in_settle_period: true,
                total_hash: vec![0; 32],
                num_total_messages: 0,
            },
            Stack {
                owner_address: "owner3".to_string(),
                stack_small_id: 3,
                stack_id: "stack3".to_string(),
                task_small_id: 1,
                selected_node_id: 2,
                num_compute_units: 300,
                price_per_one_million_compute_units: 3000,
                already_computed_units: 100,
                in_settle_period: false,
                total_hash: vec![0; 32],
                num_total_messages: 0,
            },
        ];

        for stack in stacks {
            state_manager.insert_new_stack(stack).await.unwrap();
        }

        state_manager
            .insert_new_stack_settlement_ticket(StackSettlementTicket {
                stack_small_id: 1,
                selected_node_id: 1,
                num_claimed_compute_units: 100,
                requested_attestation_nodes: String::new(),
                committed_stack_proofs: vec![],
                stack_merkle_leaves: vec![],
                dispute_settled_at_epoch: None,
                already_attested_nodes: String::new(),
                is_in_dispute: false,
                user_refund_amount: 0,
                is_claimed: true,
            })
            .await
            .unwrap();

        state_manager
            .insert_new_stack_settlement_ticket(StackSettlementTicket {
                stack_small_id: 2,
                selected_node_id: 1,
                num_claimed_compute_units: 100,
                requested_attestation_nodes: String::new(),
                committed_stack_proofs: vec![],
                stack_merkle_leaves: vec![],
                dispute_settled_at_epoch: None,
                already_attested_nodes: String::new(),
                is_in_dispute: false,
                user_refund_amount: 0,
                is_claimed: false,
            })
            .await
            .unwrap();

        state_manager
            .insert_new_stack_settlement_ticket(StackSettlementTicket {
                stack_small_id: 3,
                selected_node_id: 2,
                num_claimed_compute_units: 200,
                requested_attestation_nodes: String::new(),
                committed_stack_proofs: vec![],
                stack_merkle_leaves: vec![],
                dispute_settled_at_epoch: None,
                already_attested_nodes: String::new(),
                is_in_dispute: false,
                user_refund_amount: 0,
                is_claimed: true,
            })
            .await
            .unwrap();

        // Test 1: Get claimed stack settlement tickets for node 1
        let claimed_stacks_node1 = state_manager.get_claimed_stacks(&[1]).await.unwrap();
        assert_eq!(claimed_stacks_node1.len(), 1);
        assert_eq!(claimed_stacks_node1[0].stack_small_id, 1);

        // Test 2: Get claimed stack settlement tickets for node 2
        let claimed_stacks_node2 = state_manager.get_claimed_stacks(&[2]).await.unwrap();
        assert_eq!(claimed_stacks_node2.len(), 1);
        assert_eq!(claimed_stacks_node2[0].stack_small_id, 3);

        // Test 3: Get claimed stack settlement tickets for both nodes
        let claimed_stacks_both = state_manager.get_claimed_stacks(&[1, 2]).await.unwrap();
        assert_eq!(claimed_stacks_both.len(), 2);
        assert!(claimed_stacks_both.iter().any(|s| s.stack_small_id == 1));
        assert!(claimed_stacks_both.iter().any(|s| s.stack_small_id == 3));

        // Test 4: Get claimed stacks for non-existent node
        let claimed_stacks_none = state_manager.get_claimed_stacks(&[99]).await.unwrap();
        assert!(claimed_stacks_none.is_empty());

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_all_total_hashes() {
        let state_manager = setup_test_db().await;

        // Setup: Create a Task
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();

        state_manager
            .subscribe_node_to_task(1, 1, 100, 1000)
            .await
            .unwrap();

        // Test case 1: Empty input
        let result = state_manager.get_all_total_hashes(&[]).await.unwrap();
        assert!(result.is_empty(), "Empty input should return empty result");

        // Test case 2: Single stack
        let hash1 = vec![1u8; 32];
        let stack1 = Stack {
            owner_address: "owner1".to_string(),
            stack_small_id: 1,
            stack_id: "stack1".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 100,
            price_per_one_million_compute_units: 1000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: hash1.clone(),
            num_total_messages: 0,
        };
        state_manager.insert_new_stack(stack1).await.unwrap();

        let result = state_manager.get_all_total_hashes(&[1]).await.unwrap();
        assert_eq!(result.len(), 1, "Should return single hash");
        assert_eq!(result[0], hash1, "Hash should match inserted value");

        // Test case 3: Multiple stacks with different hash sizes
        let hash2 = vec![2u8; 64]; // Double size hash
        let hash3 = vec![3u8; 32]; // 32 bytes hash

        let stack2 = Stack {
            owner_address: "owner2".to_string(),
            stack_small_id: 2,
            stack_id: "stack2".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 200,
            price_per_one_million_compute_units: 2000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: hash2.clone(),
            num_total_messages: 0,
        };

        let stack3 = Stack {
            owner_address: "owner3".to_string(),
            stack_small_id: 3,
            stack_id: "stack3".to_string(),
            task_small_id: 1,
            selected_node_id: 1,
            num_compute_units: 300,
            price_per_one_million_compute_units: 3000,
            already_computed_units: 0,
            in_settle_period: false,
            total_hash: hash3.clone(),
            num_total_messages: 0,
        };

        state_manager.insert_new_stack(stack2).await.unwrap();
        state_manager.insert_new_stack(stack3).await.unwrap();

        let result = state_manager
            .get_all_total_hashes(&[1, 2, 3])
            .await
            .unwrap();
        assert_eq!(result.len(), 3, "Should return three hashes");
        assert_eq!(result[0], hash1, "First hash should match");
        assert_eq!(result[1], hash2, "Second hash should match");
        assert_eq!(result[2], hash3, "Third hash should match");

        // Test case 4: Non-existent stacks
        let result = state_manager
            .get_all_total_hashes(&[999, 1000])
            .await
            .unwrap();
        assert!(
            result.is_empty(),
            "Non-existent stacks should return empty result"
        );

        // Test case 5: Mix of existing and non-existing stacks
        let result = state_manager
            .get_all_total_hashes(&[1, 999, 2])
            .await
            .unwrap();
        assert_eq!(
            result.len(),
            2,
            "Should return only existing stacks' hashes"
        );
        assert_eq!(result[0], hash1, "First hash should match");
        assert_eq!(result[1], hash2, "Second hash should match");

        // Test case 6: Duplicate stack IDs
        let result = state_manager
            .get_all_total_hashes(&[1, 1, 1])
            .await
            .unwrap();
        assert_eq!(result.len(), 1, "Duplicate IDs should return single result");
        assert_eq!(result[0], hash1, "Hash should match for duplicate IDs");

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_stack_settlement_tickets() {
        let state_manager = setup_test_db().await;

        // Setup: Create a task and subscribe nodes
        let task = Task {
            task_small_id: 1,
            task_id: "task1".to_string(),
            role: 1,
            model_name: Some("model1".to_string()),
            is_deprecated: false,
            valid_until_epoch: Some(100),
            deprecated_at_epoch: None,
            security_level: 1,
            minimum_reputation_score: Some(50),
        };
        state_manager.insert_new_task(task).await.unwrap();

        // Subscribe multiple nodes
        for node_id in 1..=3 {
            state_manager
                .subscribe_node_to_task(node_id, 1, 100, 1000)
                .await
                .unwrap();
        }

        // Create and insert multiple stacks
        let stacks = vec![
            Stack {
                owner_address: "owner1".to_string(),
                stack_small_id: 1,
                stack_id: "stack1".to_string(),
                task_small_id: 1,
                selected_node_id: 1,
                num_compute_units: 100,
                price_per_one_million_compute_units: 1000,
                already_computed_units: 50,
                in_settle_period: true,
                total_hash: vec![],
                num_total_messages: 0,
            },
            Stack {
                owner_address: "owner2".to_string(),
                stack_small_id: 2,
                stack_id: "stack2".to_string(),
                task_small_id: 1,
                selected_node_id: 2,
                num_compute_units: 200,
                price_per_one_million_compute_units: 2000,
                already_computed_units: 150,
                in_settle_period: true,
                total_hash: vec![],
                num_total_messages: 0,
            },
            Stack {
                owner_address: "owner3".to_string(),
                stack_small_id: 3,
                stack_id: "stack3".to_string(),
                task_small_id: 1,
                selected_node_id: 3,
                num_compute_units: 300,
                price_per_one_million_compute_units: 3000,
                already_computed_units: 250,
                in_settle_period: true,
                total_hash: vec![],
                num_total_messages: 0,
            },
        ];

        for stack in stacks {
            state_manager.insert_new_stack(stack).await.unwrap();
        }

        // Create settlement tickets with different characteristics
        let tickets = vec![
            StackSettlementTicket {
                stack_small_id: 1,
                selected_node_id: 1,
                num_claimed_compute_units: 50,
                requested_attestation_nodes: "[2,3]".to_string(),
                committed_stack_proofs: vec![1; 32],
                stack_merkle_leaves: vec![2; 32],
                dispute_settled_at_epoch: None,
                already_attested_nodes: "[2]".to_string(),
                is_in_dispute: false,
                user_refund_amount: 0,
                is_claimed: false,
            },
            StackSettlementTicket {
                stack_small_id: 2,
                selected_node_id: 2,
                num_claimed_compute_units: 150,
                requested_attestation_nodes: "[1,3]".to_string(),
                committed_stack_proofs: vec![3; 32],
                stack_merkle_leaves: vec![4; 32],
                dispute_settled_at_epoch: Some(100),
                already_attested_nodes: "[1,3]".to_string(),
                is_in_dispute: true,
                user_refund_amount: 1000,
                is_claimed: true,
            },
            StackSettlementTicket {
                stack_small_id: 3,
                selected_node_id: 3,
                num_claimed_compute_units: 250,
                requested_attestation_nodes: "[1,2]".to_string(),
                committed_stack_proofs: vec![5; 32],
                stack_merkle_leaves: vec![6; 32],
                dispute_settled_at_epoch: None,
                already_attested_nodes: "[]".to_string(),
                is_in_dispute: false,
                user_refund_amount: 0,
                is_claimed: false,
            },
        ];

        for ticket in tickets.clone() {
            state_manager
                .insert_new_stack_settlement_ticket(ticket)
                .await
                .unwrap();
        }

        // Test 1: Empty input
        let empty_result = state_manager
            .get_stack_settlement_tickets(&[])
            .await
            .unwrap();
        assert!(
            empty_result.is_empty(),
            "Empty input should return empty result"
        );

        // Test 2: Single ticket retrieval
        let single_result = state_manager
            .get_stack_settlement_tickets(&[1])
            .await
            .unwrap();
        assert_eq!(single_result.len(), 1, "Should return exactly one ticket");
        assert_eq!(
            single_result[0], tickets[0],
            "Retrieved ticket should match original"
        );

        // Test 3: Multiple tickets retrieval
        let multiple_result = state_manager
            .get_stack_settlement_tickets(&[1, 2])
            .await
            .unwrap();
        assert_eq!(
            multiple_result.len(),
            2,
            "Should return exactly two tickets"
        );
        assert!(
            multiple_result.contains(&tickets[0]) && multiple_result.contains(&tickets[1]),
            "Should contain both requested tickets"
        );

        // Test 4: Non-existent ticket IDs
        let non_existent = state_manager
            .get_stack_settlement_tickets(&[999])
            .await
            .unwrap();
        assert!(
            non_existent.is_empty(),
            "Non-existent ID should return empty result"
        );

        // Test 5: Mixed existing and non-existing IDs
        let mixed_result = state_manager
            .get_stack_settlement_tickets(&[1, 999, 2])
            .await
            .unwrap();
        assert_eq!(mixed_result.len(), 2, "Should return only existing tickets");
        assert!(
            mixed_result.contains(&tickets[0]) && mixed_result.contains(&tickets[1]),
            "Should contain only existing tickets"
        );

        // Test 6: Duplicate IDs in input
        let duplicate_result = state_manager
            .get_stack_settlement_tickets(&[1, 1, 1])
            .await
            .unwrap();
        assert_eq!(
            duplicate_result.len(),
            1,
            "Should return unique tickets only"
        );
        assert_eq!(
            duplicate_result[0], tickets[0],
            "Should return correct ticket"
        );

        // Test 7: All tickets retrieval
        let all_result = state_manager
            .get_stack_settlement_tickets(&[1, 2, 3])
            .await
            .unwrap();
        assert_eq!(all_result.len(), 3, "Should return all tickets");
        assert!(
            all_result.contains(&tickets[0])
                && all_result.contains(&tickets[1])
                && all_result.contains(&tickets[2]),
            "Should contain all tickets"
        );

        truncate_tables(&state_manager.db).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_insert_node_public_key_rotation() -> Result<()> {
        let state_manager = setup_test_db().await;

        // Test data
        let epoch = 123u64;
        let key_rotation_counter = 456u64;
        let node_small_id = 789u64;
        let public_key_bytes = vec![1, 2, 3, 4];
        let tee_remote_attestation_bytes = vec![5, 6, 7, 8];
        let tee_certificate_chain_bytes = vec![9, 10, 11, 12];
        // Insert initial rotation
        state_manager
            .insert_node_public_key_rotation(
                epoch,
                key_rotation_counter,
                node_small_id,
                public_key_bytes.clone(),
                tee_remote_attestation_bytes.clone(),
                0,
            )
            .await?;

        // Verify the insertion
        let row = sqlx::query("SELECT * FROM node_public_key_rotations WHERE node_small_id = $1")
            .bind(node_small_id as i64)
            .fetch_one(&state_manager.db)
            .await?;

        assert_eq!(row.get::<i64, _>("epoch"), epoch as i64);
        assert_eq!(
            row.get::<i64, _>("key_rotation_counter"),
            key_rotation_counter as i64
        );
        assert_eq!(row.get::<i64, _>("node_small_id"), node_small_id as i64);
        assert_eq!(row.get::<Vec<u8>, _>("public_key_bytes"), public_key_bytes);
        assert_eq!(
            row.get::<Vec<u8>, _>("remote_attestation_bytes"),
            tee_remote_attestation_bytes
        );
        assert_eq!(row.get::<i32, _>("device_type"), 0);
        assert_eq!(
            row.get::<Vec<u8>, _>("certificate_chain_bytes"),
            tee_certificate_chain_bytes
        );

        // Test update with new values
        let new_epoch = 999u64;
        let new_key_rotation_counter = 888u64;
        let new_public_key_bytes = vec![9, 10, 11, 12];
        let new_tee_remote_attestation_bytes = vec![13, 14, 15, 16];
        state_manager
            .insert_node_public_key_rotation(
                new_epoch,
                new_key_rotation_counter,
                node_small_id,
                new_public_key_bytes.clone(),
                new_tee_remote_attestation_bytes.clone(),
                0,
            )
            .await?;

        // Verify the update
        let updated_row =
            sqlx::query("SELECT * FROM node_public_key_rotations WHERE node_small_id = $1")
                .bind(node_small_id as i64)
                .fetch_one(&state_manager.db)
                .await?;

        assert_eq!(updated_row.get::<i64, _>("epoch"), new_epoch as i64);
        assert_eq!(
            updated_row.get::<i64, _>("key_rotation_counter"),
            new_key_rotation_counter as i64
        );
        assert_eq!(
            updated_row.get::<i64, _>("node_small_id"),
            node_small_id as i64
        );
        assert_eq!(
            updated_row.get::<Vec<u8>, _>("public_key_bytes"),
            new_public_key_bytes
        );
        assert_eq!(
            updated_row.get::<Vec<u8>, _>("evidence_data_bytes"),
            new_tee_remote_attestation_bytes
        );

        // Verify only one row exists
        let count = sqlx::query(
            "SELECT COUNT(*) as count FROM node_public_key_rotations WHERE node_small_id = $1",
        )
        .bind(node_small_id as i64)
        .fetch_one(&state_manager.db)
        .await?;

        assert_eq!(count.get::<i64, _>("count"), 1);

        truncate_tables(&state_manager.db).await;
        Ok(())
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_insert_node_public_key_rotation_multiple_nodes() -> Result<()> {
        let state_manager = setup_test_db().await;
        truncate_tables(&state_manager.db).await;

        // Test data for multiple nodes
        let nodes = [
            (1u64, vec![1, 2, 3], vec![4, 5, 6]),
            (2u64, vec![7, 8, 9], vec![10, 11, 12]),
            (3u64, vec![13, 14, 15], vec![16, 17, 18]),
        ];

        // Insert rotations for multiple nodes
        for (node_id, pub_key, tee_bytes) in &nodes {
            state_manager
                .insert_node_public_key_rotation(
                    100u64,
                    1u64,
                    *node_id,
                    pub_key.clone(),
                    tee_bytes.clone(),
                    u16::try_from(*node_id).unwrap(),
                )
                .await?;
        }

        // Verify all insertions
        for (node_id, pub_key, tee_bytes) in &nodes {
            let row =
                sqlx::query("SELECT * FROM node_public_key_rotations WHERE node_small_id = $1")
                    .bind(*node_id as i64)
                    .fetch_one(&state_manager.db)
                    .await?;

            assert_eq!(row.get::<i64, _>("node_small_id"), *node_id as i64);
            assert_eq!(row.get::<Vec<u8>, _>("public_key_bytes"), *pub_key);
            assert_eq!(row.get::<Vec<u8>, _>("evidence_data_bytes"), *tee_bytes);
            assert_eq!(
                row.get::<i32, _>("device_type"),
                i32::try_from(*node_id).unwrap()
            );
        }

        // Verify total count
        let count = sqlx::query("SELECT COUNT(*) as count FROM node_public_key_rotations")
            .fetch_one(&state_manager.db)
            .await?;

        assert_eq!(count.get::<i64, _>("count"), 3);

        truncate_tables(&state_manager.db).await;
        Ok(())
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_insert_node_public_key_rotation_empty_bytes() -> Result<()> {
        let state_manager = setup_test_db().await;

        // Test with empty byte vectors
        let node_small_id = 1u64;
        let empty_bytes = vec![];

        state_manager
            .insert_node_public_key_rotation(
                100u64,
                1u64,
                node_small_id,
                empty_bytes.clone(),
                empty_bytes.clone(),
                0,
            )
            .await?;

        let row = sqlx::query("SELECT * FROM node_public_key_rotations WHERE node_small_id = $1")
            .bind(node_small_id as i64)
            .fetch_one(&state_manager.db)
            .await?;

        assert_eq!(row.get::<Vec<u8>, _>("public_key_bytes"), empty_bytes);
        assert_eq!(row.get::<Vec<u8>, _>("evidence_data_bytes"), empty_bytes);

        truncate_tables(&state_manager.db).await;
        Ok(())
    }
}
