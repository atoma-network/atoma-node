use atoma_state::{SqlitePool, StateManager};
use serde_json::Value;
use std::time::Duration;
use tracing::{error, instrument, trace};

use crate::{
    events::{
        AtomaEvent, NewStackSettlementAttestationEvent, NodeSubscribedToTaskEvent,
        NodeSubscriptionUpdatedEvent, NodeUnsubscribedFromTaskEvent, StackAttestationDisputeEvent,
        StackCreatedEvent, StackSettlementTicketClaimedEvent, StackSettlementTicketEvent,
        StackTrySettleEvent, TaskDeprecationEvent, TaskRegisteredEvent,
    },
    subscriber::Result,
};

/// The duration for retries in milliseconds.
const DURATION_FOR_RETRY_IN_MILLIS: u64 = 100;
/// The maximum number of retries for events to which handling fails.
const MAX_RETRIES_FOR_UNHANDLED_EVENTS: usize = 3;

/// Handles various Atoma events by delegating to specific handler functions based on the event type.
///
/// This function serves as the main event dispatcher for the Atoma system, routing different event types
/// to their corresponding handler functions. For each event type, it either calls the appropriate handler
/// or returns an unimplemented error for events that are not yet supported.
///
/// # Arguments
///
/// * `event` - A reference to the `AtomaEvent` enum indicating the type of event to handle
/// * `value` - The serialized event data as a `serde_json::Value`
/// * `db` - A reference to the SQLite connection pool for database operations
/// * `node_small_ids` - A slice of node IDs that are relevant for the current context
///
/// # Returns
///
/// Returns a `Result<()>` which is:
/// * `Ok(())` if the event was handled successfully
/// * `Err(_)` if there was an error processing the event or if the event type is not yet implemented
///
/// # Event Types
///
/// Currently implemented events:
/// * `NodeSubscribedToTaskEvent` - Handles node task subscription events
/// * `NodeSubscriptionUpdatedEvent` - Handles updates to node task subscriptions
/// * `NodeUnsubscribedFromTaskEvent` - Handles node task unsubscription events
/// * `TaskRegisteredEvent` - Handles new task registration
/// * `TaskDeprecationEvent` - Handles task deprecation
/// * `StackCreatedEvent` - Handles stack creation
/// * `StackTrySettleEvent` - Handles stack settlement attempts
/// * `NewStackSettlementAttestationEvent` - Handles new stack settlement attestations
/// * `StackSettlementTicketEvent` - Handles stack settlement tickets
/// * `StackSettlementTicketClaimedEvent` - Handles claimed stack settlement tickets
/// * `StackAttestationDisputeEvent` - Handles stack attestation disputes
///
/// Unimplemented events will return an `unimplemented!()` error with a descriptive message.
///
/// # Examples
///
/// ```ignore
/// use atoma_state::SqlitePool;
/// use serde_json::Value;
///
/// async fn example(event: AtomaEvent, value: Value, db: &SqlitePool, node_ids: &[u64]) {
///     match handle_atoma_event(&event, value, db, node_ids).await {
///         Ok(()) => println!("Event handled successfully"),
///         Err(e) => eprintln!("Error handling event: {}", e),
///     }
/// }
/// ```
#[instrument(level = "trace", skip_all)]
pub(crate) async fn handle_atoma_event(
    event: &AtomaEvent,
    value: Value,
    db: &SqlitePool,
    node_small_ids: &[u64],
) -> Result<()> {
    match event {
        AtomaEvent::DisputeEvent => {
            unimplemented!("Dispute event not implemented");
        }
        AtomaEvent::SettledEvent => {
            unimplemented!("Settled event not implemented");
        }
        AtomaEvent::PublishedEvent => {
            unimplemented!("Published event not implemented");
        }
        AtomaEvent::NewlySampledNodesEvent => {
            unimplemented!("Newly sampled nodes event not implemented");
        }
        AtomaEvent::NodeRegisteredEvent => {
            unimplemented!("Node registered event not implemented");
        }
        AtomaEvent::NodeSubscribedToModelEvent => {
            unimplemented!("Node subscribed to model event not implemented");
        }
        AtomaEvent::NodeSubscribedToTaskEvent => {
            handle_node_task_subscription_event(value, db).await
        }
        AtomaEvent::NodeSubscriptionUpdatedEvent => {
            handle_node_task_subscription_updated_event(value, db).await
        }
        AtomaEvent::NodeUnsubscribedFromTaskEvent => {
            handle_node_task_unsubscription_event(value, db).await
        }
        AtomaEvent::TaskRegisteredEvent => handle_new_task_event(value, db).await,
        AtomaEvent::TaskDeprecationEvent => handle_task_deprecation_event(value, db).await,
        AtomaEvent::FirstSubmissionEvent => {
            todo!()
        }
        AtomaEvent::StackCreatedEvent => {
            handle_stack_created_event(value, db, node_small_ids).await
        }
        AtomaEvent::StackTrySettleEvent => handle_stack_try_settle_event(value, db).await,
        AtomaEvent::NewStackSettlementAttestationEvent => {
            handle_new_stack_settlement_attestation_event(value, db).await
        }
        AtomaEvent::StackSettlementTicketEvent => {
            handle_stack_settlement_ticket_event(value, db).await
        }
        AtomaEvent::StackSettlementTicketClaimedEvent => {
            handle_stack_settlement_ticket_claimed_event(value, db).await
        }
        AtomaEvent::StackAttestationDisputeEvent => {
            handle_stack_attestation_dispute_event(value, db).await
        }
        AtomaEvent::TaskRemovedEvent => {
            unimplemented!("Task removed event not implemented");
        }
        AtomaEvent::RetrySettlementEvent => {
            unimplemented!("Retry settlement event not implemented");
        }
        AtomaEvent::Text2ImagePromptEvent => {
            unimplemented!("Text2Image prompt event not implemented");
        }
        AtomaEvent::Text2TextPromptEvent => {
            unimplemented!("Text2Text prompt event not implemented");
        }
    }
}

/// Handles a new task event by processing and inserting it into the database.
///
/// # Arguments
///
/// * `value` - A `serde_json::Value` containing the serialized `TaskRegisteredEvent`.
/// * `db` - A reference to the `SqlitePool` for database operations.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the task was successfully processed and inserted, or an error otherwise.
///
/// # Errors
///
/// This function will return an error if:
/// * The `value` cannot be deserialized into a `TaskRegisteredEvent`.
/// * The `StateManager` fails to insert the new task into the database.
///
/// # Instrumentation
///
/// This function is instrumented with tracing at the trace level, skipping all arguments
/// but including an `event_name` field for logging purposes.
#[instrument(level = "trace", skip_all)]
pub(crate) async fn handle_new_task_event(value: Value, db: &SqlitePool) -> Result<()> {
    trace!("Processing new task event");
    let task_event: TaskRegisteredEvent = serde_json::from_value(value)?;
    let state_manager = StateManager::new(db.clone());
    let task = task_event.into();
    state_manager.insert_new_task(task).await?;
    Ok(())
}

/// Handles a task deprecation event.
///
/// This function processes a task deprecation event by parsing the event data,
/// extracting the necessary information, and updating the task's status in the database.
///
/// # Arguments
///
/// * `value` - A `serde_json::Value` containing the serialized task deprecation event data.
/// * `db` - A reference to the `SqlitePool` for database operations.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the event was processed successfully, or an error if something went wrong.
///
/// # Errors
///
/// This function will return an error if:
/// * The event data cannot be deserialized into a `TaskDeprecationEvent`.
/// * The database operation to deprecate the task fails.
#[instrument(level = "trace", skip_all)]
pub(crate) async fn handle_task_deprecation_event(value: Value, db: &SqlitePool) -> Result<()> {
    trace!("Processing task deprecation event");
    let task_event: TaskDeprecationEvent = serde_json::from_value(value)?;
    let task_small_id = task_event.task_small_id;
    let epoch = task_event.epoch;
    let state_manager = StateManager::new(db.clone());
    state_manager
        .deprecate_task(task_small_id.inner as i64, epoch as i64)
        .await?;
    Ok(())
}

/// Handles a node task subscription event.
///
/// This function processes a node task subscription event by parsing the event data,
/// extracting the necessary information, and updating the node's subscription to the task in the database.
///
/// # Arguments
///
/// * `value` - A `serde_json::Value` containing the serialized node task subscription event data.
/// * `db` - A reference to the `SqlitePool` for database operations.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the event was processed successfully, or an error if something went wrong.
///
/// # Errors
///
/// This function will return an error if:
/// * The event data cannot be deserialized into a `NodeSubscribedToTaskEvent`.
/// * The database operation to subscribe the node to the task fails.
#[instrument(level = "trace", skip_all)]
pub(crate) async fn handle_node_task_subscription_event(
    value: Value,
    db: &SqlitePool,
) -> Result<()> {
    trace!("Processing node subscription event");
    let node_subscription_event: NodeSubscribedToTaskEvent = serde_json::from_value(value)?;
    let node_small_id = node_subscription_event.node_small_id.inner as i64;
    let task_small_id = node_subscription_event.task_small_id.inner as i64;
    let price_per_compute_unit = node_subscription_event.price_per_compute_unit as i64;
    let max_num_compute_units = node_subscription_event.max_num_compute_units as i64;
    let state_manager = StateManager::new(db.clone());
    state_manager
        .subscribe_node_to_task(
            node_small_id,
            task_small_id,
            price_per_compute_unit,
            max_num_compute_units,
        )
        .await?;
    Ok(())
}

/// Handles a node task subscription updated event.
///
/// This function processes a node task subscription updated event by parsing the event data,
/// extracting the necessary information, and updating the node's subscription to the task in the database.
///
/// # Arguments
///
/// * `value` - A `serde_json::Value` containing the serialized node task subscription updated event data.
/// * `db` - A reference to the `SqlitePool` for database operations.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the event was processed successfully, or an error if something went wrong.
///
/// # Errors
///
/// This function will return an error if:
/// * The event data cannot be deserialized into a `NodeSubscriptionUpdatedEvent`.
/// * The database operation to update the node's subscription to the task fails.
#[instrument(level = "trace", skip_all)]
pub(crate) async fn handle_node_task_subscription_updated_event(
    value: Value,
    db: &SqlitePool,
) -> Result<()> {
    trace!("Processing node subscription updated event");
    let node_subscription_event: NodeSubscriptionUpdatedEvent = serde_json::from_value(value)?;
    let node_small_id = node_subscription_event.node_small_id.inner as i64;
    let task_small_id = node_subscription_event.task_small_id.inner as i64;
    let price_per_compute_unit = node_subscription_event.price_per_compute_unit as i64;
    let max_num_compute_units = node_subscription_event.max_num_compute_units as i64;
    let state_manager = StateManager::new(db.clone());
    state_manager
        .update_node_subscription(
            node_small_id,
            task_small_id,
            price_per_compute_unit,
            max_num_compute_units,
        )
        .await?;
    Ok(())
}

/// Handles a node task unsubscription event.
///
/// This function processes a node task unsubscription event by parsing the event data,
/// extracting the necessary information, and updating the node's subscription status in the database.
///
/// # Arguments
///
/// * `value` - A `serde_json::Value` containing the serialized node task unsubscription event data.
/// * `db` - A reference to the `SqlitePool` for database operations.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the event was processed successfully, or an error if something went wrong.
///
/// # Errors
///
/// This function will return an error if:
/// * The event data cannot be deserialized into a `NodeUnsubscribedFromTaskEvent`.
/// * The database operation to unsubscribe the node from the task fails.
#[instrument(level = "trace", skip_all)]
pub(crate) async fn handle_node_task_unsubscription_event(
    value: Value,
    db: &SqlitePool,
) -> Result<()> {
    trace!("Processing node unsubscription event");
    let node_subscription_event: NodeUnsubscribedFromTaskEvent = serde_json::from_value(value)?;
    let node_small_id = node_subscription_event.node_small_id.inner as i64;
    let task_small_id = node_subscription_event.task_small_id.inner as i64;
    let state_manager = StateManager::new(db.clone());
    state_manager
        .unsubscribe_node_from_task(node_small_id, task_small_id)
        .await?;
    Ok(())
}

/// Handles a stack created event.
///
/// This function processes a stack created event by parsing the event data,
/// checking if the selected node is one of the current nodes, and if so,
/// inserting the new stack into the database.
///
/// # Arguments
///
/// * `value` - A `serde_json::Value` containing the serialized stack created event data.
/// * `db` - A reference to the `SqlitePool` for database operations.
/// * `node_small_ids` - A slice of `u64` values representing the small IDs of the current nodes.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the event was processed successfully, or an error if something went wrong.
///
/// # Errors
///
/// This function will return an error if:
/// * The event data cannot be deserialized into a `StackCreatedEvent`.
/// * The database operation to insert the new stack fails.
///
/// # Behavior
///
/// The function only inserts the new stack into the database if the selected node's ID
/// is present in the `node_small_ids` slice. This ensures that only stacks relevant to
/// the current node(s) are processed and stored.
#[instrument(level = "trace", skip_all)]
pub(crate) async fn handle_stack_created_event(
    value: Value,
    db: &SqlitePool,
    node_small_ids: &[u64],
) -> Result<()> {
    trace!("Processing stack created event");
    let stack_event: StackCreatedEvent = serde_json::from_value(value)?;
    let node_small_id = stack_event.selected_node_id.inner;
    if node_small_ids.contains(&node_small_id) {
        trace!("Stack selected current node, with id {node_small_id}, inserting new stack");
        let state_manager = StateManager::new(db.clone());
        let stack = stack_event.into();
        state_manager.insert_new_stack(stack).await?;
    }
    Ok(())
}

/// Handles a stack try settle event.
///
/// This function processes a stack try settle event by parsing the event data,
/// converting it into a stack settlement ticket, and inserting it into the database.
///
/// # Arguments
///
/// * `value` - A `serde_json::Value` containing the serialized stack try settle event data.
/// * `db` - A reference to the `SqlitePool` for database operations.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the event was processed successfully, or an error if something went wrong.
///
/// # Errors
///
/// This function will return an error if:
/// * The event data cannot be deserialized into a `StackTrySettleEvent`.
/// * The database operation to insert the new stack settlement ticket fails.
///
/// # Behavior
///
/// The function performs the following steps:
/// 1. Deserializes the input `value` into a `StackTrySettleEvent`.
/// 2. Converts the event into a stack settlement ticket.
/// 3. Creates a new `StateManager` instance.
/// 4. Inserts the new stack settlement ticket into the database using the `StateManager`.
#[instrument(level = "trace", skip_all)]
pub(crate) async fn handle_stack_try_settle_event(value: Value, db: &SqlitePool) -> Result<()> {
    trace!("Processing stack try settle event");
    let stack_settlement_ticket_event: StackTrySettleEvent = serde_json::from_value(value)?;
    let stack_settlement_ticket = stack_settlement_ticket_event.into();
    let state_manager = StateManager::new(db.clone());
    state_manager
        .insert_new_stack_settlement_ticket(stack_settlement_ticket)
        .await?;
    Ok(())
}

/// Handles a new stack settlement attestation event.
///
/// This function processes a new stack settlement attestation event by parsing the event data
/// and updating the corresponding stack settlement ticket in the database with attestation commitments.
///
/// # Arguments
///
/// * `value` - A `serde_json::Value` containing the serialized new stack settlement attestation event data.
/// * `db` - A reference to the `SqlitePool` for database operations.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the event was processed successfully, or an error if something went wrong.
///
/// # Errors
///
/// This function will return an error if:
/// * The event data cannot be deserialized into a `NewStackSettlementAttestationEvent`.
/// * The database operation to update the stack settlement ticket with attestation commitments fails.
#[instrument(level = "trace", skip_all)]
pub(crate) async fn handle_new_stack_settlement_attestation_event(
    value: Value,
    db: &SqlitePool,
) -> Result<()> {
    trace!("Processing new stack settlement attestation event");
    let stack_attestation_event: NewStackSettlementAttestationEvent =
        serde_json::from_value(value)?;
    let stack_small_id = stack_attestation_event.stack_small_id.inner as i64;
    let attestation_node_id = stack_attestation_event.attestation_node_id.inner as i64;
    let committed_stack_proof = stack_attestation_event.committed_stack_proof;
    let stack_merkle_leaf = stack_attestation_event.stack_merkle_leaf;

    let state_manager = StateManager::new(db.clone());
    state_manager
        .update_stack_settlement_ticket_with_attestation_commitments(
            stack_small_id,
            committed_stack_proof,
            stack_merkle_leaf,
            attestation_node_id,
        )
        .await?;
    Ok(())
}

/// Handles a stack settlement ticket event.
///
/// This function processes a stack settlement ticket event by parsing the event data
/// and updating the corresponding stack settlement ticket in the database.
///
/// # Arguments
///
/// * `value` - A `serde_json::Value` containing the serialized stack settlement ticket event data.
/// * `db` - A reference to the `SqlitePool` for database operations.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the event was processed successfully, or an error if something went wrong.
///
/// # Errors
///
/// This function will return an error if:
/// * The event data cannot be deserialized into a `StackSettlementTicketEvent`.
/// * The database operation to settle the stack settlement ticket fails.
///
/// # Behavior
///
/// The function performs the following steps:
/// 1. Deserializes the input `value` into a `StackSettlementTicketEvent`.
/// 2. Extracts the `stack_small_id` and `dispute_settled_at_epoch` from the event.
/// 3. Creates a new `StateManager` instance.
/// 4. Calls the `settle_stack_settlement_ticket` method on the `StateManager` to update the database.
#[instrument(level = "trace", skip_all)]
pub(crate) async fn handle_stack_settlement_ticket_event(
    value: Value,
    db: &SqlitePool,
) -> Result<()> {
    trace!("Processing stack settlement ticket event");
    let stack_settlement_ticket_event: StackSettlementTicketEvent = serde_json::from_value(value)?;
    let stack_small_id = stack_settlement_ticket_event.stack_small_id.inner as i64;
    let dispute_settled_at_epoch = stack_settlement_ticket_event.dispute_settled_at_epoch as i64;
    let state_manager = StateManager::new(db.clone());
    state_manager
        .settle_stack_settlement_ticket(stack_small_id, dispute_settled_at_epoch)
        .await?;
    Ok(())
}

/// Handles a stack settlement ticket claimed event.
///
/// This function processes a stack settlement ticket claimed event by parsing the event data
/// and updating the corresponding stack settlement ticket in the database with claim information.
///
/// # Arguments
///
/// * `value` - A `serde_json::Value` containing the serialized stack settlement ticket claimed event data.
/// * `db` - A reference to the `SqlitePool` for database operations.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the event was processed successfully, or an error if something went wrong.
///
/// # Errors
///
/// This function will return an error if:
/// * The event data cannot be deserialized into a `StackSettlementTicketClaimedEvent`.
/// * The database operation to update the stack settlement ticket with claim information fails.
///
/// # Behavior
///
/// The function performs the following steps:
/// 1. Deserializes the input `value` into a `StackSettlementTicketClaimedEvent`.
/// 2. Extracts the `stack_small_id` and `user_refund_amount` from the event.
/// 3. Creates a new `StateManager` instance.
/// 4. Calls the `update_stack_settlement_ticket_with_claim` method on the `StateManager` to update the database.
#[instrument(level = "trace", skip_all)]
pub(crate) async fn handle_stack_settlement_ticket_claimed_event(
    value: Value,
    db: &SqlitePool,
) -> Result<()> {
    trace!("Processing stack settlement ticket claimed event");
    let stack_settlement_ticket_event: StackSettlementTicketClaimedEvent =
        serde_json::from_value(value)?;
    let stack_small_id = stack_settlement_ticket_event.stack_small_id.inner as i64;
    let user_refund_amount = stack_settlement_ticket_event.user_refund_amount as i64;
    let state_manager = StateManager::new(db.clone());
    state_manager
        .update_stack_settlement_ticket_with_claim(stack_small_id, user_refund_amount)
        .await?;
    Ok(())
}

/// Handles a stack attestation dispute event.
///
/// This function processes a stack attestation dispute event by parsing the event data
/// and inserting the dispute information into the database.
///
/// # Arguments
///
/// * `value` - A `serde_json::Value` containing the serialized stack attestation dispute event data.
/// * `db` - A reference to the `SqlitePool` for database operations.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the event was processed successfully, or an error if something went wrong.
///
/// # Errors
///
/// This function will return an error if:
/// * The event data cannot be deserialized into a `StackAttestationDisputeEvent`.
/// * The database operation to insert the stack attestation dispute fails.
///
/// # Behavior
///
/// The function performs the following steps:
/// 1. Deserializes the input `value` into a `StackAttestationDisputeEvent`.
/// 2. Converts the event into a stack attestation dispute object.
/// 3. Creates a new `StateManager` instance.
/// 4. Inserts the stack attestation dispute into the database using the `StateManager`.
#[instrument(level = "trace", skip_all)]
pub(crate) async fn handle_stack_attestation_dispute_event(
    value: Value,
    db: &SqlitePool,
) -> Result<()> {
    trace!("Processing stack attestation dispute event");
    let stack_attestation_event: StackAttestationDisputeEvent = serde_json::from_value(value)?;
    let stack_attestation_dispute = stack_attestation_event.into();
    let state_manager = StateManager::new(db.clone());
    state_manager
        .insert_stack_attestation_dispute(stack_attestation_dispute)
        .await?;
    Ok(())
}

/// Attempts to handle an Atoma event with retries.
///
/// This function will try to handle the event up to `MAX_RETRIES_FOR_UNHANDLED_EVENTS` times
/// if the initial attempt fails.
///
/// # Arguments
///
/// * `atoma_event` - A reference to the AtomaEvent to be handled.
/// * `event_name` - The name of the event, used for logging purposes.
///
/// # Returns
///
/// * `Result<(), Box<dyn std::error::Error>>` - Ok(()) if the event was handled successfully,
///   or an error if all retry attempts failed.
#[instrument(level = "trace", skip_all, fields(event))]
pub(crate) async fn handle_event_with_retries(
    event: &AtomaEvent,
    value: Value,
    db: &SqlitePool,
    node_small_ids: &[u64],
) {
    let mut retries = 0;
    while retries < MAX_RETRIES_FOR_UNHANDLED_EVENTS {
        retries += 1;
        trace!("Retrying event handling, attempt {retries}");
        match handle_atoma_event(event, value.clone(), db, node_small_ids).await {
            Ok(_) => return,
            Err(e) => {
                error!("Failed to handle event: {e}");
            }
        }
        tokio::time::sleep(Duration::from_millis(DURATION_FOR_RETRY_IN_MILLIS)).await;
    }
}
