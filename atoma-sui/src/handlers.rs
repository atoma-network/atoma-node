use atoma_state::{SqlitePool, StateManager};
use serde_json::Value;
use tracing::{error, instrument, trace};

use crate::{
    events::{AtomaEvent, NodeSubscribedToTaskEvent, NodeSubscriptionUpdatedEvent, NodeUnsubscribedFromTaskEvent, StackCreatedEvent, TaskDeprecationEvent, TaskRegisteredEvent},
    subscriber::Result,
};

/// The maximum number of retries for events to which handling fails.
const MAX_RETRIES_FOR_UNHANDLED_EVENTS: usize = 3;

pub(crate) async fn handle_atoma_event(
    event: &AtomaEvent,
    value: Value,
    db: &SqlitePool,
    node_small_ids: &[u64],
) -> Result<()> {
    match event {
        AtomaEvent::DisputeEvent => {
            todo!()
        }
        AtomaEvent::SettledEvent => {
            todo!()
        }
        AtomaEvent::PublishedEvent => {
            todo!()
        }
        AtomaEvent::NewlySampledNodesEvent => {
            todo!()
        }
        AtomaEvent::NodeRegisteredEvent => {
            todo!()
        }
        AtomaEvent::NodeSubscribedToModelEvent => {
            todo!()
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
        AtomaEvent::StackTrySettleEvent => {
            todo!()
        }
        AtomaEvent::NewStackSettlementAttestationEvent => {
            todo!()
        }
        AtomaEvent::StackSettlementTicketEvent => {
            todo!()
        }
        AtomaEvent::StackSettlementTicketClaimedEvent => {
            todo!()
        }
        AtomaEvent::StackAttestationDisputeEvent => {
            todo!()
        }
        AtomaEvent::TaskRemovedEvent => {
            todo!()
        }
        AtomaEvent::RetrySettlementEvent => {
            todo!()
        }
        AtomaEvent::Text2ImagePromptEvent => {
            todo!()
        }
        AtomaEvent::Text2TextPromptEvent => {
            todo!()
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
        .remove_node_subscription(node_small_id, task_small_id)
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
pub(crate) async fn handle_stack_created_event(value: Value, db: &SqlitePool, node_small_ids: &[u64]) -> Result<()> {
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


/// Attempts to handle an Atoma event with retries.
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
pub(crate) async fn handle_event_with_retries(event: &AtomaEvent, value: Value, db: &SqlitePool, node_small_ids: &[u64]) {
    let mut retries = 0;
    while retries < MAX_RETRIES_FOR_UNHANDLED_EVENTS {
        retries += 1;
        match handle_atoma_event(event, value.clone(), db, node_small_ids).await {
            Ok(_) => return,
            Err(e) => {
                error!("Failed to handle event: {e}");
            }
        }
    }
}
