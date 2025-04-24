use atoma_p2p::types::AtomaP2pEvent;
use atoma_sui::events::{
    AtomaEvent, ClaimedStackEvent, NewStackSettlementAttestationEvent,
    NodePublicKeyCommittmentEvent, NodeRegisteredEvent, NodeSubscribedToTaskEvent,
    NodeSubscriptionUpdatedEvent, NodeUnsubscribedFromTaskEvent, StackAttestationDisputeEvent,
    StackCreateAndUpdateEvent, StackCreatedEvent, StackSettlementTicketClaimedEvent,
    StackSettlementTicketEvent, StackSmallId, StackTrySettleEvent, TaskDeprecationEvent,
    TaskRegisteredEvent,
};
use tokio::sync::oneshot;
use tracing::{info, instrument};

use crate::{
    state_manager::Result,
    types::{
        AtomaAtomaStateManagerEvent, StackSettlementTicket, UpdateStackNumComputeUnitsAndClaimFunds,
    },
    AtomaStateManager, AtomaStateManagerError,
};

const RATIO_FOR_CLAIM_STACK_THRESHOLD: f64 = 0.95;

#[instrument(level = "info", skip_all)]
pub async fn handle_atoma_event(
    event: AtomaEvent,
    state_manager: &AtomaStateManager,
) -> Result<()> {
    match event {
        AtomaEvent::TaskRegisteredEvent(event) => handle_new_task_event(state_manager, event).await,
        AtomaEvent::TaskDeprecationEvent(event) => {
            handle_task_deprecation_event(state_manager, event).await
        }
        AtomaEvent::NodeSubscribedToTaskEvent(event) => {
            handle_node_task_subscription_event(state_manager, event).await
        }
        AtomaEvent::NodeSubscriptionUpdatedEvent(event) => {
            handle_node_task_subscription_updated_event(state_manager, event).await
        }
        AtomaEvent::NodeUnsubscribedFromTaskEvent(event) => {
            handle_node_task_unsubscription_event(state_manager, event).await
        }
        AtomaEvent::StackCreatedEvent((event, _)) => {
            handle_stack_created_event(state_manager, event).await
        }
        AtomaEvent::StackCreateAndUpdateEvent(event) => {
            handle_stack_create_and_update_event(state_manager, event).await
        }
        AtomaEvent::StackTrySettleEvent((event, _)) => {
            handle_stack_try_settle_event(state_manager, event).await
        }
        AtomaEvent::StackSettlementTicketEvent(event) => {
            handle_stack_settlement_ticket_event(state_manager, event).await
        }
        AtomaEvent::StackSettlementTicketClaimedEvent(event) => {
            handle_stack_settlement_ticket_claimed_event(state_manager, event).await
        }
        AtomaEvent::StackAttestationDisputeEvent(event) => {
            handle_stack_attestation_dispute_event(state_manager, event).await
        }
        AtomaEvent::NewStackSettlementAttestationEvent(event) => {
            handle_new_stack_settlement_attestation_event(state_manager, event).await
        }
        AtomaEvent::PublishedEvent(event) => {
            info!("Published event: {:?}", event);
            Ok(())
        }
        AtomaEvent::ClaimedStackEvent(event) => {
            handle_claimed_stack_event(state_manager, event).await
        }
        AtomaEvent::NodeRegisteredEvent((event, sender)) => {
            handle_node_registered_event(state_manager, event, sender.to_string()).await
        }
        AtomaEvent::NodeSubscribedToModelEvent(event) => {
            info!("Node subscribed to model event: {:?}", event);
            Ok(())
        }
        AtomaEvent::FirstSubmissionEvent(event) => {
            info!("First submission event: {:?}", event);
            Ok(())
        }
        AtomaEvent::DisputeEvent(event) => {
            info!("Dispute event: {:?}", event);
            Ok(())
        }
        AtomaEvent::NewlySampledNodesEvent(event) => {
            info!("Newly sampled nodes event: {:?}", event);
            Ok(())
        }
        AtomaEvent::SettledEvent(event) => {
            info!("Settled event: {:?}", event);
            Ok(())
        }
        AtomaEvent::RetrySettlementEvent(event) => {
            info!("Retry settlement event: {:?}", event);
            Ok(())
        }
        AtomaEvent::TaskRemovedEvent(event) => {
            info!("Task removed event: {:?}", event);
            Ok(())
        }
        AtomaEvent::Text2ImagePromptEvent(event) => {
            info!("Text2Image prompt event: {:?}", event);
            Ok(())
        }
        AtomaEvent::Text2TextPromptEvent(event) => {
            info!("Text2Text prompt event: {:?}", event);
            Ok(())
        }
        AtomaEvent::NewKeyRotationEvent(event) => {
            info!("New key rotation event: {:?}", event);
            Ok(())
        }
        AtomaEvent::NodePublicKeyCommittmentEvent(event) => {
            handle_node_key_rotation_event(state_manager, event).await
        }
    }
}

/// Handles a new task event by processing and inserting it into the database.
///
/// This function takes a serialized `TaskRegisteredEvent`, deserializes it, and
/// inserts the corresponding task into the database using the provided `AtomaStateManager`.
///
/// # Arguments
///
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations.
/// * `value` - A `serde_json::Value` containing the serialized `TaskRegisteredEvent`.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the task was successfully processed and inserted, or an error otherwise.
///
/// # Errors
///
/// This function will return an error if:
/// * The `value` cannot be deserialized into a `TaskRegisteredEvent`.
/// * The `AtomaStateManager` fails to insert the new task into the database.
#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_new_task_event(
    state_manager: &AtomaStateManager,
    event: TaskRegisteredEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-new-task-event",
        "Processing new task event"
    );
    let task = event.into();
    state_manager.state.insert_new_task(task).await?;
    Ok(())
}

/// Handles a task deprecation event.
///
/// This function processes a task deprecation event by parsing the event data,
/// extracting the necessary information, and updating the task's status in the database.
///
/// # Arguments
///
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations.
/// * `value` - A `serde_json::Value` containing the serialized task deprecation event data.
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
///
/// # Behavior
///
/// The function performs the following steps:
/// 1. Deserializes the input `value` into a `TaskDeprecationEvent`.
/// 2. Extracts the `task_small_id` and `epoch` from the event.
/// 3. Calls the `deprecate_task` method on the `AtomaStateManager` to update the task's status in the database.
#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_task_deprecation_event(
    state_manager: &AtomaStateManager,
    event: TaskDeprecationEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-task-deprecation-event",
        "Processing task deprecation event"
    );
    let task_small_id = event.task_small_id;
    let epoch = event.epoch;
    state_manager
        .state
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
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations.
/// * `event` - A `NodeSubscribedToTaskEvent` containing the details of the subscription event.
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
///
/// # Behavior
///
/// The function performs the following steps:
/// 1. Extracts the `node_small_id`, `task_small_id`, `price_per_one_million_compute_units`, and `max_num_compute_units` from the event.
/// 2. Calls the `subscribe_node_to_task` method on the `AtomaStateManager` to update the node's subscription in the database.
#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_node_task_subscription_event(
    state_manager: &AtomaStateManager,
    event: NodeSubscribedToTaskEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-node-task-subscription-event",
        "Processing node subscription event"
    );
    let node_small_id = event.node_small_id.inner as i64;
    let task_small_id = event.task_small_id.inner as i64;
    let price_per_one_million_compute_units = event.price_per_one_million_compute_units as i64;
    let max_num_compute_units = event.max_num_compute_units as i64;
    state_manager
        .state
        .subscribe_node_to_task(
            node_small_id,
            task_small_id,
            price_per_one_million_compute_units,
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
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations.
/// * `event` - A `NodeSubscriptionUpdatedEvent` containing the details of the subscription update.
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
///
/// # Behavior
///
/// The function performs the following steps:
/// 1. Extracts the `node_small_id`, `task_small_id`, `price_per_one_million_compute_units`, and `max_num_compute_units` from the event.
/// 2. Calls the `update_node_subscription` method on the `AtomaStateManager` to update the node's subscription in the database.
#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_node_task_subscription_updated_event(
    state_manager: &AtomaStateManager,
    event: NodeSubscriptionUpdatedEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-node-task-subscription-updated-event",
        "Processing node subscription updated event"
    );
    let node_small_id = event.node_small_id.inner as i64;
    let task_small_id = event.task_small_id.inner as i64;
    let price_per_one_million_compute_units = event.price_per_one_million_compute_units as i64;
    let max_num_compute_units = event.max_num_compute_units as i64;
    state_manager
        .state
        .update_node_subscription(
            node_small_id,
            task_small_id,
            price_per_one_million_compute_units,
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
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations.
/// * `event` - A `NodeUnsubscribedFromTaskEvent` containing the details of the unsubscription event.
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
///
/// # Behavior
///
/// The function performs the following steps:
/// 1. Extracts the `node_small_id` and `task_small_id` from the event.
/// 2. Calls the `unsubscribe_node_from_task` method on the `AtomaStateManager` to update the node's subscription status in the database.
#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_node_task_unsubscription_event(
    state_manager: &AtomaStateManager,
    event: NodeUnsubscribedFromTaskEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-node-task-unsubscription-event",
        "Processing node unsubscription event"
    );
    let node_small_id = event.node_small_id.inner as i64;
    let task_small_id = event.task_small_id.inner as i64;
    state_manager
        .state
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
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations.
/// * `event` - A `StackCreatedEvent` containing the details of the stack creation event.
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
/// The function performs the following steps:
/// 1. Extracts the `selected_node_id` from the event.
/// 2. Checks if the `selected_node_id` is present in the `node_small_ids` slice.
/// 3. If the node is valid, it converts the event into a stack object and inserts it into the database.
#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_stack_created_event(
    state_manager: &AtomaStateManager,
    event: StackCreatedEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-stack-created-event",
        "Processing stack created event"
    );
    info!(
        target = "atoma-state-handlers",
        event = "handle-stack-created-event",
        "Stack selected current node, with id {}, inserting new stack with id {}",
        event.selected_node_id.inner,
        event.stack_small_id.inner
    );
    let stack = event.into();
    state_manager.state.insert_new_stack(stack).await?;
    Ok(())
}

#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_stack_create_and_update_event(
    state_manager: &AtomaStateManager,
    event: StackCreateAndUpdateEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-stack-create-and-update-event",
        "Processing stack create and update event for stack with id {}",
        event.stack_small_id.inner
    );
    let stack = event.into();
    state_manager.state.insert_new_stack(stack).await?;
    Ok(())
}

/// Handles a stack try settle event.
///
/// This function processes a stack try settle event by parsing the event data,
/// converting it into a stack settlement ticket, and inserting it into the database.
///
/// # Arguments
///
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations.
/// * `event` - A `StackTrySettleEvent` containing the details of the stack try settle event.
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
/// 1. Converts the `StackTrySettleEvent` into a stack settlement ticket.
/// 2. Calls the `insert_new_stack_settlement_ticket` method on the `AtomaStateManager` to insert the ticket into the database.
#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_stack_try_settle_event(
    state_manager: &AtomaStateManager,
    event: StackTrySettleEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-stack-try-settle-event",
        "Processing stack try settle event"
    );
    let stack_settlement_ticket = StackSettlementTicket::try_from(event)?;
    state_manager
        .state
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
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations.
/// * `event` - A `NewStackSettlementAttestationEvent` containing the details of the attestation event.
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
///
/// # Behavior
///
/// The function performs the following steps:
/// 1. Extracts the `stack_small_id`, `attestation_node_id`, `committed_stack_proof`, and `stack_merkle_leaf` from the event.
/// 2. Calls the `update_stack_settlement_ticket_with_attestation_commitments` method on the `AtomaStateManager` to update the database.
#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_new_stack_settlement_attestation_event(
    state_manager: &AtomaStateManager,
    event: NewStackSettlementAttestationEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-new-stack-settlement-attestation-event",
        "Processing new stack settlement attestation event"
    );
    let stack_small_id = event.stack_small_id.inner as i64;
    let attestation_node_id = event.attestation_node_id.inner as i64;
    let committed_stack_proof = event.committed_stack_proof;
    let stack_merkle_leaf = event.stack_merkle_leaf;

    state_manager
        .state
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
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations.
/// * `event` - A `StackSettlementTicketEvent` containing the details of the stack settlement ticket event.
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
/// 1. Extracts the `stack_small_id` and `dispute_settled_at_epoch` from the event.
/// 2. Calls the `settle_stack_settlement_ticket` method on the `AtomaStateManager` to update the database.
#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_stack_settlement_ticket_event(
    state_manager: &AtomaStateManager,
    event: StackSettlementTicketEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-stack-settlement-ticket-event",
        "Processing stack settlement ticket event"
    );
    let stack_small_id = event.stack_small_id.inner as i64;
    let dispute_settled_at_epoch = event.dispute_settled_at_epoch as i64;
    state_manager
        .state
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
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations.
/// * `event` - A `StackSettlementTicketClaimedEvent` containing the details of the stack settlement ticket claimed event.
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
/// 1. Extracts the `stack_small_id` and `user_refund_amount` from the event.
/// 2. Calls the `update_stack_settlement_ticket_with_claim` method on the `AtomaStateManager` to update the database.
#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_stack_settlement_ticket_claimed_event(
    state_manager: &AtomaStateManager,
    event: StackSettlementTicketClaimedEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-stack-settlement-ticket-claimed-event",
        "Processing stack settlement ticket claimed event"
    );
    let stack_small_id = event.stack_small_id.inner as i64;
    let user_refund_amount = event.user_refund_amount as i64;
    state_manager
        .state
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
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations.
/// * `event` - A `StackAttestationDisputeEvent` containing the details of the dispute event.
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
/// 1. Converts the `StackAttestationDisputeEvent` into a stack attestation dispute object.
/// 2. Calls the `insert_stack_attestation_dispute` method on the `AtomaStateManager` to insert the dispute into the database.
#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_stack_attestation_dispute_event(
    state_manager: &AtomaStateManager,
    event: StackAttestationDisputeEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-stack-attestation-dispute-event",
        "Processing stack attestation dispute event"
    );
    let stack_attestation_dispute = event.into();
    state_manager
        .state
        .insert_stack_attestation_dispute(stack_attestation_dispute)
        .await?;
    Ok(())
}

/// Marks a stack as claimed in the database after a successful claim event.
///
/// This method is typically called as part of processing a `ClaimedStackEvent`, after updating
/// the stack settlement ticket with claim information. It updates the `is_claimed` field to `TRUE`
/// for the specified stack in the `stacks` table.
///
/// # Arguments
///
/// * `stack_small_id` - The unique identifier of the stack to be marked as claimed.
///
/// # Returns
///
/// * `Result<()>` - Returns `Ok(())` if the update was successful, or an error if the database
///   operation fails.
///
/// # Errors
///
/// Returns `AtomaStateManagerError::DatabaseConnectionError` if there's an issue executing
/// the database query.
///
/// # Usage
///
/// This method is commonly used in conjunction with `update_stack_settlement_ticket_with_claim`
/// when processing claimed stack events:
///
/// ```rust,ignore
/// state_manager.state.update_stack_settlement_ticket_with_claim(stack_small_id, user_refund_amount).await?;
/// state_manager.state.update_stack_is_claimed(stack_small_id).await?;
/// ```
#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_claimed_stack_event(
    state_manager: &AtomaStateManager,
    event: ClaimedStackEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-claimed-stack-event",
        "Processing claimed stack event"
    );
    let ClaimedStackEvent {
        stack_small_id: StackSmallId {
            inner: stack_small_id,
        },
        user_refund_amount,
        ..
    } = event;
    state_manager
        .state
        .update_stack_is_claimed(stack_small_id as i64, user_refund_amount as i64)
        .await?;
    Ok(())
}

/// Handles events related to the state manager.
///
/// This function processes various events that are sent to the state manager,
/// including requests to get available stacks with compute units, update the number
/// of compute units for a stack, and update the total hash of a stack.
///
/// # Arguments
///
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations.
/// * `event` - An `AtomaAtomaStateManagerEvent` enum that specifies the type of event to handle.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the event was processed successfully, or an error if something went wrong.
///
/// # Errors
///
/// This function may return an error if:
/// * The database operations for updating compute units or hashes fail.
/// * The result sender fails to send the result for the `GetAvailableStackWithComputeUnits` event.
///
/// # Behavior
///
/// The function performs the following steps:
/// 1. Matches the incoming event to determine the type of operation to perform.
/// 2. For `GetAvailableStackWithComputeUnits`, it retrieves the available stack and sends the result.
/// 3. For `UpdateStackNumComputeUnits`, it updates the number of compute units for the specified stack.
/// 4. For `UpdateStackTotalHash`, it updates the total hash for the specified stack.
#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_state_manager_event(
    state_manager: &AtomaStateManager,
    event: AtomaAtomaStateManagerEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-state-manager-event",
        "Processing state manager event"
    );
    match event {
        AtomaAtomaStateManagerEvent::GetAvailableStackWithComputeUnits {
            stack_small_id,
            sui_address,
            total_num_compute_units,
            result_sender,
        } => {
            let result = state_manager
                .state
                .get_available_stack_with_compute_units(
                    stack_small_id,
                    &sui_address,
                    total_num_compute_units,
                )
                .await;
            result_sender
                .send(result)
                .map_err(|_| AtomaStateManagerError::ChannelSendError)?;
        }
        AtomaAtomaStateManagerEvent::UpdateStackNumComputeUnits {
            stack_small_id,
            estimated_total_compute_units,
            total_compute_units,
            concurrent_requests,
        } => {
            handle_update_stack_num_compute_units_and_claim_funds(
                state_manager,
                stack_small_id,
                estimated_total_compute_units,
                total_compute_units,
                concurrent_requests,
            )
            .await?;
        }
        AtomaAtomaStateManagerEvent::UpdateStackTotalHash {
            stack_small_id,
            total_hash,
        } => {
            state_manager
                .state
                .update_stack_total_hash(stack_small_id, total_hash)
                .await?;
        }
    }
    Ok(())
}

/// Handles an update to the number of compute units in a stack.
///
/// This function processes an update to the number of compute units in a stack by parsing the event data
/// and updating the corresponding stack in the database. If the ratio of compute units is greater than or equal to 95%,
/// it will submit a claim funds for stacks transaction.
///
/// # Arguments
///
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations.
/// * `stack_small_id` - The unique identifier of the stack to be updated.
/// * `estimated_total_compute_units` - The estimated total number of compute units in the stack.
/// * `total_compute_units` - The total number of compute units in the stack.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the update was successful, or an error if something went wrong.
///
/// # Errors
///
/// This function will return an error if:
/// * The database operation to update the stack number of compute units fails.
/// * The Sui client operation to submit the claim funds for stacks transaction fails.
#[instrument(
    level = "info",
    skip_all,
    fields(
        stack_small_id,
        estimated_total_compute_units,
        total_compute_units,
        ratio
    )
)]
#[allow(clippy::cast_sign_loss)]
pub(crate) async fn handle_update_stack_num_compute_units_and_claim_funds(
    state_manager: &AtomaStateManager,
    stack_small_id: i64,
    estimated_total_compute_units: i64,
    total_compute_units: i64,
    concurrent_requests: u64,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-update-stack-num-compute-units-and-claim-funds",
        "Processing update stack num compute units and claim funds"
    );
    let UpdateStackNumComputeUnitsAndClaimFunds {
        ratio,
        stack_computed_units,
        is_confidential,
        is_locked_for_claim,
        was_claimed,
    } = state_manager
        .state
        .update_stack_num_compute_units(
            stack_small_id,
            estimated_total_compute_units,
            total_compute_units,
            RATIO_FOR_CLAIM_STACK_THRESHOLD,
            concurrent_requests as i64,
        )
        .await?;
    info!(
        target = "atoma-state-handlers",
        event = "handle-update-stack-num-compute-units-and-claim-funds",
        "Stack {} has ratio {} with total compute units {} confidential state {} and is locked for claim {}",
        stack_small_id, ratio, total_compute_units, is_confidential, is_locked_for_claim
    );
    if is_confidential
        && ratio >= RATIO_FOR_CLAIM_STACK_THRESHOLD
        && concurrent_requests == 0
        && !was_claimed
    {
        info!(
            target = "atoma-state-handlers",
            event = "handle-update-stack-num-compute-units-and-claim-funds",
            "Submitting claim funds for locked stack {} with ratio {} with total compute units {} confidential state {} and is locked for claim {}",
            stack_small_id, ratio, total_compute_units, is_confidential, is_locked_for_claim
        );
        let mut client = state_manager.client.write().await;
        if let Err(e) = client
            .submit_claim_funds_for_stacks_tx(
                vec![stack_small_id as u64],
                None,
                vec![stack_computed_units as u64],
                None,
                None,
                None,
            )
            .await
        {
            tracing::error!(
                    target = "atoma-state-handlers",
                    event = "handle-update-stack-num-compute-units-and-claim-funds",
                    "Failed to submit claim funds for locked stack {} with ratio {} with total compute units {} confidential state {} and is locked for claim {}, with error {}",
                    stack_small_id, ratio, total_compute_units, is_confidential, is_locked_for_claim, e
                );
            return Err(AtomaStateManagerError::SuiClientError(e));
        }
    }
    Ok(())
}

/// Handles a p2p event.
///
/// This function processes a p2p event by parsing the event data
/// and updating the corresponding stack settlement ticket in the database.
#[instrument(level = "info", skip_all)]
pub(crate) async fn handle_p2p_event(
    state_manager: &AtomaStateManager,
    event: AtomaP2pEvent,
    sender: oneshot::Sender<bool>,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-p2p-event",
        "Processing p2p event"
    );
    match event {
        AtomaP2pEvent::NodeMetricsRegistrationEvent { .. } => {
            // NOTE: Atoma nodes do not need to register public URLs of other peer nodes, and this event is unreachable
            // from the Atoma state manager service
            unreachable!("Atoma nodes do not need to register public URLs of other peer nodes");
        }
        AtomaP2pEvent::VerifyNodeSmallIdOwnership {
            node_small_id,
            sui_address,
        } => {
            let result = handle_node_small_id_ownership_verification_event(
                state_manager,
                node_small_id,
                sui_address,
            )
            .await;
            sender
                .send(result.is_ok())
                .map_err(|_| AtomaStateManagerError::ChannelSendError)?;
        }
    }
    Ok(())
}

/// Handles a node key rotation event.
///
/// This function processes a node key rotation event by extracting the relevant data
/// and updating the node's public key in the database.
///
/// # Arguments
///
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations.
/// * `event` - A `NodePublicKeyCommittmentEvent` containing the details of the key rotation event.
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the event was processed successfully, or an error if something went wrong.
///
/// # Errors
///
/// This function will return an error if:
/// * The database operation to insert the new node public key rotation fails.
///
/// # Behavior
///
/// The function performs the following steps:
/// 1. Extracts the `epoch`, `node_id`, `new_public_key`, and `tee_remote_attestation_bytes` from the event.
/// 2. Calls the `insert_node_public_key_rotation` method on the `AtomaStateManager` to update the node's public key in the database.
#[instrument(level = "info", skip_all)]
async fn handle_node_key_rotation_event(
    state_manager: &AtomaStateManager,
    event: NodePublicKeyCommittmentEvent,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-new-key-rotation-event",
        "Processing new key rotation event"
    );
    let NodePublicKeyCommittmentEvent {
        epoch,
        key_rotation_counter,
        node_id,
        new_public_key,
        evidence_bytes,
        device_type,
    } = event;
    state_manager
        .state
        .insert_node_public_key_rotation(
            epoch,
            key_rotation_counter,
            node_id.inner,
            new_public_key,
            evidence_bytes,
            device_type,
        )
        .await?;
    Ok(())
}

/// Handles a node small ID ownership verification event.
///
/// This function processes events that verify the ownership relationship between a node's small ID
/// and its corresponding Sui blockchain address. It updates the database to record this verified
/// ownership relationship.
///
/// # Arguments
///
/// * `state_manager` - A reference to the `AtomaStateManager` for database operations
/// * `node_small_id` - The small ID of the node being verified
/// * `sui_address` - The Sui blockchain address claiming ownership of the node
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the verification was processed successfully, or an error if something went wrong
///
/// # Errors
///
/// This function will return an error if:
/// * The database operation to verify the node small ID ownership fails
/// * The state manager encounters any internal errors during verification
///
/// # Example
///
/// ```rust,ignore
/// use atoma_state::AtomaStateManager;
///
/// async fn example(state_manager: &AtomaStateManager) {
///     let node_small_id = 123;
///     let sui_address = "0x123...".to_string();
///     
///     handle_node_small_id_ownership_verification_event(
///         state_manager,
///         node_small_id,
///         sui_address
///     ).await.expect("Failed to verify node ownership");
/// }
/// ```
#[instrument(level = "info", skip_all)]
async fn handle_node_small_id_ownership_verification_event(
    state_manager: &AtomaStateManager,
    node_small_id: u64,
    sui_address: String,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-node-small-id-ownership-verification-event",
        "Processing node small ID ownership verification event"
    );
    state_manager
        .state
        .verify_node_small_id_ownership(node_small_id, sui_address)
        .await?;
    Ok(())
}

/// Handles a node registration event by recording the node's registration details in the database.
///
/// This function processes events that occur when a new node is registered in the system. It records
/// the node's small ID (a compact identifier), badge ID (representing node capabilities/permissions),
/// and blockchain address in the state database.
///
/// # Arguments
///
/// * `state_manager` - A reference to the `AtomaStateManager` that provides database operations
/// * `event` - A `NodeRegisteredEvent` containing the node's registration details:
///   * `node_small_id` - Compact identifier for the node
///   * `badge_id` - Identifier representing the node's capabilities/permissions
/// * `node_sui_address` - The blockchain address associated with the registered node
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if the registration was recorded successfully, or an error if something went wrong
///
/// # Errors
///
/// This function will return an error if:
/// * The database operation to insert the node registration fails
/// * The state manager encounters any internal errors during the insertion
///
/// # Example
///
/// ```rust,ignore
/// use atoma_state::AtomaStateManager;
/// use atoma_sui::events::NodeRegisteredEvent;
///
/// async fn example(state_manager: &AtomaStateManager) {
///     let event = NodeRegisteredEvent {
///         node_small_id: /* ... */,
///         badge_id: /* ... */
///     };
///     let node_sui_address = "0x123...".to_string();
///     
///     handle_node_registered_event(
///         state_manager,
///         event,
///         node_sui_address
///     ).await.expect("Failed to handle node registration");
/// }
/// ```
///
/// # Implementation Details
///
/// The function:
/// 1. Logs the processing of the registration event
/// 2. Extracts the node_small_id and badge_id from the event
/// 3. Converts the node_small_id to i64 for database compatibility
/// 4. Inserts the registration record into the database via the state manager
#[instrument(level = "info", skip_all)]
async fn handle_node_registered_event(
    state_manager: &AtomaStateManager,
    event: NodeRegisteredEvent,
    node_sui_address: String,
) -> Result<()> {
    info!(
        target = "atoma-state-handlers",
        event = "handle-node-registered-event",
        "Processing node registered event"
    );
    let NodeRegisteredEvent {
        node_small_id,
        badge_id,
    } = event;
    state_manager
        .state
        .insert_node_registration_event(node_small_id.inner as i64, badge_id, node_sui_address)
        .await?;
    Ok(())
}
