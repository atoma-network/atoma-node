use std::time::Duration;

use tracing::error;

use crate::{events::AtomaEvent, subscriber::SuiEventSubscriberError};

/// The duration for retries for events to which handling fails.
const DURATION_FOR_RETRY_IN_MILLIS: u64 = 500;
/// The maximum number of retries for events to which handling fails.
const MAX_RETRIES_FOR_UNHANDLED_EVENTS: usize = 3;

pub type Result<T> = std::result::Result<T, SuiEventSubscriberError>;

pub(crate) fn handle_atoma_event(event: &AtomaEvent) -> Result<()> {
    match event {
        AtomaEvent::DisputeEvent => {
            println!("DisputeEvent");
        }
        AtomaEvent::SettledEvent => {
            println!("SettledEvent");
        }
        AtomaEvent::PublishedEvent => {
            println!("PublishedEvent");
        }
        AtomaEvent::NewlySampledNodesEvent => {
            println!("NewlySampledNodesEvent");
        }
        AtomaEvent::NodeRegisteredEvent => {
            println!("NodeRegisteredEvent");
        }
        AtomaEvent::NodeSubscribedToModelEvent => {
            println!("NodeSubscribedToModelEvent");
        }
        AtomaEvent::NodeSubscribedToTaskEvent => {
            println!("NodeSubscribedToTaskEvent");
        }
        AtomaEvent::NodeUnsubscribedFromTaskEvent => {
            println!("NodeUnsubscribedFromTaskEvent");
        }
        AtomaEvent::TaskRegisteredEvent => {
            println!("TaskRegisteredEvent");
        }
        AtomaEvent::TaskDeprecationEvent => {
            println!("TaskDeprecationEvent");
        }
        AtomaEvent::FirstSubmissionEvent => {
            println!("FirstSubmissionEvent");
        }
        AtomaEvent::StackCreatedEvent => {
            println!("StackCreatedEvent");
        }
        AtomaEvent::StackTrySettleEvent => {
            println!("StackTrySettleEvent");
        }
        AtomaEvent::NewStackSettlementAttestationEvent => {
            println!("NewStackSettlementAttestationEvent");
        }
        AtomaEvent::StackSettlementTicketEvent => {
            println!("StackSettlementTicketEvent");
        }
        AtomaEvent::StackSettlementTicketClaimedEvent => {
            println!("StackSettlementTicketClaimedEvent");
        }
        AtomaEvent::StackAttestationDisputeEvent => {
            println!("StackAttestationDisputeEvent");
        }
        AtomaEvent::TaskRemovedEvent => {
            println!("TaskRemovedEvent");
        }
        AtomaEvent::RetrySettlementEvent => {
            println!("RetrySettlementEvent");
        }
        AtomaEvent::Text2ImagePromptEvent => {
            println!("Text2ImagePromptEvent");
        }
        AtomaEvent::Text2TextPromptEvent => {
            println!("Text2TextPromptEvent");
        }
    }
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
pub(crate) async fn handle_event_with_retries(event: &AtomaEvent) {
    let mut retries = 0;
    while retries < MAX_RETRIES_FOR_UNHANDLED_EVENTS {
        retries += 1;
        match handle_atoma_event(event) {
            Ok(_) => return,
            Err(e) => {
                error!("Failed to handle event: {e}");
            }
        }
        tokio::time::sleep(Duration::from_millis(DURATION_FOR_RETRY_IN_MILLIS)).await;
    }
}
