use crate::{events::AtomaEvent, subscriber::SuiEventSubscriberError};

pub type Result<T> = std::result::Result<T, SuiEventSubscriberError>;

pub(crate) fn handle_atoma_event(event: AtomaEvent) -> Result<()> {
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