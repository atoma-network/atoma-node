use std::{fmt::Display, str::FromStr};

use subscriber::SuiSubscriberError;

pub mod config;
pub mod subscriber;

/// Represents the various events emitted by the Atoma smart contract on the Sui blockchain.
///
/// This enum keeps track of all available events that can be emitted by the Atoma smart contract.
/// Each variant corresponds to a specific type of event that can occur within the Atoma ecosystem.
///
/// # Variants
///
/// * `DisputeEvent` - Emitted when a dispute is raised.
/// * `FirstSubmissionEvent` - Emitted when the first submission for a task is received.
/// * `NewlySampledNodesEvent` - Emitted when new nodes are sampled for a task.
/// * `NodeRegisteredEvent` - Emitted when a new node is registered in the network.
/// * `NodeSubscribedToModelEvent` - Emitted when a node subscribes to a specific model.
/// * `SettledEvent` - Emitted when a dispute or task is settled.
/// * `Text2ImagePromptEvent` - Emitted when a text-to-image prompt is submitted.
/// * `Text2TextPromptEvent` - Emitted when a text-to-text prompt is submitted.
/// * `ChatSessionEvent` - Emitted when a chat session is initiated or updated.
pub enum AtomaEvent {
    DisputeEvent,
    FirstSubmissionEvent,
    NewlySampledNodesEvent,
    NodeRegisteredEvent,
    NodeSubscribedToModelEvent,
    SettledEvent,
    Text2ImagePromptEvent,
    Text2TextPromptEvent,
    ChatSessionEvent,
}

impl FromStr for AtomaEvent {
    type Err = SuiSubscriberError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "DisputeEvent" => Ok(Self::DisputeEvent),
            "FirstSubmissionEvent" => Ok(Self::FirstSubmissionEvent),
            "NewlySampledNodesEvent" => Ok(Self::NewlySampledNodesEvent),
            "NodeRegisteredEvent" => Ok(Self::NodeRegisteredEvent),
            "NodeSubscribedToModelEvent" => Ok(Self::NodeSubscribedToModelEvent),
            "SettledEvent" => Ok(Self::SettledEvent),
            "Text2ImagePromptEvent" => Ok(Self::Text2ImagePromptEvent),
            "Text2TextPromptEvent" => Ok(Self::Text2TextPromptEvent),
            "ChatSessionEvent" => Ok(Self::ChatSessionEvent),
            _ => panic!("Invalid `AtomaEvent` string"),
        }
    }
}

impl Display for AtomaEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DisputeEvent => write!(f, "DisputeEvent"),
            Self::FirstSubmissionEvent => write!(f, "FirstSubmissionEvent"),
            Self::NewlySampledNodesEvent => write!(f, "NewlySampledNodesEvent"),
            Self::NodeRegisteredEvent => write!(f, "NodeRegisteredEvent"),
            Self::NodeSubscribedToModelEvent => write!(f, "NodeSubscribedToModelEvent"),
            Self::SettledEvent => write!(f, "SettledEvent"),
            Self::Text2ImagePromptEvent => write!(f, "Text2ImagePromptEvent"),
            Self::Text2TextPromptEvent => write!(f, "Text2TextPromptEvent"),
            Self::ChatSessionEvent => write!(f, "ChatSessionEvent"),
        }
    }
}
