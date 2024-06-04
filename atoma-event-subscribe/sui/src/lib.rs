use std::{fmt::Display, str::FromStr};

use subscriber::SuiSubscriberError;

pub mod config;
pub mod subscriber;

/// `AtomaEvent` - enum to keep track of all available
/// events being emitted by the Atoma smart contract,
/// on the Sui blockchain.
pub enum AtomaEvent {
    DisputeEvent,
    FirstSubmissionEvent,
    NewlySampledNodesEvent,
    NodeRegisteredEvent,
    NodeSubscribedToModelEvent,
    SettledEvent,
    Text2ImagePromptEvent,
    Text2TextPromptEvent,
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
        }
    }
}
