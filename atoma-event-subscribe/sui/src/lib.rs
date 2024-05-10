use std::str::FromStr;

use subscriber::SuiSubscriberError;

pub mod config;
pub mod subscriber;

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

impl ToString for AtomaEvent {
    fn to_string(&self) -> String {
        match self {
            Self::DisputeEvent => "DisputeEvent".into(),
            Self::FirstSubmissionEvent => "FirstSubmissionEvent".into(),
            Self::NewlySampledNodesEvent => "NewlySampledNodesEvent".into(),
            Self::NodeRegisteredEvent => "NodeRegisteredEvent".into(),
            Self::NodeSubscribedToModelEvent => "NodeSubscribedToModelEvent".into(),
            Self::SettledEvent => "SettledEvent".into(),
            Self::Text2ImagePromptEvent => "Text2ImagePromptEvent".into(),
            Self::Text2TextPromptEvent => "Text2TextPromptEvent".into(),
        }
    }
}
