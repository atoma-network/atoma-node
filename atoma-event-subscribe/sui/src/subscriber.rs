use std::{
    fmt::Write,
    path::Path,
    str::FromStr,
    time::Duration,
};

use futures::StreamExt;
use serde_json::Value;
use sui_sdk::rpc_types::{EventFilter, SuiEvent};
use sui_sdk::types::base_types::{ObjectID, ObjectIDParseError};
use sui_sdk::types::event::EventID;
use sui_sdk::{SuiClient, SuiClientBuilder};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use crate::config::SuiSubscriberConfig;
use crate::AtomaEvent;
use atoma_types::{Request, SmallId, NON_SAMPLED_NODE_ERR};

const REQUEST_ID_HEX_SIZE: usize = 64;

pub struct SuiSubscriber {
    id: SmallId,
    sui_client: SuiClient,
    filter: EventFilter,
    event_sender: mpsc::Sender<Request>,
    last_event_id: Option<EventID>,
}

impl SuiSubscriber {
    pub async fn new(
        id: SmallId,
        http_url: &str,
        ws_url: Option<&str>,
        package_id: ObjectID,
        event_sender: mpsc::Sender<Request>,
        request_timeout: Option<Duration>,
    ) -> Result<Self, SuiSubscriberError> {
        let mut sui_client_builder = SuiClientBuilder::default();
        if let Some(duration) = request_timeout {
            sui_client_builder = sui_client_builder.request_timeout(duration);
        }
        if let Some(url) = ws_url {
            sui_client_builder = sui_client_builder.ws_url(url);
        }
        info!("Starting sui client..");
        let sui_client = sui_client_builder.build(http_url).await?;
        let filter = EventFilter::Package(package_id);
        Ok(Self {
            id,
            sui_client,
            filter,
            event_sender,
            last_event_id: None
        })
    }

    pub async fn new_from_config<P: AsRef<Path>>(
        config_path: P,
        event_sender: mpsc::Sender<Request>,
    ) -> Result<Self, SuiSubscriberError> {
        let config = SuiSubscriberConfig::from_file_path(config_path);
        let small_id = config.small_id();
        let http_url = config.http_url();
        let ws_url = config.ws_url();
        let package_id = config.package_id();
        let request_timeout = config.request_timeout();
        Self::new(
            small_id,
            &http_url,
            Some(&ws_url),
            package_id,
            event_sender,
            Some(request_timeout),
        )
        .await
    }

    pub async fn subscribe(mut self) -> Result<(), SuiSubscriberError> {
        let event_api = self.sui_client.event_api();
        let mut subscribe_event = event_api.subscribe_event(self.filter.clone()).await?;
        info!("Starting event while loop");
        while let Some(event) = subscribe_event.next().await {
            match event {
                Ok(event) => self.handle_event(event).await?,
                Err(e) => {
                    error!("Failed to get event with error: {e}");
                }
            }
        }
        Ok(())
    }

    pub async fn subscribe_pagination(self) -> Result<(), SuiSubscriberError> {
        let paged_events = self
            .sui_client
            .event_api()
            .query_events(self.filter, None, None, false)
            .await?;
        for event in paged_events.data.iter() {

        }
        Ok(())
    }
}

impl SuiSubscriber {
    async fn handle_event(&mut self, event: SuiEvent) -> Result<(), SuiSubscriberError> {
        self.last_event_id = Some(event.id);
        let event_type = event.type_.name.as_str();
        
        match AtomaEvent::from_str(event_type)? {
            AtomaEvent::DisputeEvent => todo!(),
            AtomaEvent::FirstSubmissionEvent
            | AtomaEvent::NodeRegisteredEvent
            | AtomaEvent::NodeSubscribedToModelEvent
            | AtomaEvent::SettledEvent => {
                info!("Received event: {}", event_type);
            }
            AtomaEvent::NewlySampledNodesEvent => {
                let event_data = event.parsed_json;
                match self.handle_newly_sampled_nodes_event(event_data).await {
                    Ok(()) => {}
                    Err(err) => {
                        error!("Failed to process request, with error: {err}")
                    }
                }
            }
            AtomaEvent::Text2ImagePromptEvent | AtomaEvent::Text2TextPromptEvent => {
                let event_data = event.parsed_json;
                match self.handle_prompt_event(event_data).await {
                    Ok(()) => {}
                    Err(SuiSubscriberError::TypeConversionError(err)) => {
                        if err.to_string().contains(NON_SAMPLED_NODE_ERR) {
                            info!("Node has not been sampled for current request")
                        } else {
                            error!("Failed to process request, with error: {err}")
                        }
                    }
                    Err(err) => {
                        error!("Failed to process request, with error: {err}")
                    }
                }
            }
        }
        Ok(())
    }

    async fn handle_prompt_event(&self, event_data: Value) -> Result<(), SuiSubscriberError> {
        debug!("event data: {}", event_data);
        let request = Request::try_from((self.id, event_data))?;
        info!("Received new request: {:?}", request);
        let request_id =
            request
                .id()
                .iter()
                .fold(String::with_capacity(REQUEST_ID_HEX_SIZE), |mut acc, &b| {
                    write!(acc, "{:02x}", b).expect("Failed to write to request_id");
                    acc
                });
        info!(
            "Current node has been sampled for request with id: {}",
            request_id
        );
        self.event_sender.send(request).await.map_err(Box::new)?;

        Ok(())
    }

    async fn handle_newly_sampled_nodes_event(
        &self,
        event_data: Value,
    ) -> Result<(), SuiSubscriberError> {
        debug!("event data: {}", event_data);
        let newly_sampled_nodes = extract_sampled_node_index(self.id, &event_data)?;
        if let Some(sampled_node_index) = newly_sampled_nodes {
            let ticket_id = extract_ticket_id(&event_data)?;
            let event_filter = EventFilter::MoveEventField {
                path: "ticket_id".to_string(),
                value: serde_json::from_str(ticket_id)?,
            };
            let data = self
                .sui_client
                .event_api()
                .query_events(event_filter, None, Some(1), false)
                .await?;
            let event = data
                .data
                .first()
                .ok_or(SuiSubscriberError::MalformedEvent(format!(
                    "Missing data from event with ticket id = {}",
                    ticket_id
                )))?;
            let request = Request::try_from((
                ticket_id,
                sampled_node_index as usize,
                event.parsed_json.clone(),
            ))?;
            info!("Received new request: {:?}", request);
            info!(
                "Current node has been newly sampled for request with id: {}",
                ticket_id
            );
            self.event_sender.send(request).await.map_err(Box::new)?;
        }

        Ok(())
    }
}

fn extract_sampled_node_index(id: u64, value: &Value) -> Result<Option<u64>, SuiSubscriberError> {
    let new_nodes = value
        .get("new_nodes")
        .ok_or_else(|| SuiSubscriberError::MalformedEvent("missing `new_nodes` field".into()))?
        .as_array()
        .ok_or_else(|| SuiSubscriberError::MalformedEvent("invalid `new_nodes` field".into()))?;

    Ok(new_nodes.iter().find_map(|n| {
        let node_id = n.get("node_id")?.get("inner")?.as_u64()?;
        let index = n.get("order")?.as_u64()?;
        if node_id == id {
            Some(index)
        } else {
            None
        }
    }))
}

fn extract_ticket_id(value: &Value) -> Result<&str, SuiSubscriberError> {
    value
        .get("ticket_id")
        .ok_or(SuiSubscriberError::MalformedEvent(
            "missing `ticket_id` field".into(),
        ))?
        .as_str()
        .ok_or(SuiSubscriberError::MalformedEvent(
            "invalid `ticket_id` field".into(),
        ))
}

#[derive(Debug, Error)]
pub enum SuiSubscriberError {
    #[error("Sui Builder error: `{0}`")]
    SuiBuilderError(#[from] sui_sdk::error::Error),
    #[error("Deserialize error: `{0}`")]
    DeserializeError(#[from] serde_json::Error),
    #[error("Object ID parse error: `{0}`")]
    ObjectIDParseError(#[from] ObjectIDParseError),
    #[error("Sender error: `{0}`")]
    SendError(#[from] Box<mpsc::error::SendError<Request>>),
    #[error("Type conversion error: `{0}`")]
    TypeConversionError(#[from] anyhow::Error),
    #[error("Malformed event: `{0}`")]
    MalformedEvent(String),
}
