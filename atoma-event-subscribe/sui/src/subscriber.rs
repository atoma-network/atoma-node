use std::{fmt::Write, path::Path, time::Duration};

use futures::StreamExt;
use serde_json::Value;
use sui_sdk::rpc_types::{EventFilter, SuiEvent};
use sui_sdk::types::base_types::{ObjectID, ObjectIDParseError};
use sui_sdk::{SuiClient, SuiClientBuilder};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use crate::config::SuiSubscriberConfig;
use atoma_types::{Request, SmallId};

const REQUEST_ID_HEX_SIZE: usize = 64;

pub struct SuiSubscriber {
    id: SmallId,
    sui_client: SuiClient,
    filter: EventFilter,
    event_sender: mpsc::Sender<Request>,
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

    pub async fn subscribe(self) -> Result<(), SuiSubscriberError> {
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
}

impl SuiSubscriber {
    async fn handle_event(&self, event: SuiEvent) -> Result<(), SuiSubscriberError> {
        match event.type_.name.as_str() {
            "DisputeEvent" => todo!(),
            "FirstSubmissionEvent"
            | "NodeRegisteredEvent"
            | "NodeSubscribedToModelEvent"
            | "SettledEvent" => {
                info!("Received event: {}", event.type_.name.as_str());
            }
            "Text2TextPromptEvent" | "NewlySampledNodesEvent" => {
                let event_data = event.parsed_json;
                self.handle_text2text_prompt_event(event_data).await?;
            }
            "Text2ImagePromptEvent" => {
                let event_data = event.parsed_json;
                self.handle_text2image_prompt_event(event_data).await?;
            }
            _ => panic!("Invalid Event type found!"),
        }
        Ok(())
    }

    async fn handle_text2image_prompt_event(
        &self,
        _event_data: Value,
    ) -> Result<(), SuiSubscriberError> {
        Ok(())
    }

    async fn handle_text2text_prompt_event(
        &self,
        event_data: Value,
    ) -> Result<(), SuiSubscriberError> {
        debug!("event data: {}", event_data);
        let request = Request::try_from(event_data)?;
        info!("Received new request: {:?}", request);
        let request_id =
            request
                .id()
                .iter()
                .fold(String::with_capacity(REQUEST_ID_HEX_SIZE), |mut acc, &b| {
                    write!(acc, "{:02x}", b).expect("Failed to write to request_id");
                    acc
                });
        info!("request_id: {request_id}");
        let sampled_nodes = request.sampled_nodes();
        if sampled_nodes.contains(&self.id) {
            info!(
                "Current node has been sampled for request with id: {}",
                request_id
            );
            self.event_sender.send(request).await.map_err(Box::new)?;
        } else {
            info!(
                "Current node has not been sampled for request with id: {}, ignoring it..",
                request_id
            );
        }
        Ok(())
    }
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
}
