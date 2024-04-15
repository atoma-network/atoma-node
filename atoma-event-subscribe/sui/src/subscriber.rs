use std::path::Path;

use futures::StreamExt;
use serde_json::Value;
use sui_sdk::rpc_types::EventFilter;
use sui_sdk::types::base_types::{ObjectID, ObjectIDParseError};
use sui_sdk::{SuiClient, SuiClientBuilder};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{error, info};

use crate::config::SuiSubscriberConfig;
use crate::TextPromptParams;

pub struct SuiSubscriber {
    sui_client: SuiClient,
    filter: EventFilter,
    event_sender: mpsc::Sender<Value>,
}

impl SuiSubscriber {
    pub async fn new(
        http_url: &str,
        ws_url: Option<&str>,
        object_id: ObjectID,
        event_sender: mpsc::Sender<Value>,
    ) -> Result<Self, SuiSubscriberError> {
        let mut sui_client_builder = SuiClientBuilder::default();
        if let Some(url) = ws_url {
            sui_client_builder = sui_client_builder.ws_url(url);
        }
        info!("Starting sui client..");
        let sui_client = sui_client_builder.build(http_url).await?;
        let filter = EventFilter::Package(object_id);
        Ok(Self {
            sui_client,
            filter,
            event_sender,
        })
    }

    pub async fn new_from_config<P: AsRef<Path>>(
        config_path: P,
        event_sender: mpsc::Sender<Value>,
    ) -> Result<Self, SuiSubscriberError> {
        let config = SuiSubscriberConfig::from_file_path(config_path);
        let http_url = config.http_url();
        let ws_url = config.ws_url();
        let object_id = config.object_id();
        Self::new(&http_url, Some(&ws_url), object_id, event_sender).await
    }

    pub async fn subscribe(self) -> Result<(), SuiSubscriberError> {
        let event_api = self.sui_client.event_api();
        let mut subscribe_event = event_api.subscribe_event(self.filter).await.map_err(|e| {
            error!("Got error: {e}");
            e
        })?;
        info!("Starting event while loop");
        while let Some(event) = subscribe_event.next().await {
            match event {
                Ok(event) => {
                    let event_data = event.parsed_json;
                    info!("Received event: {event_data}");
                    let sampled_nodes = &event_data["nodes"];
                    let request =
                        serde_json::from_value::<TextPromptParams>(event_data["params"].clone())?;
                    info!(
                        "The request = {:?} and sampled_nodes = {:?}",
                        request, sampled_nodes
                    );
                    self.event_sender.send(event_data).await?;
                }
                Err(e) => {
                    error!("Failed to get event with error: {e}");
                }
            }
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
    SendError(#[from] mpsc::error::SendError<Value>),
}
