use futures::StreamExt;
use sui_sdk::rpc_types::EventFilter;
use sui_sdk::types::base_types::{ObjectID, ObjectIDParseError};
use sui_sdk::{SuiClient, SuiClientBuilder};
use thiserror::Error;
use tracing::{error, info};

use crate::TextPromptParams;

pub struct SuiSubscriber {
    sui_client: SuiClient,
    filter: EventFilter,
}

impl SuiSubscriber {
    pub async fn new(
        http_url: &str,
        ws_url: Option<&str>,
        object_id: ObjectID,
    ) -> Result<Self, SuiSubscriberError> {
        let mut sui_client_builder = SuiClientBuilder::default();
        if let Some(url) = ws_url {
            sui_client_builder = sui_client_builder.ws_url(url);
        }
        info!("Starting sui client..");
        let sui_client = sui_client_builder.build(http_url).await?;
        let filter = EventFilter::Package(object_id);
        Ok(Self { sui_client, filter })
    }

    pub async fn subscribe(self) -> Result<(), SuiSubscriberError> {
        let event_api = self.sui_client.event_api();
        let mut subscribe_event = event_api.subscribe_event(self.filter).await?;
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
}
