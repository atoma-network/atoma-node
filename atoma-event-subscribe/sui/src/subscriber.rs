use futures::StreamExt;
use sui_sdk::rpc_types::EventFilter;
use sui_sdk::types::base_types::ObjectID;
use sui_sdk::{SuiClient, SuiClientBuilder};
use thiserror::Error;
use tracing::{error, info};

use crate::RequestType;

pub struct SuiSubscriber {
    sui_client: SuiClient,
    filter: EventFilter,
}

impl SuiSubscriber {
    pub async fn new(http_url: &str, ws_url: Option<&str>, object_id: ObjectID) -> Result<Self, SuiSubscriberError> {
        let mut sui_client_builder = SuiClientBuilder::default();
        if let Some(url) = ws_url {
            sui_client_builder = sui_client_builder.ws_url(url);
        }
        let sui_client = sui_client_builder.build(http_url).await?;
        let filter = EventFilter::Package(object_id);
        Ok(Self { sui_client, filter })
    }

    pub async fn subscribe(self) -> Result<(), SuiSubscriberError> {
        let event_api = self.sui_client.event_api();
        let mut subscribe_event = event_api.subscribe_event(self.filter).await?;
        while let Some(event) = subscribe_event.next().await { 
            match event { 
                Ok(event) => {
                    let event_data = event.parsed_json;
                    let request_type = serde_json::from_value::<RequestType>(event_data["type"].clone())?;
                    info!("The request type is: {:?}", request_type);
                },
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
    DeserializeError(#[from] serde_json::Error)
}
