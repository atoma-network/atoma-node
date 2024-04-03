use sui_sdk::rpc_types::EventFilter;
use sui_sdk::{SuiClient, SuiClientBuilder};
use thiserror::Error;

pub struct Subscriber {
    sui_client: SuiClient,
}

impl Subscriber {
    pub async fn new(http_url: &str, ws_url: Option<&str>) -> Result<Self, SuiSubscriberError> {
        let mut sui_client_builder = SuiClientBuilder::default();
        if let Some(url) = ws_url {
            sui_client_builder = sui_client_builder.ws_url(url);
        }
        let sui_client = sui_client_builder.build(http_url).await?;
        Ok(Self { sui_client })
    }

    pub fn subscribe(&mut self) -> Result<(), SuiSubscriberError> {
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum SuiSubscriberError {
    #[error("Sui Builder error: `{0}`")]
    SuiBuilderError(#[from] sui_sdk::error::Error),
}
