use ethers::{
    prelude::*,
    providers::{Provider, Ws},
};
use thiserror::Error;
use tracing::info;

use crate::AtomaEvent;

pub struct EthSubscriber {
    provider: Provider<Ws>,
    filter: Filter,
}

impl EthSubscriber {
    pub async fn new(ws_url: &str, contract_address: Address) -> Result<Self, EthSubscriberError> {
        let ws = Ws::connect(ws_url).await?;
        let provider = Provider::new(ws);
        let filter = Filter::new().address(contract_address);
        Ok(Self { provider, filter })
    }

    pub async fn subscribe(self) -> Result<(), EthSubscriberError> {
        let mut stream = self.provider.subscribe_logs(&self.filter).await?;

        while let Some(event) = stream.next().await {
            info!("Received a new event: {:?}", event);
            let event_data = serde_json::from_slice::<AtomaEvent>(&event.data)?;
            info!("New model request: {}", event_data.model);
        }

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum EthSubscriberError {
    #[error("WsClientError: `{0}`")]
    WsClientError(#[from] WsClientError),
    #[error("ProviderError: `{0}`")]
    ProviderError(#[from] ProviderError),
    #[error("DeserializeError: `{0}`")]
    DeserializeError(#[from] serde_json::Error),
}
