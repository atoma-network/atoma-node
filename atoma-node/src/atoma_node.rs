
use std::{io, path::Path};

use atoma_inference::{
    models::config::ModelsConfig,
    service::{ModelService, ModelServiceError},
    PrivateKey,
};
use atoma_sui::subscriber::{SuiSubscriber, SuiSubscriberError};
use serde_json::Value;
use thiserror::Error;
use tokio::{
    sync::{mpsc, mpsc::Receiver, oneshot},
    task::JoinHandle,
};
use tracing::info;

const CHANNEL_SIZE: usize = 32;

pub struct AtomaNode {
    pub model_service_handle: JoinHandle<Result<(), AtomaNodeError>>,
    pub sui_subscriber_handle: JoinHandle<Result<(), AtomaNodeError>>,
}

impl AtomaNode {
    pub async fn start<P>(
        model_config_path: P,
        private_key_path: P,
        sui_subscriber_path: P,
        json_server_req_rx: Receiver<(Value, oneshot::Sender<Value>)>,
    ) -> Result<Self, AtomaNodeError>
    where
        P: AsRef<Path> + Send + 'static,
    {
        let model_config = ModelsConfig::from_file_path(model_config_path);

        let private_key_bytes = std::fs::read(private_key_path)?;
        let private_key_bytes: [u8; 32] = private_key_bytes
            .try_into()
            .expect("Incorrect private key bytes length");

        let private_key = PrivateKey::from(private_key_bytes);

        let (subscriber_req_tx, subscriber_req_rx) = mpsc::channel(CHANNEL_SIZE);
        let (atoma_node_resp_tx, mut atoma_node_resp_rx) = mpsc::channel(CHANNEL_SIZE);

        let model_service_handle = tokio::spawn(async move {
            let mut model_service = ModelService::start(
                model_config,
                private_key,
                json_server_req_rx,
                subscriber_req_rx,
                atoma_node_resp_tx,
            )?;
            model_service
                .run()
                .await
                .map_err(AtomaNodeError::ModelServiceError)
        });

        let sui_subscriber_handle = tokio::spawn(async move {
            let sui_event_subscriber =
                SuiSubscriber::new_from_config(sui_subscriber_path, subscriber_req_tx).await?;
            sui_event_subscriber
                .subscribe()
                .await
                .map_err(AtomaNodeError::SuiSubscriberError)
        });

        while let Some(response) = atoma_node_resp_rx.recv().await {
            info!("Received new response: {response}");
        }

        Ok(Self {
            model_service_handle,
            sui_subscriber_handle,
        })
    }
}

#[derive(Debug, Error)]
pub enum AtomaNodeError {
    #[error("Model service error: `{0}`")]
    ModelServiceError(#[from] ModelServiceError),
    #[error("Sui subscriber error: `{0}`")]
    SuiSubscriberError(#[from] SuiSubscriberError),
    #[error("Private key error: `{0}`")]
    PrivateKeyError(#[from] io::Error),
}
