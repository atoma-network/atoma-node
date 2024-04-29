use std::path::Path;

use atoma_client::{AtomaSuiClient, AtomaSuiClientError};
use atoma_inference::{
    models::config::ModelsConfig,
    service::{ModelService, ModelServiceError},
};
use atoma_sui::subscriber::{SuiSubscriber, SuiSubscriberError};
use atoma_types::{Request, Response};
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
    pub atoma_sui_client_handle: JoinHandle<Result<(), AtomaNodeError>>,
}

impl AtomaNode {
    pub async fn start<P>(
        atoma_sui_client_config_path: P,
        model_config_path: P,
        sui_subscriber_path: P,
        json_server_req_rx: Receiver<(Request, oneshot::Sender<Response>)>,
    ) -> Result<Self, AtomaNodeError>
    where
        P: AsRef<Path> + Send + 'static,
    {
        let model_config = ModelsConfig::from_file_path(model_config_path);

        let (subscriber_req_tx, subscriber_req_rx) = mpsc::channel(CHANNEL_SIZE);
        let (atoma_node_resp_tx, atoma_node_resp_rx) = mpsc::channel(CHANNEL_SIZE);

        let model_service_handle = tokio::spawn(async move {
            info!("Spawning Model service..");
            let mut model_service = ModelService::start(
                model_config,
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
            info!("Starting Sui subscriber service..");
            let sui_event_subscriber =
                SuiSubscriber::new_from_config(sui_subscriber_path, subscriber_req_tx).await?;
            sui_event_subscriber
                .subscribe()
                .await
                .map_err(AtomaNodeError::SuiSubscriberError)
        });

        let atoma_sui_client_handle = tokio::spawn(async move {
            info!("Starting Atoma Sui client service..");
            let atoma_sui_client =
                AtomaSuiClient::new_from_config(atoma_sui_client_config_path, atoma_node_resp_rx)?;
            atoma_sui_client
                .run()
                .await
                .map_err(AtomaNodeError::AtomaSuiClientError)
        });

        Ok(Self {
            model_service_handle,
            sui_subscriber_handle,
            atoma_sui_client_handle,
        })
    }
}

#[derive(Debug, Error)]
pub enum AtomaNodeError {
    #[error("Model service error: `{0}`")]
    ModelServiceError(#[from] ModelServiceError),
    #[error("Sui subscriber error: `{0}`")]
    SuiSubscriberError(#[from] SuiSubscriberError),
    #[error("Atoma Sui client error: `{0}`")]
    AtomaSuiClientError(#[from] AtomaSuiClientError),
}
