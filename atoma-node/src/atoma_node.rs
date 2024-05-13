use std::path::Path;

use atoma_client::{AtomaSuiClient, AtomaSuiClientError};
use atoma_inference::{
    models::config::ModelsConfig,
    service::{ModelService, ModelServiceError},
};
use atoma_output_manager::{AtomaOutputManager, AtomaOutputManagerError};
use atoma_streamer::{AtomaStreamer, AtomaStreamerError};
use atoma_sui::subscriber::{SuiSubscriber, SuiSubscriberError};
use atoma_types::{Request, Response};
use thiserror::Error;
use tokio::{
    sync::{
        mpsc::{self, Receiver},
        oneshot,
    },
    task::JoinHandle,
    try_join,
};
use tracing::{error, info};

const CHANNEL_SIZE: usize = 32;

pub struct AtomaNode {
    pub atoma_sui_client_handle: JoinHandle<Result<(), AtomaNodeError>>,
    pub atoma_output_manager_handle: JoinHandle<Result<(), AtomaNodeError>>,
    pub atoma_streamer_handle: JoinHandle<Result<(), AtomaNodeError>>,
    pub model_service_handle: JoinHandle<Result<(), AtomaNodeError>>,
    pub sui_subscriber_handle: JoinHandle<Result<(), AtomaNodeError>>,
}

impl AtomaNode {
    pub async fn start<P>(
        atoma_sui_client_config_path: P,
        model_config_path: P,
        sui_subscriber_path: P,
        output_manager_config_path: P,
        streamer_config_path: P,
        json_server_req_rx: Receiver<(Request, oneshot::Sender<Response>)>,
    ) -> Result<(), AtomaNodeError>
    where
        P: AsRef<Path> + Send + 'static,
    {
        let model_config = ModelsConfig::from_file_path(model_config_path);

        let (subscriber_req_tx, subscriber_req_rx) = mpsc::channel(CHANNEL_SIZE);
        let (atoma_node_resp_tx, atoma_node_resp_rx) = mpsc::channel(CHANNEL_SIZE);
        let (output_manager_tx, output_manager_rx) = mpsc::channel(CHANNEL_SIZE);
        let (sync_streamer_tx, sync_streamer_rx) = std::sync::mpsc::channel();
        let (async_streamer_tx, async_streamer_rx) = tokio::sync::mpsc::channel(CHANNEL_SIZE);

        let model_service_handle = tokio::spawn(async move {
            info!("Spawning Model service..");
            let mut model_service = ModelService::start(
                model_config,
                json_server_req_rx,
                subscriber_req_rx,
                atoma_node_resp_tx,
                sync_streamer_tx,
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
            let atoma_sui_client = AtomaSuiClient::new_from_config_file(
                atoma_sui_client_config_path,
                atoma_node_resp_rx,
                output_manager_tx,
            )?;
            atoma_sui_client
                .run()
                .await
                .map_err(AtomaNodeError::AtomaSuiClientError)
        });

        let atoma_output_manager_handle = tokio::spawn(async move {
            info!("Starting Atoma output manager service..");
            let atoma_output_manager = AtomaOutputManager::new_from_config(
                output_manager_config_path.as_ref(),
                output_manager_rx,
            );
            atoma_output_manager
                .run()
                .await
                .map_err(AtomaNodeError::AtomaOutputManagerError)
        });

        // TODO: this seems a waste of resources, spanning a thread just to send a message to the streamer.
        // However, this allows to have a workaround to the fact that a `std::sync::mpsc::Receiver` shared across tasks
        // cannot work (as the later is not thread-safe). To be improved.
        std::thread::spawn(move || {
            while let Ok(message) = sync_streamer_rx.recv() {
                let _ = async_streamer_tx.blocking_send(message);
            }
        });

        let atoma_streamer_handle = tokio::spawn(async move {
            info!("Starting Atoma streamer service..");
            let atoma_streamer =
                AtomaStreamer::new_from_config(streamer_config_path, async_streamer_rx);
            atoma_streamer
                .run()
                .await
                .map_err(AtomaNodeError::AtomaStreamerError)
        });

        match try_join!(
            model_service_handle,
            sui_subscriber_handle,
            atoma_sui_client_handle,
            atoma_output_manager_handle,
            atoma_streamer_handle,
        ) {
            Ok((
                model_service_result,
                sui_subscriber_result,
                atoma_sui_client_result,
                atoma_output_manager_result,
                atoma_streamer_result,
            )) => {
                model_service_result?;
                sui_subscriber_result?;
                atoma_sui_client_result?;
                atoma_output_manager_result?;
                atoma_streamer_result?;
            }
            Err(e) => {
                error!("Failed to join handles, with error: {e}")
            }
        }

        Ok(())
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
    #[error("Atoma output manager error: `{0}`")]
    AtomaOutputManagerError(#[from] AtomaOutputManagerError),
    #[error("Atoma streamer error: `{0}`")]
    AtomaStreamerError(#[from] AtomaStreamerError),
}
