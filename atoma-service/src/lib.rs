use std::{
    io,
    path::{Path, PathBuf},
};

use atoma_inference::{
    models::config::{ModelConfig, ModelsConfig},
    service::{ModelService, ModelServiceError},
    PrivateKey,
};
use atoma_sui::subscriber::{SuiSubscriber, SuiSubscriberError};
use thiserror::Error;
use tokio::{sync::mpsc, task::JoinHandle};

const CHANNEL_SIZE: usize = 32;

pub struct AtomaNode {
    pub inference_service_handle: JoinHandle<Result<(), AtomaNodeError>>,
    pub sui_subscriber_handle: JoinHandle<Result<(), AtomaNodeError>>,
}

impl AtomaNode {
    pub async fn start<P: AsRef<Path>>(
        model_config_path: P,
        private_key_path: P,
        sui_subscriber_path: P,
    ) -> Result<Self, AtomaNodeError> {
        let model_config = ModelsConfig::from_file_path(model_config_path);

        let private_key_bytes = std::fs::read(private_key_path)?;
        let private_key_bytes: [u8; 32] = private_key_bytes
            .try_into()
            .expect("Incorrect private key bytes length");

        let private_key = PrivateKey::from(private_key_bytes);

        let (request_sender, request_receiver) = mpsc::channel(CHANNEL_SIZE);

        let inference_service_handle = tokio::spawn(async move {
            let model_service = ModelService::start(model_config, private_key, request_receiver)?;
            model_service
                .run()
                .await
                .map_err(AtomaNodeError::ModelServiceError)
        });

        let sui_subscriber_handle = tokio::spawn(async move {
            let sui_event_subscriber =
                SuiSubscriber::new_from_config(sui_subscriber_path, request_sender).await?;
            sui_event_subscriber
                .subscribe()
                .await
                .map_err(AtomaNodeError::SuiSubscriberError)
        });

        Self {
            inference_service_handle,
            sui_subscriber_handle,
        }
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
