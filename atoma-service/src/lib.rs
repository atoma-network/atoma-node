use atoma_inference::service::{ModelService, ModelServiceError};
use atoma_sui::subscriber::{SuiSubscriber, SuiSubscriberError};
use thiserror::Error;
use tokio::{sync::mpsc, task::JoinHandle};

const CHANNEL_SIZE: usize = 32;

pub struct AtomaNode {
    inference_service_handle: JoinHandle<Result<(), AtomaNodeError>>,
    sui_subscriber_handle: JoinHandle<Result<(), AtomaNodeError>>,
    event_receiver: mpsc::Receiver<serde_json::Value>,
    inference_sender: mpsc::Sender<serde_json::Value>,
}

impl AtomaNode {
    pub fn start(mut inference_service: ModelService, sui_event_subscriber: SuiSubscriber) -> Self {
        let (inference_sender, inference_receiver) = mpsc::channel(CHANNEL_SIZE);
        let inference_service_handle = tokio::spawn(async move {
            inference_service
                .run()
                .await
                .map_err(AtomaNodeError::ModelServiceError)
        });

        let (event_sender, event_receiver) = mpsc::channel(CHANNEL_SIZE);

        let sui_subscriber_handle = tokio::spawn(async move {
            sui_event_subscriber
                .subscribe(event_sender)
                .await
                .map_err(AtomaNodeError::SuiSubscriberError)
        });

        Self {
            inference_service_handle,
            sui_subscriber_handle,
            event_receiver,
        }
    }

    pub async fn run(mut self) -> Result<(), AtomaNodeError> {
        while let Some(event) = self.event_receiver.recv().await {
            
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
}
