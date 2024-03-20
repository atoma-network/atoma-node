use thiserror::Error;
use tokio::sync::{mpsc, oneshot, oneshot::error::RecvError};
use tracing::{debug, error};

use crate::{
    service::{ApiTrait, InferenceCore, InferenceCoreError},
    types::{InferenceRequest, InferenceResponse, ModelRequest, ModelResponse},
};

pub enum CoreThreadCommand {
    RunInference(InferenceRequest, oneshot::Sender<InferenceResponse>),
    FetchModel(ModelRequest, oneshot::Sender<ModelResponse>),
}

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("Core thread shutdown: `{0}`")]
    FailedInference(InferenceCoreError),
    #[error("Core thread shutdown: `{0}`")]
    FailedModelFetch(InferenceCoreError),
    #[error("Core thread shutdown: `{0}`")]
    Shutdown(RecvError),
}

pub struct CoreThread<T> {
    core: InferenceCore<T>,
    receiver: mpsc::Receiver<CoreThreadCommand>,
}

impl<T: ApiTrait> CoreThread<T> {
    pub async fn run(mut self) -> Result<(), CoreError> {
        debug!("Starting Core thread");

        while let Some(command) = self.receiver.recv().await {
            match command {
                CoreThreadCommand::RunInference(request, sender) => {
                    let InferenceRequest {
                        prompt,
                        model,
                        max_tokens,
                        temperature,
                        top_k,
                        top_p,
                        sampled_nodes,
                    } = request;
                    if !sampled_nodes.contains(&self.core.public_key) {
                        error!("Current node, with verification key = {:?} was not sampled from {sampled_nodes:?}", self.core.public_key);
                        continue;
                    }
                    let response = self.core.inference(
                        prompt,
                        model,
                        temperature,
                        max_tokens,
                        top_p,
                        top_k,
                    )?;
                    sender.send(response).ok();
                }
                CoreThreadCommand::FetchModel(request, sender) => {
                    let ModelRequest {
                        model,
                        quantization_method,
                    } = request;
                    let response = self.core.fetch_model(model, quantization_method)?;
                    sender.send(response).ok();
                }
            }
        }

        Ok(())
    }
}

impl From<InferenceCoreError> for CoreError {
    fn from(error: InferenceCoreError) -> Self {
        match error {
            InferenceCoreError::FailedInference(_) => CoreError::FailedInference(error),
            InferenceCoreError::FailedModelFetch(_) => CoreError::FailedModelFetch(error),
            InferenceCoreError::FailedApiConnection(_) => {
                panic!("API connection should have been already established")
            }
        }
    }
}
