use thiserror::Error;
use tokio::{
    sync::{
        mpsc,
        oneshot::{self, error::RecvError},
    },
    task::JoinHandle,
};
use tracing::{debug, error, warn};

use crate::{
    apis::ApiTrait,
    core::{InferenceCore, InferenceCoreError},
    types::{InferenceRequest, InferenceResponse, ModelRequest, ModelResponse},
};

const CORE_THREAD_COMMANDS_CHANNEL_SIZE: usize = 32;

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

pub struct CoreThreadHandle {
    sender: mpsc::Sender<CoreThreadCommand>,
    join_handle: JoinHandle<()>,
}

impl CoreThreadHandle {
    pub async fn stop(self) {
        // drop the sender, this will force all the other weak senders to not be able to upgrade.
        drop(self.sender);
        self.join_handle.await.ok();
    }
}

pub struct CoreThread<T> {
    core: InferenceCore<T>,
    receiver: mpsc::Receiver<CoreThreadCommand>,
}

impl<T: ApiTrait> CoreThread<T> {
    pub async fn run(mut self) -> Result<(), CoreError> {
        debug!("Starting Core thread");

        // let models = self.core.config.models();
        // for model_type in models {
        //     let (model_sender, model_receiver) = std::sync::mpsc::channel();
        //     let
        //     std::thread::spawn(move || {
        //         while Ok(request) = model_receiver.recv() {

        //         }
        //     });
        // }

        while let Some(command) = self.receiver.recv().await {
            match command {
                CoreThreadCommand::RunInference(request, sender) => {
                    let InferenceRequest {
                        prompt,
                        model,
                        max_tokens,
                        temperature,
                        random_seed,
                        repeat_penalty,
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
                        random_seed,
                        repeat_penalty,
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
                    let response = self.core.fetch_model(model, quantization_method).await?;
                    sender.send(response).ok();
                }
            }
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct CoreThreadDispatcher {
    sender: mpsc::WeakSender<CoreThreadCommand>,
}

impl CoreThreadDispatcher {
    pub(crate) fn start<T: ApiTrait + Send + 'static>(
        core: InferenceCore<T>,
    ) -> (Self, CoreThreadHandle) {
        let (sender, receiver) = mpsc::channel(CORE_THREAD_COMMANDS_CHANNEL_SIZE);
        let core_thread = CoreThread { core, receiver };

        let join_handle = tokio::task::spawn(async move {
            if let Err(e) = core_thread.run().await {
                if !matches!(e, CoreError::Shutdown(_)) {
                    panic!("Fatal error occurred: {e}");
                }
            }
        });

        let dispatcher = Self {
            sender: sender.downgrade(),
        };
        let handle = CoreThreadHandle {
            join_handle,
            sender,
        };

        (dispatcher, handle)
    }

    async fn send(&self, command: CoreThreadCommand) {
        if let Some(sender) = self.sender.upgrade() {
            if let Err(e) = sender.send(command).await {
                warn!("Could not send command to thread core, it might be shutting down: {e}");
            }
        }
    }
}

impl CoreThreadDispatcher {
    pub(crate) async fn fetch_model(
        &self,
        request: ModelRequest,
    ) -> Result<ModelResponse, CoreError> {
        let (sender, receiver) = oneshot::channel();
        self.send(CoreThreadCommand::FetchModel(request, sender))
            .await;
        receiver.await.map_err(CoreError::Shutdown)
    }

    pub(crate) async fn run_inference(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, CoreError> {
        let (sender, receiver) = oneshot::channel();
        self.send(CoreThreadCommand::RunInference(request, sender))
            .await;
        receiver.await.map_err(CoreError::Shutdown)
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
