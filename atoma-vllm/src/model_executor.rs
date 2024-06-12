use std::time::Duration;

use async_trait::async_trait;
use futures::stream::FuturesUnordered;
use thiserror::Error;
use tokio::{
    sync::{
        mpsc,
        oneshot::{self, error::RecvError},
    },
    task::JoinHandle,
};
use tracing::{error, info, instrument, trace};

use crate::{
    sampling_params::SamplingParams,
    sequence::{ExecuteModelRequest, SequenceGroupOutput},
};

/// Duration to wait, if there are no scheduled requests to be processed
const AWAIT_DURATION_EMPTY_REQUESTS: Duration = Duration::from_millis(500);

#[async_trait]
/// `ModelLoader` trait - interface for fetching
/// and loading a LLM model weights
pub trait ModelLoader {
    type Error;

    async fn fetch();
    async fn load() -> Result<Self, Self::Error>
    where
        Self: Sized;
}

#[async_trait]
/// `ModelExecutor` trait - interface for running AI inference
/// from a LLM
pub trait ModelExecutor: ModelLoader {
    type Input;
    type Logits;
    type Output;

    async fn forward(&mut self, input: Self::Input) -> Result<Self::Logits, Self::Error>;
    async fn sample(
        &mut self,
        input: Self::Logits,
        sampling_params: SamplingParams,
    ) -> Result<Self::Output, Self::Error>;
}

/// `ModelThreadCommand` - encapsulates a `ValidGenerateRequest`
/// to run AI inference on, together with a oneshot::Sender
/// channel to communicate the AI generated output with the
/// main task
pub struct ModelThreadCommand {
    request: ExecuteModelRequest,
    sender: oneshot::Sender<SequenceGroupOutput>,
}

/// `ModelThread` - encapsulates the logic
/// to run a model thread/task in the background.
/// It receives new coming requests and start processing
/// AI inference on it.
pub struct ModelThread<M: ModelExecutor> {
    model: M,
    receiver: mpsc::UnboundedReceiver<ModelThreadCommand>,
}

impl<M> ModelThread<M>
where
    M: ModelExecutor + Send + Sync,
{
    /// Main loop, it listenings to incoming requests, in the form `ModelThreadCommand`.
    /// When a new request is received, it starts a new inference loop for the encapsulated
    /// AI model `M`. Once the AI generated output is ready, it sends it back using the corresponding
    /// `oneshot` `Sender` encapsulated in the `ModelThreadCommand`.
    #[instrument(skip(self))]
    pub fn run(mut self) -> Result<(), ModelThreadError> {
        info!("Start Model thread");

        while let Some(command) = self.receiver.blocking_recv() {
            let ModelThreadCommand { request, sender } = command;
            if request.is_empty() {
                // await an instant before sending the empty response
                // TODO: Check if this makes sense, or we can improve this logic
                std::thread::sleep(AWAIT_DURATION_EMPTY_REQUESTS);
                sender
                    .send(SequenceGroupOutput::empty())
                    .map_err(|_| ModelThreadError::SendError)?;
            }
            // sender.send(response).ok();
        }

        Ok(())
    }
}

/// `ModelThreadDispatcher` - Responsible for managing incoming requests to
/// different the background LLM inference task
pub struct ModelThreadDispatcher {
    /// Mapping from each model id to the remove `Sender`'s `ModelThreadCommand`
    pub sender: mpsc::UnboundedSender<ModelThreadCommand>,
    /// A `FuturesUnordered` containing each generated `Response`'s oneshot receiver.
    /// It should yield everytime a new AI inference output is generated.
    pub responses: FuturesUnordered<oneshot::Receiver<SequenceGroupOutput>>,
    /// The model's thread join handle
    pub join_handle: JoinHandle<Result<(), ModelThreadError>>,
}

impl ModelThreadDispatcher {
    /// Starts a new instance of a `ModelThreadDispatcher`. It further spawns a new thread model
    /// that continuously listens to incoming AI inference requests, and processes these.
    #[instrument(skip(model))]
    pub(crate) fn start<M>(model: M) -> Result<Self, ModelThreadError>
    where
        M: ModelExecutor + Send + Sync + 'static,
    {
        let (sender, receiver) = mpsc::unbounded_channel();

        let join_handle = tokio::task::spawn_blocking(|| {
            let model_thread = ModelThread { model, receiver };
            if let Err(e) = model_thread.run() {
                error!("Model thread error: {e}");
                if !matches!(e, ModelThreadError::Shutdown(_)) {
                    panic!("Fatal error occurred: {e}");
                }
            }

            Ok(())
        });

        let model_dispatcher = ModelThreadDispatcher {
            sender,
            responses: FuturesUnordered::new(),
            join_handle,
        };

        Ok(model_dispatcher)
    }

    /// Sends a `ModelThreadCommand` instance into the corresponding
    /// `Model`'s thread, to be processed by the `Model` itself.
    #[instrument(skip(self))]
    pub fn send(&self, request: ExecuteModelRequest) {
        trace!("Sending new `ExecuteModelRequest` to model executor task");

        let (sender, receiver) = oneshot::channel();
        let command = ModelThreadCommand { request, sender };

        if let Err(e) = self.sender.send(command) {
            error!("Could not send command to model core, it might be shutting down: {e}");
        }

        self.responses.push(receiver);
    }
}

#[derive(Debug, Error)]
pub enum ModelThreadError {
    #[error("Core thread shutdown: `{0}`")]
    Shutdown(RecvError),
    #[error("Send error")]
    SendError,
}
