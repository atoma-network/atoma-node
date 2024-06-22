use std::{collections::HashMap, path::PathBuf};

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
    sequence::{ExecuteModelRequest, LogProb, SequenceGroupOutput, SequenceOutput},
    validation::{NextTokenChooserParameters, StoppingCriteriaParameters},
};

#[async_trait]
/// `ModelLoader` trait - interface for fetching
/// and loading a LLM model weights. Also has a method
/// providing the `eos_token_id` for the current model's
/// tokenizer.
pub trait ModelLoader {
    type FilePaths;

    async fn fetch(api_key: String, model_name: String, revision: String) -> Result<Self::FilePaths, ModelLoaderError>;
    async fn load(file_paths: Self::FilePaths) -> Result<Self, ModelLoaderError>
    where
        Self: Sized;
    fn cache_dir(&self) -> PathBuf;
    fn eos_token_id(&self) -> Option<u32>;
    fn head_size(&self) -> usize;
    fn num_attention_heads(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn sliding_window(&self) -> Option<usize>;
}

/// `ModelExecutor` trait - interface for running AI inference
/// from a LLM
pub trait ModelExecutor: ModelLoader {
    type Input: From<ExecuteModelRequest>;
    type Logits: Clone;
    type Output: Into<u32>;

    fn forward(&mut self, input: Self::Input) -> Result<Self::Logits, ModelExecutorError>;
    fn sample(
        &mut self,
        input: Self::Logits,
        next_token_params: NextTokenChooserParameters,
        stopping_params: StoppingCriteriaParameters,
    ) -> Result<Self::Output, ModelExecutorError>;
}

/// `ModelThreadCommand` - encapsulates a `ValidGenerateRequest`
/// to run AI inference on, together with a oneshot::Sender
/// channel to communicate the AI generated output with the
/// main task
pub struct ModelThreadCommand {
    request: ExecuteModelRequest,
    sender: oneshot::Sender<Vec<SequenceGroupOutput>>,
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

            let sequence_groups_metadata = request.sequence_groups_metadata.clone();
            let next_token_chooser_params: Vec<NextTokenChooserParameters> = request
                .sequence_groups_metadata
                .iter()
                .map(|s| s.next_token_chooser_params.clone())
                .collect();
            let stopping_params: Vec<StoppingCriteriaParameters> = request
                .sequence_groups_metadata
                .iter()
                .map(|s| s.stopping_criteria_params.clone())
                .collect();

            let logits = match self.model.forward(request.into()) {
                Ok(logits) => logits,
                Err(e) => {
                    error!("Failed to run forward pass on model, with error: {e}");
                    return Err(ModelThreadError::ModelExecutorError(e));
                }
            };

            let mut responses = Vec::with_capacity(next_token_chooser_params.len());

            // TODO: should we parallelize this loop, with rayon or within the async runtime ?
            for (next_token_params, (stopping_params, metadata)) in next_token_chooser_params
                .iter()
                .zip(stopping_params.iter().zip(sequence_groups_metadata))
            {
                let mut outputs = HashMap::with_capacity(metadata.sequence_data.len());

                for sequence_id in metadata.sequence_data.keys() {
                    let decode_token = match self.model.sample(
                        logits.clone(),
                        next_token_params.clone(),
                        stopping_params.clone(),
                    ) {
                        Ok(token) => token,
                        Err(e) => {
                            error!("Failed to sample next decoding token, with error: {e}");
                            return Err(ModelThreadError::ModelExecutorError(e));
                        }
                    };

                    let output_token = decode_token.into();

                    outputs.insert(
                        *sequence_id,
                        SequenceOutput {
                            parent_sequence_id: *sequence_id,
                            output_token,
                            logprob: HashMap::from_iter([(
                                output_token,
                                LogProb::new(0.8, None, None),
                            )]), // TODO: replace hardcoded values with logic
                        },
                    );
                }

                // TODO: Check this is the correct logic, once we integrate
                // with model executor
                let response = SequenceGroupOutput {
                    outputs,
                    sampled_token_ids: None,
                    sampled_token_probs: None,
                    logprobs: None,
                    spec_decode_worker_metrics: None,
                };
                responses.push(response);
            }

            sender.send(responses).ok();
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
    pub responses: FuturesUnordered<oneshot::Receiver<Vec<SequenceGroupOutput>>>,
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
    #[error("Model loader error: `{0}`")]
    ModelLoaderError(#[from] ModelLoaderError),
    #[error("Model executor error: `{0}`")]
    ModelExecutorError(#[from] ModelExecutorError),
}

#[derive(Debug, Error)]
pub enum ModelLoaderError {}

#[derive(Debug, Error)]
pub enum ModelExecutorError {}
