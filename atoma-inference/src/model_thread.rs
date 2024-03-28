use std::{collections::HashMap, sync::mpsc};

use candle_nn::VarBuilder;
use ed25519_consensus::VerificationKey as PublicKey;
use futures::stream::FuturesUnordered;
use thiserror::Error;
use tokio::sync::oneshot::{self, error::RecvError};
use tracing::{debug, error, warn};

use crate::{
    models::{ModelApi, ModelError, ModelSpecs, ModelType},
    types::{InferenceRequest, InferenceResponse},
};

pub struct ModelThreadCommand(InferenceRequest, oneshot::Sender<InferenceResponse>);

#[derive(Debug, Error)]
pub enum ModelThreadError {
    #[error("Model thread shutdown: `{0}`")]
    ModelError(ModelError),
    #[error("Core thread shutdown: `{0}`")]
    Shutdown(RecvError),
}

pub struct ModelThreadHandle {
    sender: mpsc::Sender<ModelThreadCommand>,
    join_handle: std::thread::JoinHandle<()>,
}

impl ModelThreadHandle {
    pub fn stop(self) {
        drop(self.sender);
        self.join_handle.join().ok();
    }
}

pub struct ModelThread<T: ModelApi> {
    model: T,
    receiver: mpsc::Receiver<ModelThreadCommand>,
}

impl<T> ModelThread<T>
where
    T: ModelApi,
{
    pub fn run(self, public_key: PublicKey) -> Result<(), ModelThreadError> {
        debug!("Start Model thread");

        while let Ok(command) = self.receiver.recv() {
            let ModelThreadCommand(request, sender) = command;

            let InferenceRequest {
                prompt,
                max_tokens,
                temperature,
                random_seed,
                repeat_last_n,
                repeat_penalty,
                top_p,
                sampled_nodes,
                ..
            } = request;
            if !sampled_nodes.contains(&public_key) {
                error!("Current node, with verification key = {:?} was not sampled from {sampled_nodes:?}", public_key);
                continue;
            }
            let response = self
                .model
                .run(
                    prompt,
                    max_tokens,
                    random_seed,
                    repeat_last_n,
                    repeat_penalty,
                    temperature.unwrap_or_default(),
                    top_p.unwrap_or_default(),
                )
                .map_err(ModelThreadError::ModelError)?;
            let response = InferenceResponse { response };
            sender.send(response).ok();
        }

        Ok(())
    }
}

pub struct ModelThreadDispatcher {
    model_senders: HashMap<ModelType, mpsc::Sender<ModelThreadCommand>>,
    pub(crate) responses: FuturesUnordered<oneshot::Receiver<InferenceResponse>>,
}

impl ModelThreadDispatcher {
    pub(crate) fn start<T>(
        models: Vec<(ModelType, ModelSpecs, VarBuilder)>,
        public_key: PublicKey,
    ) -> Result<(Self, Vec<ModelThreadHandle>), ModelThreadError>
    where
        T: ModelApi + Send + 'static,
    {
        let mut handles = Vec::with_capacity(models.len());
        let mut model_senders = HashMap::with_capacity(models.len());

        for (model_type, model_specs, var_builder) in models {
            let (model_sender, model_receiver) = mpsc::channel::<ModelThreadCommand>();
            let model = T::load(model_specs, var_builder); // TODO: for now this piece of code cannot be shared among threads safely
            let model_thread = ModelThread {
                model,
                receiver: model_receiver,
            };
            let join_handle = std::thread::spawn(move || {
                if let Err(e) = model_thread.run(public_key) {
                    error!("Model thread error: {e}");
                    if !matches!(e, ModelThreadError::Shutdown(_)) {
                        panic!("Fatal error occurred: {e}");
                    }
                }
            });
            handles.push(ModelThreadHandle {
                join_handle,
                sender: model_sender.clone(),
            });
            model_senders.insert(model_type, model_sender);
        }

        let model_dispatcher = ModelThreadDispatcher {
            model_senders,
            responses: FuturesUnordered::new(),
        };

        Ok((model_dispatcher, handles))
    }

    fn send(&self, command: ModelThreadCommand) {
        let request = command.0.clone();
        let model_type = request.model;

        let sender = self
            .model_senders
            .get(&model_type)
            .expect("Failed to get model thread, this should not happen !");

        if let Err(e) = sender.send(command) {
            warn!("Could not send command to model core, it might be shutting down: {e}");
        }
    }
}

impl ModelThreadDispatcher {
    pub(crate) fn run_inference(&self, request: InferenceRequest) {
        let (sender, receiver) = oneshot::channel();
        self.send(ModelThreadCommand(request, sender));
        self.responses.push(receiver);
    }
}