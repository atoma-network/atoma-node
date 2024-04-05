use std::{collections::HashMap, sync::mpsc};

use ed25519_consensus::VerificationKey as PublicKey;
use futures::stream::FuturesUnordered;
use thiserror::Error;
use tokio::sync::oneshot::{self, error::RecvError};
use tracing::{debug, error, info, warn};

use crate::{
    apis::ApiError,
    models::{config::ModelsConfig, ModelError, ModelId, ModelTrait},
};

pub struct ModelThreadCommand {
    request: serde_json::Value,
    response_sender: oneshot::Sender<serde_json::Value>,
}

#[derive(Debug, Error)]
pub enum ModelThreadError {
    #[error("Model thread shutdown: `{0}`")]
    ApiError(ApiError),
    #[error("Model thread shutdown: `{0}`")]
    ModelError(ModelError),
    #[error("Core thread shutdown: `{0}`")]
    Shutdown(RecvError),
    #[error("Serde error: `{0}`")]
    SerdeError(#[from] serde_json::Error),
}

impl From<ModelError> for ModelThreadError {
    fn from(error: ModelError) -> Self {
        Self::ModelError(error)
    }
}

impl From<ApiError> for ModelThreadError {
    fn from(error: ApiError) -> Self {
        Self::ApiError(error)
    }
}

pub struct ModelThreadHandle {
    sender: mpsc::Sender<ModelThreadCommand>,
    join_handle: std::thread::JoinHandle<Result<(), ModelThreadError>>,
}

impl ModelThreadHandle {
    pub fn stop(self) {
        drop(self.sender);
        self.join_handle.join().ok();
    }
}

pub struct ModelThread<M: ModelTrait> {
    model: M,
    receiver: mpsc::Receiver<ModelThreadCommand>,
}

impl<M> ModelThread<M>
where
    M: ModelTrait,
{
    pub fn run(mut self, _public_key: PublicKey) -> Result<(), ModelThreadError> {
        debug!("Start Model thread");

        while let Ok(command) = self.receiver.recv() {
            let ModelThreadCommand {
                request,
                response_sender,
            } = command;

            // TODO: Implement node authorization
            // if !request.is_node_authorized(&public_key) {
            //     error!("Current node, with verification key = {:?} is not authorized to run request with id = {}", public_key, request.request_id());
            //     continue;
            // }

            let model_input = serde_json::from_value(request).unwrap();
            let model_output = self
                .model
                .run(model_input)
                .map_err(ModelThreadError::ModelError)?;
            let response = serde_json::to_value(model_output)?;
            response_sender.send(response).ok();
        }

        Ok(())
    }
}

pub struct ModelThreadDispatcher {
    model_senders: HashMap<ModelId, mpsc::Sender<ModelThreadCommand>>,
    pub(crate) responses: FuturesUnordered<oneshot::Receiver<serde_json::Value>>,
}

impl ModelThreadDispatcher {
    pub(crate) fn start<M>(
        config: ModelsConfig,
        public_key: PublicKey,
    ) -> Result<(Self, Vec<ModelThreadHandle>), ModelThreadError>
    where
        M: ModelTrait, //<Input = Req::ModelInput, Output = Resp::ModelOutput> + Send + 'static,
    {
        let mut handles = Vec::new();
        let mut model_senders = HashMap::new();

        let cache_dir = config.cache_dir();

        for model_config in config.models() {
            info!("Spawning new thread for model: {}", model_config.model_id());

            let model_cache_dir = cache_dir.clone();
            let (model_sender, model_receiver) = mpsc::channel::<ModelThreadCommand>();
            let model_name = model_config.model_id().clone();
            model_senders.insert(model_name.clone(), model_sender.clone());

            let join_handle = std::thread::spawn(move || {
                info!("Fetching files for model: {model_name}");
                let load_data = M::fetch(model_cache_dir, model_config)?;

                let model = M::load(load_data)?;
                let model_thread = ModelThread {
                    model,
                    receiver: model_receiver,
                };

                if let Err(e) = model_thread.run(public_key) {
                    error!("Model thread error: {e}");
                    if !matches!(e, ModelThreadError::Shutdown(_)) {
                        panic!("Fatal error occurred: {e}");
                    }
                }

                Ok(())
            });
            handles.push(ModelThreadHandle {
                join_handle,
                sender: model_sender.clone(),
            });
        }

        let model_dispatcher = ModelThreadDispatcher {
            model_senders,
            responses: FuturesUnordered::new(),
        };

        Ok((model_dispatcher, handles))
    }

    fn send(&self, command: ModelThreadCommand) {
        let request = command.request.clone();
        let model_id = request.get("model").unwrap().as_str().unwrap().to_string();
        println!("model_id {model_id}");

        println!("{:?}", self.model_senders);
        let sender = self
            .model_senders
            .get(&model_id)
            .expect("Failed to get model thread, this should not happen !");

        if let Err(e) = sender.send(command) {
            warn!("Could not send command to model core, it might be shutting down: {e}");
        }
    }
}

impl ModelThreadDispatcher {
    pub(crate) fn run_inference(&self, request: serde_json::Value) {
        let (sender, receiver) = oneshot::channel();
        self.send(ModelThreadCommand {
            request,
            response_sender: sender,
        });
        self.responses.push(receiver);
    }
}
