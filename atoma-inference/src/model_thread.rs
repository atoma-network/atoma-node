use std::{
    collections::HashMap,
    sync::{mpsc, Arc},
};

use ed25519_consensus::VerificationKey as PublicKey;
use futures::stream::FuturesUnordered;
use thiserror::Error;
use tokio::sync::oneshot::{self, error::RecvError};
use tracing::{debug, error, warn};

use crate::{
    apis::{ApiError, ApiTrait},
    models::{config::ModelConfig, ModelError, ModelId, ModelTrait, Request, Response},
};

pub struct ModelThreadCommand<Req, Resp>(Req, oneshot::Sender<Resp>)
where
    Req: Request,
    Resp: Response;

#[derive(Debug, Error)]
pub enum ModelThreadError {
    #[error("Model thread shutdown: `{0}`")]
    ApiError(ApiError),
    #[error("Model thread shutdown: `{0}`")]
    ModelError(ModelError),
    #[error("Core thread shutdown: `{0}`")]
    Shutdown(RecvError),
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

pub struct ModelThreadHandle<Req, Resp>
where
    Req: Request,
    Resp: Response,
{
    sender: mpsc::Sender<ModelThreadCommand<Req, Resp>>,
    join_handle: std::thread::JoinHandle<Result<(), ModelThreadError>>,
}

impl<Req, Resp> ModelThreadHandle<Req, Resp>
where
    Req: Request,
    Resp: Response,
{
    pub fn stop(self) {
        drop(self.sender);
        self.join_handle.join().ok();
    }
}

pub struct ModelThread<M: ModelTrait, Req: Request, Resp: Response> {
    model: M,
    receiver: mpsc::Receiver<ModelThreadCommand<Req, Resp>>,
}

impl<M, Req, Resp> ModelThread<M, Req, Resp>
where
    M: ModelTrait<Input = Req::ModelInput, Output = Resp::ModelOutput>,
    Req: Request,
    Resp: Response,
{
    pub fn run(mut self, public_key: PublicKey) -> Result<(), ModelThreadError> {
        debug!("Start Model thread");

        while let Ok(command) = self.receiver.recv() {
            let ModelThreadCommand(request, sender) = command;

            if !request.is_node_authorized(&public_key) {
                error!("Current node, with verification key = {:?} is not authorized to run request with id = {}", public_key, request.request_id());
                continue;
            }

            let model_input = request.into_model_input();
            let model_output = self
                .model
                .run(model_input)
                .map_err(ModelThreadError::ModelError)?;
            let response = Resp::from_model_output(model_output);
            sender.send(response).ok();
        }

        Ok(())
    }
}

pub struct ModelThreadDispatcher<Req, Resp>
where
    Req: Request,
    Resp: Response,
{
    model_senders: HashMap<ModelId, mpsc::Sender<ModelThreadCommand<Req, Resp>>>,
    pub(crate) responses: FuturesUnordered<oneshot::Receiver<Resp>>,
}

impl<Req, Resp> ModelThreadDispatcher<Req, Resp>
where
    Req: Clone + Request,
    Resp: Response,
{
    pub(crate) fn start<M, F>(
        config: ModelConfig,
        public_key: PublicKey,
    ) -> Result<(Self, Vec<ModelThreadHandle<Req, Resp>>), ModelThreadError>
    where
        F: ApiTrait + Send + Sync + 'static,
        M: ModelTrait<Input = Req::ModelInput, Output = Resp::ModelOutput> + Send + 'static,
    {
        let model_ids = config.model_ids();
        let api_key = config.api_key();
        let storage_path = config.storage_path();
        let api = Arc::new(F::create(api_key, storage_path)?);

        let mut handles = Vec::with_capacity(model_ids.len());
        let mut model_senders = HashMap::with_capacity(model_ids.len());

        for (model_id, precision, revision) in model_ids {
            let api = api.clone();

            let (model_sender, model_receiver) = mpsc::channel::<ModelThreadCommand<_, _>>();
            let model_name = model_id.clone();

            let join_handle = std::thread::spawn(move || {
                let filenames = api.fetch(model_name, revision)?;

                let model = M::load(filenames, precision)?;
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
            model_senders.insert(model_id, model_sender);
        }

        let model_dispatcher = ModelThreadDispatcher {
            model_senders,
            responses: FuturesUnordered::new(),
        };

        Ok((model_dispatcher, handles))
    }

    fn send(&self, command: ModelThreadCommand<Req, Resp>) {
        let request = command.0.clone();
        let model_type = request.requested_model();

        let sender = self
            .model_senders
            .get(&model_type)
            .expect("Failed to get model thread, this should not happen !");

        if let Err(e) = sender.send(command) {
            warn!("Could not send command to model core, it might be shutting down: {e}");
        }
    }
}

impl<T, U> ModelThreadDispatcher<T, U>
where
    T: Clone + Request,
    U: Response,
{
    pub(crate) fn run_inference(&self, request: T) {
        let (sender, receiver) = oneshot::channel();
        self.send(ModelThreadCommand(request, sender));
        self.responses.push(receiver);
    }
}
