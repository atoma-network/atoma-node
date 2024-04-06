use std::{collections::HashMap, fmt::Debug, path::PathBuf, sync::mpsc, thread::JoinHandle};

use ed25519_consensus::VerificationKey as PublicKey;
use futures::stream::FuturesUnordered;
use thiserror::Error;
use tokio::sync::oneshot::{self, error::RecvError};
use tracing::{debug, error, info, warn};

use crate::{
    apis::ApiError,
    models::{
        config::{ModelConfig, ModelsConfig},
        ModelError, ModelId, ModelTrait, Request, Response,
    },
};

pub struct ModelThreadCommand<Req, Resp>
where
    Req: Request,
    Resp: Response,
{
    request: Req,
    response_sender: oneshot::Sender<Resp>,
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

pub struct ModelThread<M, Req, Resp>
where
    M: ModelTrait,
    Req: Request,
    Resp: Response,
{
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
            let ModelThreadCommand {
                request,
                response_sender,
            } = command;

            if !request.is_node_authorized(&public_key) {
                error!("Current node, with verification key = {:?} is not authorized to run request with id = {}", public_key, request.request_id());
                continue;
            }

            let model_input = request.into_model_input();
            let model_output = self.model.run(model_input)?;
            let response = Response::from_model_output(model_output);
            response_sender.send(response).ok();
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
    pub(crate) fn start<M>(
        config: ModelsConfig,
        public_key: PublicKey,
    ) -> Result<(Self, Vec<ModelThreadHandle<Req, Resp>>), ModelThreadError>
    where
        M: ModelTrait<Input = Req::ModelInput, Output = Resp::ModelOutput> + Send + 'static,
    {
        let mut handles = Vec::new();
        let mut model_senders = HashMap::new();

        let api_key = config.api_key();
        let cache_dir = config.cache_dir();

        for model_config in config.models() {
            info!("Spawning new thread for model: {}", model_config.model_id());

            let (model_sender, model_receiver) = mpsc::channel::<ModelThreadCommand<_, _>>();
            let model_name = model_config.model_id().clone();
            model_senders.insert(model_name.clone(), model_sender.clone());

            let join_handle = Self::start_model_thread::<M>(
                model_name,
                api_key.clone(),
                cache_dir.clone(),
                model_config,
                public_key,
                model_receiver,
            );

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

    fn start_model_thread<M>(
        model_name: String,
        api_key: String,
        cache_dir: PathBuf,
        model_config: ModelConfig,
        public_key: PublicKey,
        model_receiver: mpsc::Receiver<ModelThreadCommand<Req, Resp>>,
    ) -> JoinHandle<Result<(), ModelThreadError>>
    where
        M: ModelTrait<Input = Req::ModelInput, Output = Resp::ModelOutput> + Send + 'static,
    {
        std::thread::spawn(move || {
            info!("Fetching files for model: {model_name}");
            let load_data = M::fetch(api_key, cache_dir, model_config)?;

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
        })
    }

    fn send(&self, command: ModelThreadCommand<Req, Resp>) {
        let request = command.request.clone();
        let model_id = request.requested_model();

        info!("model_id {model_id}");

        let sender = self
            .model_senders
            .get(&model_id)
            .expect("Failed to get model thread, this should not happen !");

        if let Err(e) = sender.send(command) {
            warn!("Could not send command to model core, it might be shutting down: {e}");
        }
    }
}

impl<Req, Resp> ModelThreadDispatcher<Req, Resp>
where
    Req: Clone + Debug + Request,
    Resp: Debug + Response,
{
    pub(crate) fn run_inference(&self, request: Req) {
        let (sender, receiver) = oneshot::channel();
        self.send(ModelThreadCommand {
            request,
            response_sender: sender,
        });
        self.responses.push(receiver);
    }
}
