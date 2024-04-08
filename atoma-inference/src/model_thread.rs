use std::{
    collections::HashMap, fmt::Debug, path::PathBuf, str::FromStr, sync::mpsc, thread::JoinHandle,
};

use ed25519_consensus::VerificationKey as PublicKey;
use thiserror::Error;
use tokio::sync::oneshot::{self, error::RecvError};
use tracing::{debug, error, info, warn};

use crate::{
    apis::ApiError,
    models::{
        candle::{
            falcon::FalconModel, llama::LlamaModel, mamba::MambaModel,
            stable_diffusion::StableDiffusion,
        },
        config::{ModelConfig, ModelsConfig},
        types::ModelType,
        ModelError, ModelId, ModelTrait,
    },
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

            // if !request.is_node_authorized(&public_key) {
            //     error!("Current node, with verification key = {:?} is not authorized to run request with id = {}", public_key, request.request_id());
            //     continue;
            // }
            let model_input = serde_json::from_value(request)?;
            let model_output = self.model.run(model_input)?;
            let response = serde_json::to_value(model_output)?;
            response_sender.send(response).ok();
        }

        Ok(())
    }
}

pub struct ModelThreadDispatcher {
    model_senders: HashMap<ModelId, mpsc::Sender<ModelThreadCommand>>,
}

impl ModelThreadDispatcher {
    pub(crate) fn start(
        config: ModelsConfig,
        public_key: PublicKey,
    ) -> Result<(Self, Vec<ModelThreadHandle>), ModelThreadError> {
        let mut handles = Vec::new();
        let mut model_senders = HashMap::new();

        let api_key = config.api_key();
        let cache_dir = config.cache_dir();

        for model_config in config.models() {
            info!("Spawning new thread for model: {}", model_config.model_id());

            let model_name = model_config.model_id().clone();
            let model_type = ModelType::from_str(&model_name)?;

            let (model_sender, model_receiver) = mpsc::channel::<ModelThreadCommand>();
            model_senders.insert(model_name.clone(), model_sender.clone());

            let join_handle = dispatch_model_thread(
                api_key.clone(),
                cache_dir.clone(),
                model_name,
                model_type,
                model_config,
                public_key,
                model_receiver,
            );

            handles.push(ModelThreadHandle {
                join_handle,
                sender: model_sender.clone(),
            });
        }

        let model_dispatcher = ModelThreadDispatcher { model_senders };

        Ok((model_dispatcher, handles))
    }

    fn send(&self, command: ModelThreadCommand) {
        let request = command.request.clone();
        let model_id = if let Some(model_id) = request.get("model") {
            model_id.as_str().unwrap().to_string()
        } else {
            error!("Request malformed: Missing 'model' from request");
            return;
        };

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

impl ModelThreadDispatcher {
    pub(crate) fn run_inference(
        &self,
        (request, sender): (serde_json::Value, oneshot::Sender<serde_json::Value>),
    ) {
        self.send(ModelThreadCommand {
            request,
            response_sender: sender,
        });
    }
}

fn dispatch_model_thread(
    api_key: String,
    cache_dir: PathBuf,
    model_name: String,
    model_type: ModelType,
    model_config: ModelConfig,
    public_key: PublicKey,
    model_receiver: mpsc::Receiver<ModelThreadCommand>,
) -> JoinHandle<Result<(), ModelThreadError>> {
    match model_type {
        ModelType::Falcon7b | ModelType::Falcon40b | ModelType::Falcon180b => {
            spawn_model_thread::<FalconModel>(
                model_name,
                api_key.clone(),
                cache_dir.clone(),
                model_config,
                public_key,
                model_receiver,
            )
        }
        ModelType::LlamaV1
        | ModelType::LlamaV2
        | ModelType::LlamaTinyLlama1_1BChat
        | ModelType::LlamaSolar10_7B => spawn_model_thread::<LlamaModel>(
            model_name,
            api_key,
            cache_dir,
            model_config,
            public_key,
            model_receiver,
        ),
        ModelType::Mamba130m
        | ModelType::Mamba370m
        | ModelType::Mamba790m
        | ModelType::Mamba1_4b
        | ModelType::Mamba2_8b => spawn_model_thread::<MambaModel>(
            model_name,
            api_key,
            cache_dir,
            model_config,
            public_key,
            model_receiver,
        ),
        ModelType::Mistral7b => todo!(),
        ModelType::Mixtral8x7b => todo!(),
        ModelType::StableDiffusionV1_5
        | ModelType::StableDiffusionV2_1
        | ModelType::StableDiffusionTurbo
        | ModelType::StableDiffusionXl => spawn_model_thread::<StableDiffusion>(
            model_name,
            api_key,
            cache_dir,
            model_config,
            public_key,
            model_receiver,
        ),
    }
}

fn spawn_model_thread<M>(
    model_name: String,
    api_key: String,
    cache_dir: PathBuf,
    model_config: ModelConfig,
    public_key: PublicKey,
    model_receiver: mpsc::Receiver<ModelThreadCommand>,
) -> JoinHandle<Result<(), ModelThreadError>>
where
    M: ModelTrait + Send + 'static,
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
