use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    path::PathBuf,
    str::FromStr,
    sync::mpsc,
    thread::JoinHandle,
};

use atoma_types::{Digest, Request, Response};
use futures::stream::FuturesUnordered;
use thiserror::Error;
use tokio::sync::oneshot::{self, error::RecvError};
use tracing::{debug, error, info, warn};

use crate::{
    apis::ApiError,
    models::{
        candle::{
            falcon::FalconModel, llama::LlamaModel, mamba::MambaModel, mistral::MistralModel,
            mixtral::MixtralModel, phi3::Phi3Model, quantized::QuantizedModel, qwen::QwenModel,
            stable_diffusion::StableDiffusion,
        },
        config::{ModelConfig, ModelsConfig},
        types::ModelType,
        ModelError, ModelId, ModelTrait,
    },
};

pub struct ModelThreadCommand {
    pub(crate) batched_requests: Vec<Request>,
    pub(crate) sender: oneshot::Sender<Vec<Response>>,
}

#[derive(Debug, Error)]
pub enum ModelThreadError {
    #[error("Model thread shutdown: `{0}`")]
    ApiError(#[from] ApiError),
    #[error("Model thread shutdown: `{0}`")]
    ModelError(#[from] ModelError),
    #[error("Core thread shutdown: `{0}`")]
    Shutdown(RecvError),
    #[error("Serde error: `{0}`")]
    SerdeError(#[from] serde_json::Error),
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
    pub fn run(mut self) -> Result<(), ModelThreadError> {
        debug!("Start Model thread");

        while let Ok(command) = self.receiver.recv() {
            let ModelThreadCommand {
                batched_requests,
                sender,
            } = command;
            // let request_id = request.id();
            // let sampled_node_index = request.sampled_node_index();
            // let num_sampled_nodes = request.num_sampled_nodes();
            let batched_inputs = batched_requests
                .iter()
                .map(|r| {
                    let request_id = r.id();
                    let params = r.params();
                    M::Input::try_from((hex::encode(&request_id), params))
                })
                .collect();
            let model_output = self.model.run(model_input)?;
            let output = model_output
                .iter()
                .map(|o| serde_json::to_value(o))
                .collect::<Result<Vec<_>, _>>()?;
            let response = Response::new(request_id, sampled_node_index, num_sampled_nodes, output);
            sender.send(response).ok();
        }

        Ok(())
    }
}

pub struct ModelThreadDispatcher {
    pub(crate) model_senders: HashMap<ModelId, mpsc::Sender<ModelThreadCommand>>,
    pub(crate) responses: FuturesUnordered<oneshot::Receiver<Vec<Response>>>,
}

impl ModelThreadDispatcher {
    pub(crate) fn start(
        config: ModelsConfig,
        stream_tx: tokio::sync::mpsc::Sender<(Digest, String)>,
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
                model_receiver,
                stream_tx.clone(),
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

    fn send(&self, command: ModelThreadCommand) {
        let batched_requests = command.batched_requests;
        let mut model_batched_requests =
            HashMap::<ModelId, Vec<Request>>::with_capacity(batched_requests.len());
        for request in batched_requests.iter() {
            model_batched_requests
                .entry(request.model())
                .or_insert_with(Vec::new)
                .push(request.clone());
        }

        for (model_id, requests) in model_batched_requests {
            info!("sending new batch of requests for model: {:?}", model_id);

            let sender = self
                .model_senders
                .get(&model_id)
                .expect("Failed to get model thread, this should not happen !");

            let model_command = ModelThreadCommand {
                batched_requests: requests,
                sender: command.sender,
            };

            if let Err(e) = sender.send(model_command) {
                warn!("Could not send command to model core, it might be shutting down: {e}");
            }
        }
    }
}

impl ModelThreadDispatcher {
    pub(crate) fn run_json_inference(
        &self,
        (batched_requests, sender): (Vec<Request>, oneshot::Sender<Vec<Response>>),
    ) {
        self.send(ModelThreadCommand {
            batched_requests,
            sender,
        });
    }

    pub(crate) fn run_subscriber_inference(&self, batched_requests: Vec<Request>) {
        let mut model_batched_requests = HashMap::with_capacity(batched_requests.len());

        for request in batched_requests.iter() {
            model_batched_requests
                .entry(request.model())
                .or_insert_with(Vec::new)
                .push(request.clone());
        }

        for (model_id, requests) in model_batched_requests {
            let (sender, receiver) = oneshot::channel();
            self.send(ModelThreadCommand {
                batched_requests: requests,
                sender,
            });
            self.responses.push(receiver);
        }
    }
}

pub(crate) fn dispatch_model_thread(
    api_key: String,
    cache_dir: PathBuf,
    model_name: String,
    model_type: ModelType,
    model_config: ModelConfig,
    model_receiver: mpsc::Receiver<ModelThreadCommand>,
    stream_tx: tokio::sync::mpsc::Sender<(Digest, String)>,
) -> JoinHandle<Result<(), ModelThreadError>> {
    match model_type {
        ModelType::Falcon7b | ModelType::Falcon40b | ModelType::Falcon180b => {
            spawn_model_thread::<FalconModel>(
                model_name,
                api_key.clone(),
                cache_dir.clone(),
                model_config,
                model_receiver,
                stream_tx,
            )
        }
        ModelType::LlamaV1
        | ModelType::LlamaV2
        | ModelType::LlamaTinyLlama1_1BChat
        | ModelType::LlamaSolar10_7B
        | ModelType::Llama3_8b
        | ModelType::Llama3Instruct8b
        | ModelType::Llama3_70b => spawn_model_thread::<LlamaModel>(
            model_name,
            api_key,
            cache_dir,
            model_config,
            model_receiver,
            stream_tx,
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
            model_receiver,
            stream_tx,
        ),
        ModelType::Mistral7bV01
        | ModelType::Mistral7bV02
        | ModelType::Mistral7bInstructV01
        | ModelType::Mistral7bInstructV02 => spawn_model_thread::<MistralModel>(
            model_name,
            api_key,
            cache_dir,
            model_config,
            model_receiver,
            stream_tx,
        ),
        ModelType::Mixtral8x7b => spawn_model_thread::<MixtralModel>(
            model_name,
            api_key,
            cache_dir,
            model_config,
            model_receiver,
            stream_tx,
        ),
        ModelType::Phi3Mini => spawn_model_thread::<Phi3Model>(
            model_name,
            api_key,
            cache_dir,
            model_config,
            model_receiver,
            stream_tx,
        ),
        ModelType::StableDiffusionV1_5
        | ModelType::StableDiffusionV2_1
        | ModelType::StableDiffusionTurbo
        | ModelType::StableDiffusionXl => spawn_model_thread::<StableDiffusion>(
            model_name,
            api_key,
            cache_dir,
            model_config,
            model_receiver,
            stream_tx,
        ),
        ModelType::QuantizedLlamaV2_7b
        | ModelType::QuantizedLlamaV2_13b
        | ModelType::QuantizedLlamaV2_70b
        | ModelType::QuantizedLlamaV2_7bChat
        | ModelType::QuantizedLlamaV2_13bChat
        | ModelType::QuantizedLlamaV2_70bChat
        | ModelType::QuantizedLlama7b
        | ModelType::QuantizedLlama13b
        | ModelType::QuantizedLlama34b
        | ModelType::QuantizedLeo7b
        | ModelType::QuantizedLeo13b
        | ModelType::QuantizedMistral7b
        | ModelType::QuantizedMistral7bInstruct
        | ModelType::QuantizedMistral7bInstructV02
        | ModelType::QuantizedZephyr7bAlpha
        | ModelType::QuantizedZephyr7bBeta
        | ModelType::QuantizedOpenChat35
        | ModelType::QuantizedStarling7bAlpha
        | ModelType::QuantizedMixtral
        | ModelType::QuantizedMixtralInstruct
        | ModelType::QuantizedL8b => spawn_model_thread::<QuantizedModel>(
            model_name,
            api_key,
            cache_dir,
            model_config,
            model_receiver,
            stream_tx,
        ),
        ModelType::QwenW0_5b
        | ModelType::QwenW1_8b
        | ModelType::QwenW4b
        | ModelType::QwenW7b
        | ModelType::QwenW14b
        | ModelType::QwenW72b
        | ModelType::QwenMoeA27b => spawn_model_thread::<QwenModel>(
            model_name,
            api_key,
            cache_dir,
            model_config,
            model_receiver,
            stream_tx,
        ),
    }
}

pub(crate) fn spawn_model_thread<M>(
    model_name: String,
    api_key: String,
    cache_dir: PathBuf,
    model_config: ModelConfig,
    model_receiver: mpsc::Receiver<ModelThreadCommand>,
    stream_tx: tokio::sync::mpsc::Sender<(Digest, String)>,
) -> JoinHandle<Result<(), ModelThreadError>>
where
    M: ModelTrait + Send + 'static,
{
    std::thread::spawn(move || {
        info!("Fetching files for model: {model_name}");
        let load_data = M::fetch(api_key, cache_dir, model_config)?;

        let model = M::load(load_data, stream_tx)?;
        let model_thread = ModelThread {
            model,
            receiver: model_receiver,
        };

        if let Err(e) = model_thread.run() {
            error!("Model thread error: {e}");
            if !matches!(e, ModelThreadError::Shutdown(_)) {
                panic!("Fatal error occurred: {e}");
            }
        }

        Ok(())
    })
}
