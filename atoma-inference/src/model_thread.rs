use std::{collections::HashMap, path::PathBuf, str::FromStr, sync::mpsc, thread::JoinHandle};

use atoma_types::{ModelThreadError, ModelType, Request, Response};
use ed25519_consensus::VerificationKey as PublicKey;
use futures::stream::FuturesUnordered;
use tokio::sync::oneshot;
use tracing::{debug, error, info, warn};

use crate::models::{
    candle::{
        falcon::FalconModel, llama::LlamaModel, mamba::MambaModel, mistral::MistralModel,
        mixtral::MixtralModel, quantized::QuantizedModel, stable_diffusion::StableDiffusion,
    },
    config::{ModelConfig, ModelsConfig},
    ModelId, ModelTrait,
};

pub struct ModelThreadCommand {
    pub(crate) request: Request,
    pub(crate) sender: oneshot::Sender<Response>,
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
            let ModelThreadCommand { request, sender } = command;

            // if !request.is_node_authorized(&public_key) {
            //     error!("Current node, with verification key = {:?} is not authorized to run request with id = {}", public_key, request.request_id());
            //     continue;
            // }
            let request_id = request.id();
            let sampled_nodes = request.sampled_nodes();
            let body = request.body();
            let model_input = serde_json::from_value(body)?;
            let model_output = self.model.run(model_input)?;
            let output = serde_json::to_value(model_output)?;
            let response = Response::new(request_id, sampled_nodes, output);
            sender.send(response).ok();
        }

        Ok(())
    }
}

pub struct ModelThreadDispatcher {
    pub(crate) model_senders: HashMap<ModelId, mpsc::Sender<ModelThreadCommand>>,
    pub(crate) responses: FuturesUnordered<oneshot::Receiver<Response>>,
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

        let model_dispatcher = ModelThreadDispatcher {
            model_senders,
            responses: FuturesUnordered::new(),
        };

        Ok((model_dispatcher, handles))
    }

    fn send(&self, command: ModelThreadCommand) {
        let model_id = command.request.model();

        info!("sending new request with model_id: {model_id}");

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
    pub(crate) fn run_json_inference(
        &self,
        (request, sender): (Request, oneshot::Sender<Response>),
    ) {
        self.send(ModelThreadCommand { request, sender });
    }

    pub(crate) fn run_subscriber_inference(&self, request: Request) {
        let (sender, receiver) = oneshot::channel();
        self.send(ModelThreadCommand { request, sender });
        self.responses.push(receiver);
    }
}

pub(crate) fn dispatch_model_thread(
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
        | ModelType::LlamaSolar10_7B
        | ModelType::Llama3_8b
        | ModelType::Llama3Instruct8b
        | ModelType::Llama3_70b => spawn_model_thread::<LlamaModel>(
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
        ModelType::Mistral7bV01
        | ModelType::Mistral7bV02
        | ModelType::Mistral7bInstructV01
        | ModelType::Mistral7bInstructV02 => spawn_model_thread::<MistralModel>(
            model_name,
            api_key,
            cache_dir,
            model_config,
            public_key,
            model_receiver,
        ),
        ModelType::Mixtral8x7b => spawn_model_thread::<MixtralModel>(
            model_name,
            api_key,
            cache_dir,
            model_config,
            public_key,
            model_receiver,
        ),
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
            public_key,
            model_receiver,
        ),
    }
}

pub(crate) fn spawn_model_thread<M>(
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
