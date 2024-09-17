use std::{
    collections::HashMap, fmt::Debug, path::PathBuf, str::FromStr, sync::mpsc, thread::JoinHandle,
};

use atoma_types::{AtomaStreamingData, ModelParams, OutputType, Request, Response};
use futures::stream::FuturesUnordered;
use serde::Deserialize;
use thiserror::Error;
use tokio::sync::oneshot::{self, error::RecvError};
use tracing::{debug, error, info, instrument, warn, Span};

#[cfg(feature = "nccl")]
use crate::models::candle::llama_nccl::LlamaNcclModel;
#[cfg(feature = "nccl")]
use crate::models::candle::mixtral_nccl::MixtralNcclModel;

use crate::models::{
    candle::{
        falcon::FalconModel, flux::Flux, llama::LlamaModel, mamba::MambaModel,
        mistral::MistralModel, mixtral::MixtralModel, phi3::Phi3Model, quantized::QuantizedModel,
        qwen::QwenModel, stable_diffusion::StableDiffusion,
    },
    config::{ModelConfig, ModelsConfig},
    types::{LlmOutput, ModelType},
    ModelError, ModelId, ModelTrait,
};

/// `ModelThreadCommand` - Wrapper around an AI inference request to be
///     processed in the corresponding model thread. It also encapsulates
///     a `oneshot` `Sender` that is used to send the `Response` back to
///     the main thread worker.
pub struct ModelThreadCommand {
    /// The `Request` body
    pub(crate) request: Request,
    /// A `oneshot` `Sender` used to send the AI generated `Response`
    pub(crate) sender: oneshot::Sender<Response>,
}

#[derive(Debug, Error)]
pub enum ModelThreadError {
    #[error("Model thread shutdown: `{0}`")]
    ModelError(#[from] ModelError),
    #[error("Core thread shutdown: `{0}`")]
    Shutdown(RecvError),
    #[error("Serde error: `{0}`")]
    SerdeError(#[from] serde_json::Error),
}

/// `ModelThreadHandle` - Encapsulates the corresponding Model thread join handle
///
/// It also contains a `mpsc` `Sender` that can send new `ModelThreadCommand`'s to
/// the corresponding model thread.
pub struct ModelThreadHandle {
    /// A `mpsc` `Sender` channel, responsible to send new `ModelThreadCommand`
    /// to the corresponding `Model`'s thread
    sender: mpsc::Sender<ModelThreadCommand>,
    /// The join handle of the corresponding `Model`'s thread
    join_handle: std::thread::JoinHandle<Result<(), ModelThreadError>>,
}

impl ModelThreadHandle {
    /// Stops the current thread from executing
    pub fn stop(self) {
        drop(self.sender);
        self.join_handle.join().ok();
    }
}

/// `ModelThread` - Wrapper around a `Model`'s thread.
///
/// It contains the corresponding AI model, `M`, together with a
/// `mpsc` `Receiver` channel, listening to incoming `ModelThreadCommand`'s
pub struct ModelThread<M: ModelTrait> {
    model: M,
    receiver: mpsc::Receiver<ModelThreadCommand>,
}

impl<M> ModelThread<M>
where
    M: ModelTrait,
{
    /// Main loop, it listenings to incoming requests, in the form `ModelThreadCommand`.
    /// When a new request is received, it starts a new inference loop for the encapsulated
    /// AI model `M`. Once the AI generated output is ready, it sends it back using the corresponding
    /// `oneshot` `Sender` encapsulated in the `ModelThreadCommand`.
    #[instrument(skip_all)]
    pub fn run(mut self) -> Result<(), ModelThreadError> {
        debug!("Start Model thread");

        while let Ok(command) = self.receiver.recv() {
            let ModelThreadCommand { request, sender } = command;
            let request_id = request.id();
            info!("Received model thread command, with id = {request_id:?}");
            let sampled_node_index = request.sampled_node_index();
            let num_sampled_nodes = request.num_sampled_nodes();
            let params = request.params();
            let output_type = match params {
                ModelParams::Text2ImageModelParams(_) => OutputType::Image,
                ModelParams::Text2TextModelParams(_) => OutputType::Text,
            };
            let output_destination = Deserialize::deserialize(&mut rmp_serde::Deserializer::new(
                request.output_destination().as_slice(),
            ))
            .unwrap();
            let output_id = match output_destination {
                atoma_types::OutputDestination::Firebase { request_id } => request_id,
                atoma_types::OutputDestination::Gateway { gateway_user_id } => gateway_user_id,
                atoma_types::OutputDestination::Ipfs { request_id } => request_id,
            };
            let model_input = M::Input::try_from((output_id, params))?;
            let model_output = self.model.run(model_input)?;
            let time_to_generate = model_output.time_to_generate();
            let num_input_tokens = model_output.num_input_tokens();
            let num_output_tokens = model_output.num_output_tokens();
            let output = serde_json::to_value(model_output)?;
            let output_destination = request.output_destination();
            let response = Response::new(
                request_id,
                sampled_node_index,
                num_sampled_nodes,
                output,
                output_destination,
                output_type,
            );
            sender.send(response).ok();

            // set metrics
            let histogram = metrics::histogram!("atoma-inference-time");
            histogram.record(time_to_generate);
            let histogram = metrics::histogram!("atoma-inference-input-tokens");
            histogram.record(num_input_tokens as f32);
            if let Some(output_tokens) = num_output_tokens {
                let histogram = metrics::histogram!("atoma-inference-output-tokens");
                histogram.record(output_tokens as f32);
            }
        }

        Ok(())
    }
}

/// `ModelThreadDispatcher` - Responsible for managing incoming requests to
/// different AI models (being operated each on its own model threads).
pub struct ModelThreadDispatcher {
    /// Mapping from each model id to the remove `Sender`'s `ModelThreadCommand`
    pub(crate) model_senders: HashMap<ModelId, mpsc::Sender<ModelThreadCommand>>,
    /// A `FuturesUnordered` containing each generated `Response`'s oneshot receiver.
    /// It should yield everyime a new AI inference output is generated.
    pub(crate) responses: FuturesUnordered<oneshot::Receiver<Response>>,
}

impl ModelThreadDispatcher {
    /// Starts a new instance of a `ModelThreadDispatcher`. It further spawns a new thread model
    /// that continuously listens to incoming AI inference requests, and processes these.
    #[instrument(skip_all)]
    pub(crate) fn start(
        config: ModelsConfig,
        stream_tx: tokio::sync::mpsc::Sender<AtomaStreamingData>,
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

    /// Sends a `ModelThreadCommand` instance into the corresponding
    /// `Model`'s thread, to be processed by the `Model` itself.
    #[instrument(skip_all)]
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
    /// Responsible for handling requests from the node's inference JRPC service
    #[instrument(skip_all)]
    pub(crate) fn run_json_inference(
        &self,
        (request, sender): (Request, oneshot::Sender<Response>),
    ) {
        self.send(ModelThreadCommand { request, sender });
    }

    /// Responsible for handling requests from the node's event listener service.
    /// These correspond to requests that are generated through the Atoma's smart contract.
    #[instrument(skip_all)]
    pub(crate) fn run_subscriber_inference(&self, request: Request) {
        let (sender, receiver) = oneshot::channel();
        self.send(ModelThreadCommand { request, sender });
        self.responses.push(receiver);
    }
}

/// Contains logic to start a new model thread. This includes setting
/// HuggingFace's api key, specifying a cache directory for storage of models,
/// the model's name and type together with the corresponding model configuration.
#[instrument(skip_all)]
pub(crate) fn dispatch_model_thread(
    api_key: String,
    cache_dir: PathBuf,
    model_name: String,
    model_type: ModelType,
    model_config: ModelConfig,
    model_receiver: mpsc::Receiver<ModelThreadCommand>,
    stream_tx: tokio::sync::mpsc::Sender<AtomaStreamingData>,
) -> JoinHandle<Result<(), ModelThreadError>> {
    if model_config.device_ids().len() > 1 {
        #[cfg(not(feature = "nccl"))]
        panic!("Multi-GPU is not supported");
        #[cfg(feature = "nccl")]
        match model_type {
            ModelType::LlamaV1
            | ModelType::LlamaV2
            | ModelType::LlamaTinyLlama1_1BChat
            | ModelType::LlamaSolar10_7B
            | ModelType::Llama3_8b
            | ModelType::Llama3Instruct8b
            | ModelType::Llama3_70b
            | ModelType::Llama31_8b
            | ModelType::Llama31Instruct8b
            | ModelType::Llama31_70b
            | ModelType::Llama31Instruct70b
            | ModelType::Llama31_405b
            | ModelType::Llama31Instruct405b => spawn_model_thread::<LlamaNcclModel>(
                model_name,
                api_key,
                cache_dir,
                model_config,
                model_receiver,
                stream_tx,
            ),
            ModelType::Mixtral8x7bV01
            | ModelType::Mixtral8x7bInstructV01
            | ModelType::Mixtral8x22bV01
            | ModelType::Mixtral8x22bInstructV01 => spawn_model_thread::<MixtralNcclModel>(
                model_name,
                api_key,
                cache_dir,
                model_config,
                model_receiver,
                stream_tx,
            ),
            _ => panic!("This model is not supported"),
        }
    } else {
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
            ModelType::FluxSchnell | ModelType::FluxDev => spawn_model_thread::<Flux>(
                model_name,
                api_key.clone(),
                cache_dir.clone(),
                model_config,
                model_receiver,
                stream_tx,
            ),
            ModelType::LlamaV1
            | ModelType::LlamaV2
            | ModelType::LlamaTinyLlama1_1BChat
            | ModelType::LlamaSolar10_7B
            | ModelType::Llama3_8b
            | ModelType::Llama3Instruct8b
            | ModelType::Llama3_70b
            | ModelType::Llama31_8b
            | ModelType::Llama31Instruct8b
            | ModelType::Llama31_70b
            | ModelType::Llama31Instruct70b
            | ModelType::Llama31_405b
            | ModelType::Llama31Instruct405b => spawn_model_thread::<LlamaModel>(
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
            ModelType::Mixtral8x7bV01
            | ModelType::Mixtral8x7bInstructV01
            | ModelType::Mixtral8x22bV01
            | ModelType::Mixtral8x22bInstructV01 => spawn_model_thread::<MixtralModel>(
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
}

/// Spawns a new model thread
#[instrument(skip(api_key, cache_dir, model_config, model_receiver, stream_tx))]
pub(crate) fn spawn_model_thread<M>(
    model_name: String,
    api_key: String,
    cache_dir: PathBuf,
    model_config: ModelConfig,
    model_receiver: mpsc::Receiver<ModelThreadCommand>,
    stream_tx: tokio::sync::mpsc::Sender<AtomaStreamingData>,
) -> JoinHandle<Result<(), ModelThreadError>>
where
    M: ModelTrait + Send + 'static,
{
    let span = Span::current();
    std::thread::spawn(move || {
        let _enter = span.enter();
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
