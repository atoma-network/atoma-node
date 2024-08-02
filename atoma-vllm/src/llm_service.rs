use std::{path::PathBuf, time::Instant};

use crate::{
    config::{CacheConfig, SchedulerConfig},
    llm_engine::{EngineError, GenerateRequestOutput, LlmEngine},
    model_executor::{ModelExecutor, ModelThreadDispatcher, ModelThreadError},
    scheduler::{Scheduler, SchedulerError},
    sequence::{Sequence, SequenceError, SequenceGroup},
    types::GenerateRequest,
    validation::{ValidGenerateRequest, Validation, ValidationError},
};
use candle_core::{DType, Device};
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::{
    sync::mpsc::{self, error::SendError, UnboundedReceiver, UnboundedSender},
    task::JoinHandle,
};
use tracing::{error, info, instrument};

/// `LlmService` - the entrypoint of the Atoma's vLLM service.
/// It receives requests from the Atoma's event subscriber
/// service. It validates and tokenizes such requests
/// and sends the valid request to the `LlmEngine`
pub struct LlmService {
    /// A receiver channel, it is responsible for
    /// receiving incoming requests from the
    /// atoma event subscriber
    atoma_event_subscriber_receiver: UnboundedReceiver<GenerateRequest>,
    /// Sender to communicate with an underlying
    /// `LlmEngine` running instance
    atoma_engine_sender: UnboundedSender<SequenceGroup>,
    /// Block size
    block_size: usize,
    /// Model weights cache directory
    cache_dir: PathBuf,
    /// Flushes the model storage
    flush_storage: bool,
    /// Join handle for the background task
    /// running the `LlmEngine` instance
    llm_engine_handle: JoinHandle<Result<(), LlmServiceError>>,
    /// Starting time of the instance
    start_time: Instant,
    /// Request counter
    request_counter: u64,
    /// A request validation instance
    validation_service: Validation,
}

impl LlmService {
    /// Starts the service
    #[instrument(skip_all)]
    #[allow(clippy::too_many_arguments)]
    pub async fn start<M>(
        api_key: String,
        atoma_event_subscriber_receiver: UnboundedReceiver<GenerateRequest>,
        atoma_client_sender: UnboundedSender<Vec<GenerateRequestOutput>>,
        cache_config: CacheConfig,
        cache_dir: PathBuf,
        device: Device,
        dtype: DType,
        flush_storage: bool,
        model_name: String,
        revision: String,
        scheduler_config: SchedulerConfig,
        tokenizer: Tokenizer,
        validation_service: Validation,
    ) -> Result<Self, LlmServiceError>
    where
        M: ModelExecutor + Send + Sync + 'static,
    {
        let block_size = cache_config.block_size;
        let scheduler = Scheduler::new(cache_config.clone(), scheduler_config.clone())?;

        let model_thread_dispatcher = ModelThreadDispatcher::start(
            api_key,
            cache_config,
            device,
            dtype,
            model_name,
            revision,
            scheduler_config,
        )?;

        let (request_sender, request_receiver) = mpsc::unbounded_channel();
        let llm_engine_handle = tokio::spawn(async move {
            let llm_engine = LlmEngine::new(
                atoma_client_sender,
                model_thread_dispatcher,
                request_receiver,
                scheduler,
                tokenizer,
            );

            llm_engine.run().await?;
            Ok::<_, LlmServiceError>(())
        });

        let start_time = Instant::now();

        Ok(Self {
            atoma_event_subscriber_receiver,
            atoma_engine_sender: request_sender,
            cache_dir,
            block_size,
            flush_storage,
            llm_engine_handle,
            request_counter: 0,
            start_time,
            validation_service,
        })
    }

    /// Main loop - awaits for incoming requests from the atoma
    /// event subscriber channel. It then validates the request
    /// and once the request is validated, it sends it to the
    /// `LlmEngine` background task
    #[instrument(skip_all)]
    pub async fn run(&mut self) -> Result<(), LlmServiceError> {
        while let Some(request) = self.atoma_event_subscriber_receiver.recv().await {
            let sequence_group = self.handle_request(request).await?;
            self.atoma_engine_sender.send(sequence_group)?;
        }

        Ok(())
    }

    /// Handles a new received `GenerateRequest`
    /// and produces a valid `SequenceGroup` out of it
    #[instrument(skip(self))]
    async fn handle_request(
        &mut self,
        request: GenerateRequest,
    ) -> Result<SequenceGroup, LlmServiceError> {
        info!("Received new request, with id = {}", request.request_id);

        let arrival_time = Instant::now();
        let request_id = request.request_id.clone();
        let valid_request = match self.process_received_request(request).await {
            Ok(valid_request) => valid_request,
            Err(e) => {
                error!("Failed to validate request with id = {request_id}, due to error: {e}");
                return Err(e);
            }
        };

        let sequence_id = self.request_counter;
        self.request_counter += 1;

        let sequence = Sequence::new(
            sequence_id,
            valid_request.inputs.clone(),
            valid_request.encoding.get_ids().to_vec(),
            self.block_size,
            valid_request.return_full_text,
        )?;
        let sequence_group = SequenceGroup::new(
            request_id,
            vec![sequence],
            arrival_time,
            valid_request.parameters.clone(),
            valid_request.stopping_parameters.clone(),
        )?;

        Ok(sequence_group)
    }

    /// Processes newly arrived inputs
    #[instrument(skip(self))]
    async fn process_received_request(
        &self,
        request: GenerateRequest,
    ) -> Result<ValidGenerateRequest, LlmServiceError> {
        Ok(self.validation_service.validate(request).await?)
    }

    /// Stops the running instance
    #[instrument(skip(self))]
    pub fn stop(self) -> Result<(), LlmServiceError> {
        info!(
            "Stopping the `LlmService` instance, running time: {:?}",
            self.start_time.elapsed()
        );

        if self.flush_storage {
            match std::fs::remove_dir(self.cache_dir) {
                Ok(()) => {}
                Err(e) => error!("Failed to remove storage folder, on shutdown: {e}"),
            };
        }

        self.llm_engine_handle.abort();

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum LlmServiceError {
    #[error("Model thread error: `{0}`")]
    ModelThreadError(#[from] ModelThreadError),
    #[error("Scheduler error: `{0}`")]
    SchedulerError(#[from] SchedulerError),
    #[error("Engine error: `{0}`")]
    EngineError(#[from] EngineError),
    #[error("Validation error: `{0}`")]
    ValidationError(#[from] ValidationError),
    #[error("Sequence error: `{0}`")]
    SequenceError(#[from] SequenceError),
    #[error("Send error: `{0}`")]
    SendError(#[from] SendError<SequenceGroup>),
}
