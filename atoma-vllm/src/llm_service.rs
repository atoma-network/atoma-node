use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use crate::{
    config::{CacheConfig, SchedulerConfig},
    llm_engine::{EngineError, GenerateRequestOutput, LlmEngine},
    model_executor::{ModelExecutor, ModelLoaderError, ModelThreadDispatcher, ModelThreadError},
    scheduler::{Scheduler, SchedulerError},
    sequence::{Sequence, SequenceError, SequenceGroup},
    tokenizer::TokenizerWorker,
    types::GenerateRequest,
    validation::{ValidGenerateRequest, Validation, ValidationError},
};
use candle_core::{DType, Device};
use metrics::{counter, gauge};
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::{
    sync::{
        broadcast::error,
        mpsc::{self, error::SendError, UnboundedReceiver, UnboundedSender},
        oneshot,
    },
    task::JoinHandle,
};
use tracing::{error, info, info_span, instrument, Span};

// TODO:
// 1. We should have a configurable number of tokenizer workers
//     in the service. This can be a configurable parameter.
// 2. Add a configuration file for the `LlmService` struct
// 3. Add proper tokenizer shutdown logic, and other related services

/// `LlmService` - the entrypoint of the Atoma's inference service.
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
    /// Flushes the model storage, once the service is stopped
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
    /// Tokenizer handle
    tokenizer_handle: JoinHandle<Result<(), LlmServiceError>>,
    /// Shutdown signal
    shutdown_signal: oneshot::Receiver<()>,
    /// Tracing span
    span: Span,
}

impl LlmService {
    /// Starts the service
    #[instrument(skip_all)]
    #[allow(clippy::too_many_arguments)]
    pub async fn start<M, T: AsRef<Path>>(
        api_key: String,
        atoma_event_subscriber_receiver: UnboundedReceiver<GenerateRequest>,
        atoma_client_sender: UnboundedSender<Vec<GenerateRequestOutput>>,
        cache_config: CacheConfig,
        cache_dir: T,
        device: Device,
        dtype: DType,
        flush_storage: bool,
        model_name: String,
        num_tokenizer_workers: usize,
        revision: String,
        scheduler_config: SchedulerConfig,
        validation_service: Validation,
        shutdown_signal: oneshot::Receiver<()>,
    ) -> Result<Self, LlmServiceError>
    where
        M: ModelExecutor + Send + Sync + 'static,
    {
        let span = info_span!("llm-service");
        let _enter = span.enter();

        info!("Starting a new `LlmService` instance..");

        let block_size = cache_config.block_size;
        let scheduler = Scheduler::new(cache_config.clone(), scheduler_config.clone())?;

        let start_time = Instant::now();

        // We do not need to spawn a new thread for fetching model weights files,
        // as the service will not start until the model is loaded in memory
        // NOTE: for now we use a synchronous model loader
        let file_paths = M::fetch(api_key, cache_dir, model_name, revision)?;
        let tokenizer = Tokenizer::from_file(&file_paths.tokenizer_path)?;
        let model = M::load(device.clone(), dtype, &file_paths)?;

        let (tokenizer_sender, tokenizer_receiver) = mpsc::unbounded_channel();

        let tokenizer_handle = tokio::spawn(async move {
            TokenizerWorker::start(tokenizer, tokenizer_receiver, num_tokenizer_workers)
                .await
                .expect("Failed to start tokenizer");
        });

        let model_thread_dispatcher = ModelThreadDispatcher::start::<M>(
            cache_config,
            device,
            dtype,
            model,
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

        Ok(Self {
            atoma_event_subscriber_receiver,
            atoma_engine_sender: request_sender,
            cache_dir: cache_dir.to_path_buf(),
            block_size,
            flush_storage,
            llm_engine_handle,
            request_counter: 0,
            start_time,
            validation_service,
            shutdown_signal,
            tokenizer_handle,
            span,
        })
    }

    /// Main loop - awaits for incoming requests from the atoma
    /// event subscriber channel. It then validates the request
    /// and once the request is validated, it sends it to the
    /// `LlmEngine` background task
    #[instrument(skip_all)]
    pub async fn run(&mut self) -> Result<(), LlmServiceError> {
        loop {
            tokio::select! {
                Some(request) = self.atoma_event_subscriber_receiver.recv() => {
                    let sequence_group = match self.handle_request(request).await {
                        Ok(sequence_group) => sequence_group,
                        Err(e) => {
                            error!("Failed to handle request, with error: {e}");
                            continue;
                            // TODO: we need to handle errors more appropriately,
                            //       we want to commit to these errors, as validation
                            //       errors should also be committed to, by the node.
                        }
                    };
                    self.atoma_engine_sender.send(sequence_group)?;
                },
                _ = self.shutdown_signal.recv() => {
                    info!("Received shutdown signal, stopping `LlmService` instance..");
                    break;
                }
            }
        }
        self.stop().await
    }

    /// Handles a new received `GenerateRequest`
    /// and produces a valid `SequenceGroup` from it
    #[instrument(skip_all)]
    async fn handle_request(
        &mut self,
        request: GenerateRequest,
    ) -> Result<SequenceGroup, LlmServiceError> {
        let _enter = self.span.enter();

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

        counter!("llm-service-requests-total").increment(1);
        gauge!("llm-service-request-validation-time").set(arrival_time.elapsed().as_secs_f32());
        info!(
            request_id = %request_id,
            validation_time = ?arrival_time.elapsed(),
            "Received and validated new request"
        );

        let sequence_id = self.request_counter;

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
    #[instrument(skip_all)]
    async fn process_received_request(
        &self,
        request: GenerateRequest,
    ) -> Result<ValidGenerateRequest, LlmServiceError> {
        Ok(self.validation_service.validate(request).await?)
    }

    /// Stops the running instance
    #[instrument(skip(self))]
    pub async fn stop(self) -> Result<(), LlmServiceError> {
        info!(
            "Stopping the `LlmService` instance, running time: {:?}",
            self.start_time.elapsed()
        );

        // Flush storage if configured
        if self.flush_storage {
            match tokio::fs::remove_dir(self.cache_dir.as_ref()).await {
                Ok(()) => info!("Successfully removed storage folder"),
                Err(e) => error!("Failed to remove storage folder, on shutdown: {e}"),
            }
        }

        // Abort the background tasks
        self.llm_engine_handle.abort();

        // Awaits for the task to finish and handle any errors
        match self.llm_engine_handle.await {
            Ok(result) => match result {
                Ok(()) => info!("`LlmService` background task finished successfully"),
                Err(e) => error!("`LlmService` background task failed, with error: {e}"),
            },
            Err(e) => error!("Failed to abort the background task: {e}"),
        }

        self.tokenizer_handle.abort();

        match self.tokenizer_handle.await {
            Ok(result) => match result {
                Ok(()) => info!("`LlmService` tokenizer background task finished successfully"),
                Err(e) => error!("`LlmService` tokenizer background task failed, with error: {e}"),
            },
            Err(e) => error!("Failed to abort the tokenizer background task: {e}"),
        }

        info!("`LlmService` stopped successfully");
        Ok(())
    }
}

impl Drop for LlmService {
    fn drop(&mut self) {
        let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
        runtime.block_on(async {
            if let Err(e) = self.stop().await {
                error!("`LlmService` instance failed to stop gracefully, with error: {e}");
            }
        });
        info!("`LlmService` instance dropped successfully");
    }
}

#[derive(Debug, Error)]
pub enum LlmServiceError {
    #[error("Boxed error: `{0}`")]
    BoxedError(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("Model loader error: `{0}`")]
    ModelLoaderError(#[from] ModelLoaderError),
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
