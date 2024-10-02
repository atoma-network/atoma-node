use std::{str::FromStr, time::Instant};

use crate::serving::tokenizer::TokenizerWorker;

use super::{
    config::ModelConfig,
    llm_engine::{EngineError, GenerateRequestOutput, LlmEngine},
    model_executor::{ModelExecutor, ModelThreadDispatcher},
    sequence::{Sequence, SequenceError},
    tokenizer::EncodeTokenizerRequest,
    types::GenerateRequest,
    validation::{ValidGenerateRequest, Validation, ValidationError},
};
use candle::{DType, Device};
use metrics::{counter, gauge};
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::{
    sync::{
        broadcast::Receiver,
        mpsc::{self, error::SendError, UnboundedReceiver, UnboundedSender},
    },
    task::JoinHandle,
};
use tracing::{error, info, info_span, instrument, Span};

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
    atoma_engine_sender: UnboundedSender<Sequence>,
    /// Model configuration
    model_config: ModelConfig,
    /// Join handle for the background task
    /// running the `LlmEngine` instance
    llm_engine_handle: JoinHandle<Result<(), LlmServiceError>>,
    /// Starting time of the instance
    start_time: Instant,
    /// Request counter
    request_counter: u64,
    /// Tokenizer handle
    tokenizer_handle: JoinHandle<Result<(), LlmServiceError>>,
    /// Shutdown signal receiver
    shutdown_signal: Receiver<()>,
    /// A request validation instance
    validation_service: Validation,
    /// Span
    span: Span,
}

impl LlmService {
    /// Starts the service
    #[instrument(skip_all)]
    #[allow(clippy::too_many_arguments)]
    pub async fn start<M>(
        atoma_event_subscriber_receiver: UnboundedReceiver<GenerateRequest>,
        atoma_client_sender: UnboundedSender<Vec<GenerateRequestOutput>>,
        model_config: ModelConfig,
        num_tokenizer_workers: usize,
        tokenizer_receiver: mpsc::UnboundedReceiver<EncodeTokenizerRequest>,
        shutdown_signal: Receiver<()>,
        validation_service: Validation,
    ) -> Result<Self, LlmServiceError>
    where
        // M: ModelExecutor + Send + Sync + 'static,
        M: ModelExecutor + Send + Sync + 'static,
    {
        let span = info_span!("llm-service");
        let _span = span.clone();
        let _enter = _span.enter();

        info!("Starting a new `LlmService` instance..");
        let start_time = Instant::now();

        // We do not need to spawn a new thread for fetching model weights files,
        // as the service will not start until the model is loaded in memory
        // NOTE: for now we use a synchronous model loader
        let file_paths = M::fetch(
            model_config.api_key.unwrap_or_default(),
            &model_config.cache_dir,
            model_config.model_name,
            model_config.revision,
        )?;
        let tokenizer = Tokenizer::from_file(&file_paths.tokenizer_path)?;

        // TODO: support multiple devices
        let device = Device::new_cuda(model_config.device_ids[0])?;
        let dtype = DType::from_str(&model_config.dtype).unwrap();
        let model = M::load(device, dtype, &file_paths)?;

        let cloned_tokenizer = tokenizer.clone();
        let tokenizer_handle = tokio::spawn(async move {
            Ok(
                TokenizerWorker::start(cloned_tokenizer, tokenizer_receiver, num_tokenizer_workers)
                    .await?,
            )
        });

        let model_thread_dispatcher = ModelThreadDispatcher::start::<M>(device, dtype, model)?;

        let (request_sender, _request_receiver) = mpsc::unbounded_channel();
        let llm_engine_handle = tokio::spawn(async move {
            let llm_engine = LlmEngine::new(
                atoma_client_sender,
                model_thread_dispatcher,
                request_receiver,
                // scheduler,
                tokenizer,
            );

            llm_engine.run().await?;
            Ok::<_, LlmServiceError>(())
        });

        Ok(Self {
            atoma_event_subscriber_receiver,
            atoma_engine_sender: request_sender,
            model_config,
            llm_engine_handle,
            request_counter: 0,
            start_time,
            shutdown_signal,
            validation_service,
        })
    }

    /// Main loop - awaits for incoming requests from the atoma
    /// event subscriber channel. It then validates the request
    /// and once the request is validated, it sends it to the
    /// `LlmEngine` background task
    #[instrument(skip_all)]
    pub async fn run(mut self) -> Result<(), LlmServiceError> {
        loop {
            tokio::select! {
                request = self.atoma_event_subscriber_receiver.recv() => {
                    if let Some(request) = request {
                        info!("Received new request, with id = {}", request.request_id);
                        let sequence = self.handle_request(request).await?;
                        self.atoma_engine_sender.send(sequence).map_err(|e| LlmServiceError::SendError(Box::new(e)))?;
                    }
                },
                _ = self.shutdown_signal.recv() => {
                    info!("Received shutdown signal");
                    break;
                }
            }
        }
        self.stop()?;
        Ok(())
    }

    /// Handles a new received `GenerateRequest`
    /// and produces a valid `SequenceGroup` out of it
    #[instrument(skip(self))]
    async fn handle_request(
        &mut self,
        request: GenerateRequest,
    ) -> Result<Sequence, LlmServiceError> {
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
            arrival_time,
            valid_request.return_full_text,
            valid_request.parameters.clone(),
            valid_request.stopping_parameters.clone(),
        );

        Ok(sequence)
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

        if self.model_config.flush_storage {
            match std::fs::remove_dir(self.model_config.cache_dir) {
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
    // #[error("Model thread error: `{0}`")]
    // ModelThreadError(#[from] ModelThreadError),
    // #[error("Scheduler error: `{0}`")]
    // SchedulerError(#[from] SchedulerError),
    #[error("Engine error: `{0}`")]
    EngineError(#[from] EngineError),
    #[error("Validation error: `{0}`")]
    ValidationError(#[from] ValidationError),
    #[error("Sequence error: `{0}`")]
    SequenceError(#[from] SequenceError),
    #[error("Send error: `{0}`")]
    SendError(#[from] Box<SendError<Sequence>>),
}

#[cfg(test)]
mod tests {
    use crate::serving::types::GenerateParameters;

    use super::*;

    #[tokio::test]
    async fn test_llm_service() {
        let (atoma_event_subscriber_sender, atoma_event_subscriber_receiver) =
            mpsc::unbounded_channel();
        let (atoma_client_sender, _atoma_client_receiver) = mpsc::unbounded_channel();
        let (shutdown_signal_sender, shutdown_signal_receiver) = tokio::sync::broadcast::channel(1);
        let (tokenizer_sender, _tokenizer_receiver) = mpsc::unbounded_channel();
        let model_config = ModelConfig::new(
            None,
            "".to_string(),
            true,
            "".to_string(),
            "".to_string(),
            vec![0],
            "".to_string(),
        );
        let tokenizer = Tokenizer::from_pretrained("anthony/tokenizers-test", None).unwrap();

        let validation_service = Validation::new(1, 4, 16, 512, 1024, tokenizer_sender);

        let llm_service_handle = tokio::spawn(async move {
            let mut llm_service = LlmService::start::<()>(
                atoma_event_subscriber_receiver,
                atoma_client_sender,
                model_config,
                tokenizer,
                shutdown_signal_receiver,
                validation_service,
            )
            .await?;
            llm_service.run().await?;
            Ok::<_, LlmServiceError>(())
        });

        for i in 0..100 {
            atoma_event_subscriber_sender
                .send(GenerateRequest {
                    request_id: format!("request-{i}",),
                    inputs: format!("Hello world, {i}"),
                    parameters: GenerateParameters::default(),
                })
                .unwrap();
        }

        shutdown_signal_sender.send(());
        llm_service_handle.await.unwrap();
    }
}
