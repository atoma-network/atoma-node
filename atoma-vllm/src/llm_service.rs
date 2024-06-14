use crate::{
    config::{CacheConfig, SchedulerConfig},
    llm_engine::{EngineError, GenerateRequestOutput, LlmEngine},
    model_executor::{ModelExecutor, ModelThreadDispatcher, ModelThreadError},
    scheduler::{Scheduler, SchedulerError},
    sequence::SequenceGroup,
    types::GenerateRequest,
    validation::Validation,
};
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::{
    sync::mpsc::{self, UnboundedReceiver, UnboundedSender, *},
    task::JoinHandle,
};
use tracing::instrument;

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
    /// Join handle for the background task
    /// running the `LlmEngine` instance
    llm_engine_handle: JoinHandle<Result<(), LlmServiceError>>,
    /// A request validation instance
    validation_service: Validation,
}

impl LlmService {
    /// Starts the service
    #[instrument(skip_all)]
    pub async fn start<M>(
        atoma_event_subscriber_receiver: UnboundedReceiver<GenerateRequest>,
        atoma_client_sender: UnboundedSender<Vec<GenerateRequestOutput>>,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        model: M,
        tokenizer: Tokenizer,
        validation_service: Validation,
    ) -> Result<Self, LlmServiceError>
    where
        M: ModelExecutor + Send + Sync + 'static,
    {
        let scheduler = Scheduler::new(cache_config, scheduler_config)?;
        // TODO: it might be better to initialize the model `M` inside this method
        let eos_token_id = model.eos_token_id().unwrap();
        let model_thread_dispatcher = ModelThreadDispatcher::start(model)?;

        let (request_sender, request_receiver) = mpsc::unbounded_channel();
        let llm_engine_handle = tokio::spawn(async move {
            let llm_engine = LlmEngine::new(
                atoma_client_sender,
                eos_token_id,
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
            llm_engine_handle,
            validation_service,
        })
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
}
