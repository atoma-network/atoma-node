use atoma_types::{ChatInferenceRequest, ChatInferenceResponse, StartChatRequest};
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use std::{collections::HashMap, time::Instant};
use thiserror::Error;
use tokio::sync::mpsc::error::SendError;
use tokio::sync::oneshot::error::RecvError;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info_span, instrument, trace, Span};

use crate::types::ChatSessionData;
use crate::MAX_CONTEXT_WINDOW_SIZE;

/// The chat session eviction timeout in seconds, corresponding to 15 minutes
pub const CHAT_SESSION_EVICTION_TIMEOUT_SECS: u64 = 900;
/// The chat session retry timeout in seconds, corresponding to 3 seconds
pub const RETRY_TIMEOUT_SECS: u64 = 3;

/// The chat id
pub type ChatId = String;
pub type Result<T> = std::result::Result<T, ChatSessionError>;

/// A service that manages chat sessions, handles chat requests, and processes inferences.
pub struct ChatService {
    /// The model ids that the chat service can handle
    pub model_ids: Vec<String>,
    /// Receiver for start chat events.
    pub start_chat_event_receiver: mpsc::Receiver<StartChatRequest>,
    /// Receiver for chat requests.
    pub chat_request_receiver: mpsc::Receiver<ChatInferenceRequest>,
    /// Sender for chat inference requests.
    pub inference_sender:
        mpsc::Sender<(ChatInferenceRequest, oneshot::Sender<ChatInferenceResponse>)>,
    /// The number of retries for processing the next chat request
    pub num_retries: usize,
    /// A map storing data for active chat sessions, keyed by chat ID.
    pub chat_sessions_data: HashMap<ChatId, ChatSessionData>,
    /// A collection of timers for managing chat session timeouts.
    pub timers: FuturesUnordered<oneshot::Receiver<ChatId>>,
    /// A tracing span for logging and monitoring.
    pub span: Span,
}

impl ChatService {
    /// Constructor
    pub fn new<T: AsRef<Path>>(
        model_ids: Vec<String>,
        num_retries: usize,
        start_chat_event_receiver: mpsc::Receiver<StartChatRequest>,
        chat_request_receiver: mpsc::Receiver<ChatInferenceRequest>,
        inference_sender: mpsc::Sender<(
            ChatInferenceRequest,
            oneshot::Sender<ChatInferenceResponse>,
        )>,
    ) -> Self {
        Self {
            model_ids,
            start_chat_event_receiver,
            chat_request_receiver,
            inference_sender,
            num_retries,
            chat_sessions_data: HashMap::new(),
            timers: FuturesUnordered::new(),
            span: info_span!("chat-service"),
        }
    }

    #[instrument(skip_all)]
    pub async fn run(mut self) -> Result<()> {
        trace!("Starting chat service");
        let mut futures = FuturesUnordered::new();
        loop {
            tokio::select! {
                Some(event) = self.start_chat_event_receiver.recv() => {
                    let chat_id = event.chat_id.clone();
                    self.handle_start_chat_event(event)?;
                    futures.push(start_timer(chat_id));
                },
                Some(request) = self.chat_request_receiver.recv() => {
                    let mut retries = 0;
                    loop {
                        match self.handle_chat_request(request.clone()).await {
                            Ok(response) => {
                                self.handle_chat_response(response).await?;
                            },
                            Err(e) => {
                                error!("Error handling chat request: {}", e);
                                retries += 1;
                                if retries >= self.num_retries {
                                    error!("Max retries reached, evicting chat session: {}", request.chat_id);
                                    self.handle_eviction(request.chat_id).await?;
                                    break;
                                }
                                // Give it a few seconds to retry
                                tokio::time::sleep(Duration::from_secs(RETRY_TIMEOUT_SECS)).await;
                            }
                        }
                    }
                },
                Some(chat_id) = futures.next() => {
                    if let Some(chat_session_data) =  self.chat_sessions_data.get(&chat_id) {
                        let last_updated_time = chat_session_data.last_updated_time;
                        if last_updated_time.elapsed() > Duration::from_secs(CHAT_SESSION_EVICTION_TIMEOUT_SECS) {
                            // The last time that the chat session was updated is greater than the eviction timeout,
                            // for this reason, we evict the chat session to disk
                            self.handle_eviction(chat_id).await?;
                        } else {
                            // The last time that the chat session was updated is less than the eviction timeout,
                            // for this reason, we restart the timer
                            futures.push(start_timer(chat_id));
                        }
                    } else {
                        error!("Chat session not found: {}", chat_id);
                    }
                },
            }
        }
    }
}

impl ChatService {
    /// Handles the start chat event by creating a new chat session.
    ///
    /// # Arguments
    ///
    /// * `event` - The start chat request containing the chat ID and other session parameters.
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Returns an error if the chat session already exists.
    #[instrument(skip_all)]
    fn handle_start_chat_event(&mut self, event: StartChatRequest) -> Result<()> {
        trace!("Handling start chat event");
        let chat_id = event.chat_id;

        if self.chat_sessions_data.contains_key(&chat_id) {
            return Err(ChatSessionError::ChatSessionAlreadyExists(chat_id));
        }

        let max_context_window_size = MAX_CONTEXT_WINDOW_SIZE.get(&event.model_id).unwrap();
        let chat_session_data = ChatSessionData::new(
            chat_id.clone(),
            event.user_pk,
            Instant::now(),
            event.model_id,
            *max_context_window_size,
            event.max_input_tokens,
            event.max_output_tokens,
            event.max_messages,
            event.output_destination,
            event.random_seed,
        );
        self.chat_sessions_data.insert(chat_id, chat_session_data);

        Ok(())
    }

    /// Handles a chat inference request by forwarding it to the inference service and awaiting the response.
    ///
    /// # Arguments
    ///
    /// * `request` - The chat inference request containing the necessary data for processing.
    ///
    /// # Returns
    ///
    /// * `Result<ChatInferenceResponse>` - Returns the chat inference response or an error if the request fails.
    ///
    /// # Errors
    ///
    /// * `ChatSessionError::InferenceRequestFailed` - If sending the inference request to the inference service fails.
    /// * `ChatSessionError::RecvError` - If receiving the response from the inference service fails.
    #[instrument(skip_all)]
    async fn handle_chat_request(
        &mut self,
        request: ChatInferenceRequest,
    ) -> Result<ChatInferenceResponse> {
        trace!("Handling chat request");
        let (sender, receiver) = oneshot::channel();
        let prompt = request.prompt.clone();
        self.inference_sender
            .send((request.clone(), sender))
            .await?;
        let response = receiver.await?;
        let chat_session_data = self.chat_sessions_data.get_mut(&response.chat_id).unwrap();
        update_chat_session_data(chat_session_data, prompt, &response);
        Ok(response)
    }

    #[instrument(skip_all)]
    async fn handle_eviction(&mut self, chat_id: ChatId) -> Result<()> {
        trace!("Handling eviction for chat session: {}", chat_id);
        Ok(())
    }

    #[instrument(skip_all)]
    async fn handle_chat_response(&mut self, response: ChatInferenceResponse) -> Result<()> {
        trace!("Handling chat response");

        Ok(())
    }
}

/// Starts a timer for the chat session eviction.
///
/// This function asynchronously waits for the duration specified by `CHAT_SESSION_EVICTION_TIMEOUT_SECS`
/// and then returns the `chat_id` to indicate that the timer has expired.
///
/// # Arguments
///
/// * `chat_id` - The unique identifier for the chat session.
///
/// # Returns
///
/// * `ChatId` - The same `chat_id` that was passed in, indicating the timer has expired.
///
/// # Examples
///
/// ```rust
/// let chat_id = "example_chat_id".to_string();
/// let expired_chat_id = start_timer(chat_id).await;
/// assert_eq!(expired_chat_id, "example_chat_id");
/// ```
async fn start_timer(chat_id: ChatId) -> ChatId {
    tokio::time::sleep(Duration::from_secs(CHAT_SESSION_EVICTION_TIMEOUT_SECS)).await;
    chat_id
}

/// Updates the chat session data with the response from the inference service.
///
/// # Arguments
///
/// * `chat_session_data` - The chat session data to update.
/// * `response` - The response from the inference service.
fn update_chat_session_data(
    chat_session_data: &mut ChatSessionData,
    prompt: String,
    response: &ChatInferenceResponse,
) {
    let input_tokens = response.input_tokens.clone();
    let output_tokens = response.output_tokens.clone();

    chat_session_data.last_updated_time = Instant::now();
    chat_session_data.last_user_prompt = Some(prompt);
    chat_session_data
        .input_prompts_hashes
        .push(request.prompt_hash);
    chat_session_data
        .output_prompts_hashes
        .push(response.output_tokens);
    chat_session_data
        .context_token_ids
        .extend(response.input_tokens);
    chat_session_data
        .context_token_ids
        .extend(response.output_tokens);
    chat_session_data.num_input_tokens += input_tokens.len();
    chat_session_data.num_output_tokens += output_tokens.len();
}

pub enum ServiceEvent {
    NewRequest(ChatInferenceRequest),
    Eviction(ChatId),
}

#[derive(Error, Debug)]
pub enum ChatSessionError {
    #[error("The chat session already exists, with id: `{0}`")]
    ChatSessionAlreadyExists(String),
    #[error("The chat session is not found, with id: `{0}`")]
    ChatSessionNotFound(String),
    #[error("Inference request failed: {0}")]
    InferenceRequestFailed(
        #[from] SendError<(ChatInferenceRequest, oneshot::Sender<ChatInferenceResponse>)>,
    ),
    #[error("Recv error: {0}")]
    RecvError(#[from] RecvError),
}
