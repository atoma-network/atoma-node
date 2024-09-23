use atoma_crypto::blake2b_hash;
use atoma_types::{
    ChatInferenceRequest, ChatInferenceResponse, ChatSessionData, Hash, StartChatRequest,
};
use std::time::Duration;
use std::{collections::HashMap, time::Instant};
use thiserror::Error;
use tokio::sync::mpsc::error::SendError;
use tokio::sync::oneshot::error::RecvError;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, info_span, instrument, trace, Span};

use crate::MAX_CONTEXT_WINDOW_SIZE;

/// The chat session retry timeout in seconds, corresponding to 3 seconds
pub const RETRY_TIMEOUT_SECS: u64 = 3;

pub type ChatId = String;
pub type Result<T> = std::result::Result<T, ChatSessionError>;
pub type OutputData = (ChatId, String, usize, usize, f64);

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
    /// Sender for client events
    pub client_sender: mpsc::Sender<(ChatId, usize, usize, Vec<Hash>, Vec<Hash>)>,
    /// Sending chat data to the output destination
    pub output_destination_sender: mpsc::Sender<OutputData>,
    /// The number of retries for processing the next chat request
    pub num_retries: usize,
    /// A map storing data for active chat sessions, keyed by chat ID.
    pub chat_sessions_data: HashMap<ChatId, ChatSessionData>,
    /// A tracing span for logging and monitoring.
    pub span: Span,
}

impl ChatService {
    /// Constructor
    pub fn new(
        model_ids: Vec<String>,
        num_retries: usize,
        start_chat_event_receiver: mpsc::Receiver<StartChatRequest>,
        chat_request_receiver: mpsc::Receiver<ChatInferenceRequest>,
        client_sender: mpsc::Sender<(ChatId, usize, usize, Vec<Hash>, Vec<Hash>)>,
        inference_sender: mpsc::Sender<(
            ChatInferenceRequest,
            oneshot::Sender<ChatInferenceResponse>,
        )>,
        output_destination_sender: mpsc::Sender<OutputData>,
    ) -> Self {
        Self {
            model_ids,
            start_chat_event_receiver,
            chat_request_receiver,
            client_sender,
            inference_sender,
            output_destination_sender,
            num_retries,
            chat_sessions_data: HashMap::new(),
            span: info_span!("chat-service"),
        }
    }

    #[instrument(skip_all)]
    pub async fn run(mut self) -> Result<()> {
        trace!("Starting chat service");
        loop {
            tokio::select! {
                Some(event) = self.start_chat_event_receiver.recv() => {
                    self.handle_start_chat_event(event)?;
                },
                Some(request) = self.chat_request_receiver.recv() => {
                    let chat_id = request.chat_id.clone();
                    self.handle_chat_request(request).await?;
                    // Check if the updated parameters have been reached, if
                    // that's the case, we commit to the chat session and close it
                    if let Some(chat_data) = self.chat_sessions_data.get(&chat_id) {
                        if chat_data.num_input_tokens >= chat_data.chat_session_metadata.max_input_tokens
                            || chat_data.num_output_tokens >= chat_data.chat_session_metadata.max_output_tokens
                            || chat_data.input_prompts_hashes.len() >= chat_data.chat_session_metadata.max_messages {
                            self.handle_close_chat_session(chat_id).await?;
                        }
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
    async fn handle_chat_request(&mut self, request: ChatInferenceRequest) -> Result<()> {
        let mut retries = 0;
        loop {
            match self.handle_inference(request.clone()).await {
                Ok(response) => {
                    self.handle_chat_response(response).await?;
                    break;
                }
                Err(e) => {
                    error!("Error handling chat request: {}", e);
                    retries += 1;
                    if retries >= self.num_retries {
                        error!(
                            "Max retries reached, evicting chat session: {}",
                            request.chat_id
                        );
                        // We can't proceed with the processing the chat session with some errors,
                        // for now we just commit to the chat session and close the chat session
                        self.handle_close_chat_session(request.chat_id).await?;
                        break;
                    }
                    // Give it a few seconds to retry
                    tokio::time::sleep(Duration::from_secs(RETRY_TIMEOUT_SECS)).await;
                }
            }
        }

        Ok(())
    }

    /// Handles the closure of a chat session by removing it from the active sessions and sending the session data to the client.
    ///
    /// # Arguments
    ///
    /// * `chat_id` - The ID of the chat session to be closed.
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Returns an error if the session data could not be sent to the client.
    ///
    /// # Details
    ///
    /// This function performs the following steps:
    /// - Removes the chat session data from the active sessions map.
    /// - Sends the input and output prompt hashes to the client.
    /// - Logs the success or failure of sending the session data.
    #[instrument(skip_all)]
    async fn handle_close_chat_session(&mut self, chat_id: ChatId) -> Result<()> {
        trace!("Handling eviction for chat session: {}", chat_id);
        let chat_data = self.chat_sessions_data.remove(&chat_id).unwrap();
        match self
            .client_sender
            .send((
                chat_id.clone(),
                chat_data.num_input_tokens,
                chat_data.num_output_tokens,
                chat_data.input_prompts_hashes,
                chat_data.output_prompts_hashes,
            ))
            .await
        {
            Ok(_) => {
                info!(
                    "Successfully sent chat session data to the client: {}",
                    chat_id
                );
            }
            Err(e) => {
                error!("Failed to send chat session data to the client: {}", e);
            }
        }
        Ok(())
    }

    /// Handles the chat response by sending it to the output destination.
    ///
    /// # Arguments
    ///
    /// * `response` - The chat inference response containing the chat ID and the output data.
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Returns an error if sending the response to the output destination fails.
    ///
    /// # Details
    ///
    /// This function performs the following steps:
    /// - Extracts the chat ID from the response.
    /// - Sends the chat response output to the output destination sender.
    /// - Logs the success or failure of sending the response.
    #[instrument(skip_all)]
    async fn handle_chat_response(&mut self, response: ChatInferenceResponse) -> Result<()> {
        trace!("Handling chat response");
        let chat_id = response.chat_id;
        match self
            .output_destination_sender
            .send((
                chat_id.clone(),
                response.output,
                response.input_tokens.len(),
                response.output_tokens.len(),
                response.time_to_generate,
            ))
            .await
        {
            Ok(_) => {
                info!(
                    "Successfully sent chat response to the output destination: {}",
                    chat_id
                );
            }
            Err(e) => {
                error!(
                    "Failed to send chat response to the output destination: {}",
                    e
                );
            }
        }
        Ok(())
    }

    /// Handles the inference request by forwarding it to the inference service and awaiting the response.
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
    async fn handle_inference(
        &mut self,
        request: ChatInferenceRequest,
    ) -> Result<ChatInferenceResponse> {
        trace!("Handling inference");
        let (sender, receiver) = oneshot::channel();
        let prompt = request.prompt.clone();
        let params = request.params.clone();
        self.inference_sender
            .send((request.clone(), sender))
            .await?;
        let response = receiver.await?;
        let chat_session_data = self.chat_sessions_data.get_mut(&response.chat_id).unwrap();
        utils::update_chat_session_data(chat_session_data, prompt, params, &response);
        Ok(response)
    }
}

pub(crate) mod utils {
    use atoma_types::GenerateParameters;

    use super::*;

    /// Updates the chat session data with the provided prompt, parameters, and response.
    ///
    /// # Arguments
    ///
    /// * `chat_session_data` - A mutable reference to the chat session data to be updated.
    /// * `prompt` - The user prompt string that initiated the chat inference request.
    /// * `params` - The parameters used for generating the chat inference response.
    /// * `response` - The chat inference response containing the input and output tokens.
    ///
    /// # Details
    ///
    /// This function updates the chat session data with the following:
    /// - Sets the last updated time to the current time.
    /// - Stores the last user prompt.
    /// - Computes and stores the hash of the input and output tokens.
    /// - Updates the number of input and output tokens.
    /// - Extends the context token IDs with the input and output tokens.
    /// - Appends the generation parameters to the session data.
    pub(crate) fn update_chat_session_data(
        chat_session_data: &mut ChatSessionData,
        prompt: String,
        params: GenerateParameters,
        response: &ChatInferenceResponse,
    ) {
        let input_tokens = response.input_tokens.clone();
        let output_tokens = response.output_tokens.clone();
        let input_prompt_hash = blake2b_hash(
            &input_tokens
                .iter()
                .flat_map(|u| u.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let output_prompt_hash = blake2b_hash(
            &output_tokens
                .iter()
                .flat_map(|u| u.to_le_bytes())
                .collect::<Vec<_>>(),
        );

        chat_session_data.last_updated_time = Instant::now();
        chat_session_data.last_user_prompt = Some(prompt);
        chat_session_data
            .input_prompts_hashes
            .push(input_prompt_hash);
        chat_session_data
            .output_prompts_hashes
            .push(output_prompt_hash);

        chat_session_data.num_input_tokens += input_tokens.len();
        chat_session_data.num_output_tokens += output_tokens.len();

        chat_session_data.context_token_ids.extend(input_tokens);
        chat_session_data.context_token_ids.extend(output_tokens);

        chat_session_data.params.push(params);
    }
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
    #[error("Client send error: {0}")]
    ClientSendError(#[from] SendError<(ChatId, Vec<Hash>, Vec<Hash>)>),
    #[error("Output destination send error: {0}")]
    OutputDestinationSendError(#[from] SendError<(ChatId, String)>),
}
