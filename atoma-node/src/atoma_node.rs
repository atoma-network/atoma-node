use std::path::Path;

use atoma_chat::service::{ChatService, ChatSessionError};
use atoma_client::{AtomaSuiClient, AtomaSuiClientError};
use atoma_helpers::{Firebase, FirebaseConfig};
#[cfg(feature = "supabase")]
use atoma_helpers::{Supabase, SupabaseConfig};
use atoma_inference::{
    models::config::ModelsConfig,
    service::{ModelService, ModelServiceError},
};
use atoma_input_manager::{AtomaInputManager, AtomaInputManagerError};
use atoma_output_manager::{AtomaOutputManager, AtomaOutputManagerError};
use atoma_streamer::{AtomaStreamer, AtomaStreamerError};
use atoma_sui::subscriber::{SuiSubscriber, SuiSubscriberError};
use atoma_types::{Request, Response};
use thiserror::Error;
use tokio::{
    sync::{
        mpsc::{self, Receiver},
        oneshot,
    },
    task::JoinError,
    try_join,
};
use tracing::{error, info, instrument, Span};

const NUM_RETRIES_PER_CHAT_MESSAGE: usize = 3;
const CHANNEL_SIZE: usize = 32;

pub struct AtomaNode {}

impl AtomaNode {
    /// Starts a new `AtomaNode` instance
    #[instrument(skip(config_path, json_server_req_rx))]
    pub async fn start<P>(
        config_path: P,
        json_server_req_rx: Receiver<(Request, oneshot::Sender<Response>)>,
    ) -> Result<(), AtomaNodeError>
    where
        P: AsRef<Path> + Send + 'static + Clone,
    {
        let model_config = ModelsConfig::from_file_path(config_path.clone());
        let model_ids = model_config.models().iter().map(|m| m.model_id()).collect();

        let (subscriber_req_tx, subscriber_req_rx) = mpsc::channel(CHANNEL_SIZE);
        let (atoma_node_resp_tx, atoma_node_resp_rx) = mpsc::channel(CHANNEL_SIZE);
        let (output_manager_tx, output_manager_rx) = mpsc::channel(CHANNEL_SIZE);
        let (input_manager_tx, input_manager_rx) = mpsc::channel(CHANNEL_SIZE);
        let (streamer_tx, streamer_rx) = tokio::sync::mpsc::channel(CHANNEL_SIZE);
        let (chat_request_sender, chat_request_receiver) = mpsc::channel(CHANNEL_SIZE);
        let (chat_client_sender, chat_client_receiver) = mpsc::channel(CHANNEL_SIZE);
        let (inference_sender, inference_receiver) = mpsc::channel(CHANNEL_SIZE);
        let (output_destination_sender, output_destination_receiver) = mpsc::channel(CHANNEL_SIZE);

        let firebase_config = FirebaseConfig::from_file_path(config_path.clone());
        let firebase = Firebase::new(
            firebase_config.api_key(),
            firebase_config.realtime_db_url()?,
            firebase_config.storage_url()?,
            firebase_config.small_id(),
        )
        .await?;
        #[cfg(feature = "supabase")]
        let supabase_config = SupabaseConfig::from_file_path(config_path.clone());
        #[cfg(feature = "supabase")]
        let supabase =
            Supabase::new(supabase_config.url("https")?, supabase_config.anon_key()).await?;

        let span = Span::current();
        let model_service_handle = {
            let span = span.clone();
            tokio::spawn(async move {
                let _enter = span.enter();
                info!("Spawning Model service..");
                let mut model_service = ModelService::start(
                    model_config,
                    json_server_req_rx,
                    subscriber_req_rx,
                    atoma_node_resp_tx,
                    streamer_tx,
                    inference_receiver,
                )?;
                model_service
                    .run()
                    .await
                    .map_err(AtomaNodeError::ModelServiceError)
            })
        };

        let chat_service_handle = {
            let span = span.clone();
            tokio::spawn(async move {
                let _enter = span.enter();
                info!("Starting Chat service..");
                let chat_service = ChatService::new(
                    model_ids,
                    NUM_RETRIES_PER_CHAT_MESSAGE,
                    chat_request_receiver,
                    chat_client_sender,
                    inference_sender,
                    output_destination_sender,
                );
                chat_service
                    .run()
                    .await
                    .map_err(AtomaNodeError::ChatServiceError)
            })
        };

        let sui_subscriber_handle = {
            let config_path = config_path.clone();
            let span = span.clone();
            tokio::spawn(async move {
                let _enter = span.enter();
                info!("Starting Sui subscriber service..");
                let sui_event_subscriber = SuiSubscriber::new_from_config(
                    config_path,
                    subscriber_req_tx,
                    input_manager_tx,
                    chat_request_sender,
                )
                .await?;
                sui_event_subscriber
                    .subscribe()
                    .await
                    .map_err(AtomaNodeError::SuiSubscriberError)
            })
        };

        let atoma_sui_client_handle = {
            let config_path = config_path.clone();
            let span = span.clone();
            tokio::spawn(async move {
                let _enter = span.enter();
                info!("Starting Atoma Sui client service..");
                let atoma_sui_client = AtomaSuiClient::new_from_config_file(
                    config_path.clone(),
                    atoma_node_resp_rx,
                    chat_client_receiver,
                    output_manager_tx,
                )?;
                atoma_sui_client
                    .run()
                    .await
                    .map_err(AtomaNodeError::AtomaSuiClientError)
            })
        };

        let atoma_output_manager_handle = {
            let config_path = config_path.clone();
            let firebase = firebase.clone();
            #[cfg(feature = "supabase")]
            let supabase = supabase.clone();
            let span = span.clone();
            tokio::spawn(async move {
                let _enter = span.enter();
                info!("Starting Atoma output manager service..");
                let atoma_output_manager = AtomaOutputManager::new(
                    config_path,
                    output_manager_rx,
                    firebase,
                    #[cfg(feature = "supabase")]
                    supabase,
                    output_destination_receiver,
                )
                .await?;
                atoma_output_manager.run().await.map_err(|e| {
                    error!("Error with Atoma output manager: {e}");
                    AtomaNodeError::AtomaOutputManagerError(e)
                })
            })
        };

        let atoma_input_manager_handle = {
            let firebase = firebase.clone();
            let span = span.clone();
            #[cfg(feature = "supabase")]
            let supabase = supabase.clone();
            tokio::spawn(async move {
                let _enter = span.enter();
                info!("Starting Atoma input manager service..");
                let atoma_input_manager = AtomaInputManager::new(
                    config_path,
                    input_manager_rx,
                    firebase,
                    #[cfg(feature = "supabase")]
                    supabase,
                )
                .await?;
                atoma_input_manager.run().await.map_err(|e| {
                    error!("Error with Atoma input manager: {e}");
                    AtomaNodeError::AtomaInputManagerError(e)
                })
            })
        };

        let atoma_streamer_handle = tokio::spawn(async move {
            let _enter = span.enter();
            info!("Starting Atoma streamer service..");
            let atoma_streamer = AtomaStreamer::new(
                streamer_rx,
                firebase,
                #[cfg(feature = "supabase")]
                supabase,
            )
            .await?;
            atoma_streamer.run().await.map_err(|e| {
                error!("Error with Atoma streamer: {e}");
                AtomaNodeError::AtomaStreamerError(e)
            })
        });

        // Store the handles to abort them if one of the tasks fails.
        let model_service_abort_handle = model_service_handle.abort_handle();
        let sui_subscriber_abort_handle = sui_subscriber_handle.abort_handle();
        let atoma_sui_client_abort_handle = atoma_sui_client_handle.abort_handle();
        let atoma_output_manager_abort_handle = atoma_output_manager_handle.abort_handle();
        let atoma_input_manager_abort_handle = atoma_input_manager_handle.abort_handle();
        let atoma_streamer_abort_handle = atoma_streamer_handle.abort_handle();
        let chat_service_abort_handle = chat_service_handle.abort_handle();

        // This is needed for the error propagation so the try_join! will fail if one of the tasks fails.
        let model_service_task = async { model_service_handle.await? };
        let sui_subscriber_task = async { sui_subscriber_handle.await? };
        let atoma_sui_client_task = async { atoma_sui_client_handle.await? };
        let atoma_output_manager_task = async { atoma_output_manager_handle.await? };
        let atoma_input_manager_task = async { atoma_input_manager_handle.await? };
        let atoma_streamer_task = async { atoma_streamer_handle.await? };
        let chat_service_task = async { chat_service_handle.await? };

        if let Err(e) = try_join!(
            model_service_task,
            sui_subscriber_task,
            atoma_sui_client_task,
            atoma_output_manager_task,
            atoma_input_manager_task,
            atoma_streamer_task,
            chat_service_task
        ) {
            // If one of them fails, abort them all.
            model_service_abort_handle.abort();
            sui_subscriber_abort_handle.abort();
            atoma_sui_client_abort_handle.abort();
            atoma_output_manager_abort_handle.abort();
            atoma_input_manager_abort_handle.abort();
            atoma_streamer_abort_handle.abort();
            chat_service_abort_handle.abort();
            Err(e)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Error)]
pub enum AtomaNodeError {
    #[error("Model service error: `{0}`")]
    ModelServiceError(#[from] ModelServiceError),
    #[error("Sui subscriber error: `{0}`")]
    SuiSubscriberError(#[from] SuiSubscriberError),
    #[error("Atoma Sui client error: `{0}`")]
    AtomaSuiClientError(#[from] AtomaSuiClientError),
    #[error("Atoma output manager error: `{0}`")]
    AtomaOutputManagerError(#[from] AtomaOutputManagerError),
    #[error("Atoma input manager error: `{0}`")]
    AtomaInputManagerError(#[from] AtomaInputManagerError),
    #[error("Atoma streamer error: `{0}`")]
    AtomaStreamerError(#[from] AtomaStreamerError),
    #[error("Chat service error: `{0}`")]
    ChatServiceError(#[from] ChatSessionError),
    #[error("Tokio failed to join task: `{0}`")]
    JoinError(#[from] JoinError),
    #[error("Url parse error: `{0}`")]
    UrlParseError(#[from] url::ParseError),
    #[error("Firebase authentication error: `{0}`")]
    FirebaseAuthError(#[from] atoma_helpers::FirebaseAuthError),
    #[cfg(feature = "supabase")]
    #[error("Supabase error: `{0}`")]
    SupabaseError(#[from] atoma_helpers::SupabaseError),
}
