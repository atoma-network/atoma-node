use std::path::Path;

use atoma_helpers::Firebase;
use atoma_types::InputSource;
use config::AtomaInputManagerConfig;
use firebase::FirebaseInputManager;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{info, instrument};

mod config;
mod firebase;

/// `AtomaInputManager` - manages different input sources
///     requests, allowing for a flexible interaction between
///     the user and the Atoma Network.
///
/// Current available options for storing the input on behalf
/// of the user consists of Firebase.
pub struct AtomaInputManager {
    /// Firebase's input manager instance.
    firebase_input_manager: FirebaseInputManager,
    /// A mpsc receiver that receives tuples of `InputSource` and
    /// the actual user prompt, in JSON format.
    input_manager_rx: mpsc::Receiver<(InputSource, tokio::sync::oneshot::Sender<String>)>,
}

impl AtomaInputManager {
    /// Constructor
    pub async fn new<P: AsRef<Path>>(
        config_file_path: P,
        input_manager_rx: mpsc::Receiver<(InputSource, tokio::sync::oneshot::Sender<String>)>,
        firebase: Firebase,
    ) -> Result<Self, AtomaInputManagerError> {
        let config = AtomaInputManagerConfig::from_file_path(config_file_path);
        let firebase_input_manager = FirebaseInputManager::new(
            config.firebase_url,
            config.firebase_email,
            config.firebase_password,
            config.firebase_api_key,
            firebase,
            config.small_id,
        )
        .await?;
        Ok(Self {
            firebase_input_manager,
            input_manager_rx,
        })
    }

    /// Main loop, responsible for continuously listening to incoming user prompts.
    #[instrument(skip_all)]
    pub async fn run(mut self) -> Result<(), AtomaInputManagerError> {
        info!("Starting firebase input service..");
        while let Some((input_source, oneshot)) = self.input_manager_rx.recv().await {
            info!(
                "Received a new input to be submitted to a data storage {:?}..",
                input_source
            );
            let text = match input_source {
                InputSource::Firebase { request_id } => {
                    self.firebase_input_manager
                        .handle_get_request(request_id)
                        .await?
                }
                InputSource::Raw { prompt } => prompt,
            };
            oneshot
                .send(text)
                .map_err(AtomaInputManagerError::SendPromptError)?;
        }

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum AtomaInputManagerError {
    #[error("Deserialize JSON value error: `{0}`")]
    DeserializeError(#[from] serde_json::Error),
    #[error("Request error: `{0}`")]
    RequestError(#[from] reqwest::Error),
    #[error("GraphQl error: `{0}`")]
    GraphQlError(String),
    #[error("Invalid input source: `{0}`")]
    InvalidInputSource(String),
    #[error("Firebase authentication error: `{0}`")]
    FirebaseAuthError(#[from] atoma_helpers::FirebaseAuthError),
    #[error("Url error: `{0}`")]
    UrlError(String),
    #[error("Url parse error: `{0}`")]
    UrlParseError(#[from] url::ParseError),
    #[error("Error sending prompt to the model: `{0:?}`")]
    SendPromptError(String),
    #[error("Timeout error, could not get input from firebase in time")]
    TimeoutError,
}
