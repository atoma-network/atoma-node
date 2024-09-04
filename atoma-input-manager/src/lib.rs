use std::path::Path;

use atoma_helpers::Firebase;
use atoma_types::{InputFormat, InputSource, ModelInput};
use config::AtomaInputManagerConfig;
use firebase::FirebaseInputManager;
use ipfs::IpfsInputManager;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{info, instrument};

mod config;
mod firebase;
mod ipfs;

/// `AtomaInputManager` - manages different input sources
///     requests, allowing for a flexible interaction between
///     the user and the Atoma Network.
///
/// Current available options for storing the input on behalf
/// of the user consists of Firebase.
pub struct AtomaInputManager {
    /// Firebase's input manager instance.
    firebase_input_manager: FirebaseInputManager,
    /// IPFS's input manager instance.
    ipfs_input_manager: IpfsInputManager,
    /// A mpsc receiver that receives tuples of `InputSource` and
    /// the actual user prompt, in JSON format.
    input_manager_rx: mpsc::Receiver<(
        InputSource,
        tokio::sync::oneshot::Sender<Result<ModelInput, AtomaInputManagerError>>,
    )>,
}

impl AtomaInputManager {
    /// Constructor
    #[instrument(skip_all)]
    pub async fn new<P: AsRef<Path>>(
        config_file_path: P,
        input_manager_rx: mpsc::Receiver<(
            InputSource,
            tokio::sync::oneshot::Sender<Result<ModelInput, AtomaInputManagerError>>,
        )>,
        firebase: Firebase,
    ) -> Result<Self, AtomaInputManagerError> {
        info!("Starting Atoma Input Manager...");
        let start = std::time::Instant::now();
        let config = AtomaInputManagerConfig::from_file_path(config_file_path);
        let ipfs_input_manager = IpfsInputManager::new().await?;
        let firebase_input_manager = FirebaseInputManager::new(
            config.firebase_url,
            config.firebase_email,
            config.firebase_password,
            config.firebase_api_key,
            firebase,
            config.small_id,
        )
        .await?;
        info!("Atoma Input Manager started in {:?}", start.elapsed());
        Ok(Self {
            ipfs_input_manager,
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
            let model_input = match input_source {
                InputSource::Firebase { request_id } => {
                    self.firebase_input_manager
                        .handle_get_request(request_id)
                        .await
                }
                InputSource::Ipfs { cid, format } => match format {
                    InputFormat::Image => self.ipfs_input_manager.fetch_image(&cid).await,
                    InputFormat::Text => self.ipfs_input_manager.fetch_text(&cid).await,
                },
                InputSource::Raw { prompt } => Ok(ModelInput::Text(prompt)),
            };
            oneshot
                .send(model_input)
                .map_err(|_| AtomaInputManagerError::SendPromptError)?;
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
    #[error("Failed to build IPFS client: `{0}`")]
    FailedToBuildIpfsClient(String),
    #[error("Firebase authentication error: `{0}`")]
    FirebaseAuthError(#[from] atoma_helpers::FirebaseAuthError),
    #[error("Url error: `{0}`")]
    UrlError(String),
    #[error("Url parse error: `{0}`")]
    UrlParseError(#[from] url::ParseError),
    #[error("IPFS error: `{0}`")]
    IpfsError(String),
    #[error("Error sending prompt to the model")]
    SendPromptError,
    #[error("Timeout error, could not get input from firebase in time")]
    TimeoutError,
}
