use std::path::Path;

use atoma_helpers::Firebase;
use atoma_types::{InputFormat, InputSource, ModelInput};
use config::AtomaInputManagerConfig;
use firebase::FirebaseInputManager;
use ipfs::IpfsInputManager;
use ipfs_api_backend_hyper::{IpfsApi, IpfsClient};
use thiserror::Error;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tracing::{error, info, instrument, trace};

mod config;
mod firebase;
mod ipfs;

type SendRequestError = mpsc::error::SendError<(
    String,
    InputFormat,
    oneshot::Sender<Result<ModelInput, AtomaInputManagerError>>,
)>;

type IpfsRequestSender = mpsc::UnboundedSender<(
    String,
    InputFormat,
    oneshot::Sender<Result<ModelInput, AtomaInputManagerError>>,
)>;

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
    input_manager_rx: mpsc::Receiver<(
        InputSource,
        oneshot::Sender<Result<ModelInput, AtomaInputManagerError>>,
    )>,
    /// A mpsc sender that sends requests to the IPFS input manager.
    ipfs_request_tx: IpfsRequestSender,
    /// The join handle to the IPFS input manager background task.
    ipfs_join_handle: Option<JoinHandle<Result<(), AtomaInputManagerError>>>,
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
        let (ipfs_request_tx, ipfs_request_rx) = mpsc::unbounded_channel();

        info!("Building IPFS client...");
        let client = IpfsClient::default();

        let ipfs_join_handle = match client.version().await {
            Ok(version) => {
                info!(
                    "IPFS client built successfully, with version = {:?}",
                    version
                );
                let ipfs_join_handle = tokio::spawn(async move {
                    let ipfs_input_manager = IpfsInputManager::new(client, ipfs_request_rx).await?;
                    ipfs_input_manager.run().await
                });
                Some(ipfs_join_handle)
            }
            Err(e) => {
                error!("Failed to obtain IPFS client's version: {}", e);
                None
            }
        };
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
            firebase_input_manager,
            input_manager_rx,
            ipfs_request_tx,
            ipfs_join_handle,
        })
    }

    /// Main loop, responsible for continuously listening to incoming user prompts.
    #[instrument(skip_all)]
    pub async fn run(mut self) -> Result<(), AtomaInputManagerError> {
        info!("Starting `AtomaInputManager` service..");
        while let Some((input_source, oneshot)) = self.input_manager_rx.recv().await {
            info!(
                "Received a new input to be submitted to a data storage {:?}..",
                input_source
            );
            match input_source {
                InputSource::Firebase { request_id } => {
                    let model_input_result = self
                        .firebase_input_manager
                        .handle_get_request(request_id)
                        .await;
                    oneshot
                        .send(model_input_result)
                        .map_err(|_| AtomaInputManagerError::SendPromptError)?;
                }
                InputSource::Ipfs { cid, format } => {
                    self.ipfs_request_tx
                        .send((cid, format, oneshot))
                        .map_err(|_| AtomaInputManagerError::SendPromptError)?;
                }
                InputSource::Raw { prompt } => {
                    oneshot
                        .send(Ok(ModelInput::Text(prompt)))
                        .map_err(|_| AtomaInputManagerError::SendPromptError)?;
                }
            }
        }
        Ok(())
    }

    /// Graceful shutdown
    #[instrument(skip_all)]
    pub async fn shutdown(self) -> Result<(), AtomaInputManagerError> {
        info!("Shutting down Atoma Input Manager...");

        trace!("Dropping IPFS request tx...");
        drop(self.ipfs_request_tx);

        trace!("Aborting IPFS manager join handle...");
        if let Some(handle) = self.ipfs_join_handle {
            handle.abort();

            trace!("Waiting for IPFS manager to join...");
            match handle.await {
                Ok(_) => Ok(()),
                Err(e) if e.is_cancelled() => Ok(()),
                Err(e) => Err(AtomaInputManagerError::JoinError(e)),
            }?;
        }

        trace!("Dropping output manager receiver...");
        drop(self.input_manager_rx);

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
    #[error("Error sending request to IPFS input manager: `{0}`")]
    SendRequestError(#[from] SendRequestError),
    #[error("Join error: `{0}`")]
    JoinError(#[from] tokio::task::JoinError),
}
