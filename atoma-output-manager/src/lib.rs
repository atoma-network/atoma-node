use std::{path::Path, string::FromUtf8Error};

use atoma_helpers::Firebase;
use atoma_types::{AtomaOutputMetadata, OutputDestination};
use config::AtomaOutputManagerConfig;
use firebase::FirebaseOutputManager;
use gateway::GatewayOutputManager;
use ipfs::IpfsOutputManager;
use thiserror::Error;
use tokio::{sync::mpsc, task::JoinHandle};
use tracing::{info, instrument};

mod config;
mod firebase;
mod gateway;
mod ipfs;

type IpfsSendRequestError = mpsc::error::SendError<(AtomaOutputMetadata, Vec<u8>)>;

/// `AtomaOutputManager` - manages different output destination
///     requests, allowing for a flexible interaction between
///     the user and the Atoma Network.
///
/// Current available options for storing the output on behalf
/// of the user consists of Firebase and the Gateway protocol.
pub struct AtomaOutputManager {
    /// Firebase's output manager instance.
    firebase_output_manager: FirebaseOutputManager,
    /// IPFS's join handle related to the IPFS output manager background task.
    ipfs_manager_join_handle: JoinHandle<Result<(), AtomaOutputManagerError>>,
    /// Request sender to the IPFS manager background task.
    ipfs_request_tx: mpsc::UnboundedSender<(AtomaOutputMetadata, Vec<u8>)>,
    /// Gateway's output manager.
    gateway_output_manager: GatewayOutputManager,
    /// A mpsc receiver that receives tuples of `AtomaOutputMetadata` and
    /// the actual AI generated output, in JSON format.
    output_manager_rx: mpsc::Receiver<(AtomaOutputMetadata, Vec<u8>)>,
}

impl AtomaOutputManager {
    /// Constructor
    pub async fn new<P: AsRef<Path>>(
        config_file_path: P,
        output_manager_rx: mpsc::Receiver<(AtomaOutputMetadata, Vec<u8>)>,
        firebase: Firebase,
    ) -> Result<Self, AtomaOutputManagerError> {
        let config = AtomaOutputManagerConfig::from_file_path(config_file_path);
        let firebase_output_manager = FirebaseOutputManager::new(
            config.firebase_url,
            config.firebase_email,
            config.firebase_password,
            config.firebase_api_key,
            firebase,
            config.small_id,
        )
        .await?;

        let gateway_output_manager =
            GatewayOutputManager::new(&config.gateway_api_key, &config.gateway_bearer_token);

        let (ipfs_request_tx, ipfs_request_rx) = mpsc::unbounded_channel();
        let ipfs_manager_join_handle = tokio::spawn(async move {
            let mut ipfs_output_manager = IpfsOutputManager::new(ipfs_request_rx).await?;
            ipfs_output_manager.run().await?;
            Ok(())
        });

        Ok(Self {
            firebase_output_manager,
            ipfs_manager_join_handle,
            ipfs_request_tx,
            gateway_output_manager,
            output_manager_rx,
        })
    }

    /// Main loop, responsible for continuously listening to incoming
    /// AI generated outputs, together with corresponding metadata
    #[instrument(skip_all)]
    pub async fn run(mut self) -> Result<(), AtomaOutputManagerError> {
        info!("Starting firebase service..");
        while let Some((output_metadata, output)) = self.output_manager_rx.recv().await {
            info!(
                "Received a new output to be submitted to a data storage {:?}..",
                output_metadata.output_destination
            );
            match output_metadata.output_destination {
                OutputDestination::Firebase { .. } => {
                    self.firebase_output_manager
                        .handle_post_request(&output_metadata, output)
                        .await?
                }
                OutputDestination::Ipfs => {
                    self.ipfs_request_tx.send((output_metadata, output))?;
                    // TODO: we need to handle the response CID
                    // either by storing it in firebase, or submitting
                    // an on-chain transaction
                }
                OutputDestination::Gateway { .. } => {
                    self.gateway_output_manager
                        .handle_request(&output_metadata, output)
                        .await?
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum AtomaOutputManagerError {
    #[error("Deserialize JSON value error: `{0}`")]
    DeserializeError(#[from] serde_json::Error),
    #[error("Request error: `{0}`")]
    RequestError(#[from] reqwest::Error),
    #[error("GraphQl error: `{0}`")]
    GraphQlError(String),
    #[error("Invalid output destination: `{0}`")]
    InvalidOutputDestination(String),
    #[error("Failed to convert output to string: `{0}`")]
    FromUtf8Error(#[from] FromUtf8Error),
    #[error("Failed to build IPFS client: `{0}`")]
    FailedToBuildIpfsClient(String),
    #[error("Firebase authentication error: `{0}`")]
    FirebaseAuthError(#[from] atoma_helpers::FirebaseAuthError),
    #[error("Url error: `{0}`")]
    UrlError(String),
    #[error("Url parse error: `{0}`")]
    UrlParseError(#[from] url::ParseError),
    #[error("IPFS send request error: `{0}`")]
    IpfsSendRequestError(#[from] IpfsSendRequestError),
    #[error("Unknown IPFS error")]
    UnknownIpfsError,
}
