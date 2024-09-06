use std::path::Path;

use atoma_helpers::Firebase;
use atoma_types::{AtomaOutputMetadata, OutputDestination};
use config::AtomaOutputManagerConfig;
use firebase::FirebaseOutputManager;
use gateway::GatewayOutputManager;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{info, instrument};

mod config;
mod firebase;
mod gateway;

/// `AtomaOutputManager` - manages different output destination
///     requests, allowing for a flexible interaction between
///     the user and the Atoma Network.
///
/// Current available options for storing the output on behalf
/// of the user consists of Firebase and the Gateway protocol.
pub struct AtomaOutputManager {
    /// Firebase's output manager instance.
    firebase_output_manager: FirebaseOutputManager,
    /// Gateway's output manager.
    gateway_output_manager: GatewayOutputManager,
    /// A mpsc receiver that receives tuples of `AtomaOutputMetadata` and
    /// the actual AI generated output, in JSON format.
    output_manager_rx: mpsc::Receiver<(AtomaOutputMetadata, String)>,
}

impl AtomaOutputManager {
    /// Constructor
    pub async fn new<P: AsRef<Path>>(
        config_file_path: P,
        output_manager_rx: mpsc::Receiver<(AtomaOutputMetadata, String)>,
        firebase: Firebase,
    ) -> Result<Self, AtomaOutputManagerError> {
        let config = AtomaOutputManagerConfig::from_file_path(config_file_path);
        let firebase_output_manager =
            FirebaseOutputManager::new(firebase);
        let gateway_output_manager =
            GatewayOutputManager::new(&config.gateway_api_key, &config.gateway_bearer_token);
        Ok(Self {
            firebase_output_manager,
            gateway_output_manager,
            output_manager_rx,
        })
    }

    /// Main loop, responsible for continuously listening to incoming
    /// AI generated outputs, together with corresponding metadata
    #[instrument(skip_all)]
    pub async fn run(mut self) -> Result<(), AtomaOutputManagerError> {
        info!("Starting firebase service..");
        while let Some((ref output_metadata, output)) = self.output_manager_rx.recv().await {
            info!(
                "Received a new output to be submitted to a data storage {:?}..",
                output_metadata.output_destination
            );
            match output_metadata.output_destination {
                OutputDestination::Firebase { .. } => {
                    self.firebase_output_manager
                        .handle_post_request(output_metadata, output)
                        .await?
                }
                OutputDestination::Gateway { .. } => {
                    self.gateway_output_manager
                        .handle_request(output_metadata, output)
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
    #[error("Firebase authentication error: `{0}`")]
    FirebaseAuthError(#[from] atoma_helpers::FirebaseAuthError),
    #[error("Url error: `{0}`")]
    UrlError(String),
    #[error("Url parse error: `{0}`")]
    UrlParseError(#[from] url::ParseError),
}
