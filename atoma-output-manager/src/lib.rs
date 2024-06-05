use std::path::{Path, PathBuf};

use atoma_types::AtomaOutputMetadata;
use config::AtomaFirebaseConfig;
use reqwest::Client;
use serde_json::{json, Value};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

mod config;
mod gateway;

pub struct AtomaOutputManager {
    firebase_uri: PathBuf,
    firebase_auth_token: String,
    output_manager_rx: mpsc::Receiver<(AtomaOutputMetadata, Value)>,
}

impl AtomaOutputManager {
    pub fn new(
        firebase_uri: PathBuf,
        firebase_auth_token: String,
        output_manager_rx: mpsc::Receiver<(AtomaOutputMetadata, Value)>,
    ) -> Self {
        Self {
            firebase_uri,
            firebase_auth_token,
            output_manager_rx,
        }
    }

    pub fn new_from_config<P: AsRef<Path>>(
        config_path: P,
        output_manager_rx: mpsc::Receiver<(AtomaOutputMetadata, Value)>,
    ) -> Self {
        let config = AtomaFirebaseConfig::from_file_path(config_path);
        Self {
            firebase_uri: config.firebase_uri(),
            firebase_auth_token: config.firebase_auth_token(),
            output_manager_rx,
        }
    }

    pub async fn run(mut self) -> Result<(), AtomaOutputManagerError> {
        info!("Starting firebase service..");
        while let Some((metadata, output)) = self.output_manager_rx.recv().await {
            info!("Received a new output to be submitted to Firebase..");
            self.handle_post_request(metadata, output).await?;
        }

        Ok(())
    }
}

impl AtomaOutputManager {
    async fn handle_post_request(
        &self,
        output_metadata: AtomaOutputMetadata,
        output: Value,
    ) -> Result<(), AtomaOutputManagerError> {
        let client = Client::new();
        let mut url = self.firebase_uri.clone();
        url.push(format!("{}.json", output_metadata.ticket_id));
        info!("Firebase's output url: {:?}", url);
        debug!(
            "Submitting to Firebase's real time storage, with metadata: {:?}",
            output_metadata
        );
        let data = json!({
            "metadata": output_metadata,
            "data": output
        });
        let response = client
            .post(url.to_str().unwrap())
            .bearer_auth(&self.firebase_auth_token)
            .json(&data)
            .send()
            .await?;
        let text = response.text().await?;
        info!("Received response with text: {text}");
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
}
