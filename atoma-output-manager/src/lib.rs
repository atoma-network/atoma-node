use std::path::PathBuf;

use atoma_types::{Digest, Response};
use reqwest::Client;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{error, debug, info};

pub struct AtomaOutputManager {
    firebase_uri: PathBuf,
    output_manager_rx: mpsc::Receiver<(Digest, Response)>,
}

impl AtomaOutputManager {
    pub fn new(
        firebase_uri: PathBuf,
        output_manager_rx: mpsc::Receiver<(Digest, Response)>,
    ) -> Self {
        Self {
            firebase_uri,
            output_manager_rx,
        }
    }

    pub async fn run(mut self) -> Result<(), AtomaOutputManagerError> {
        info!("Starting firebase service..");
        while let Some(response) = self.output_manager_rx.recv().await {
            info!("Received a new output to be submitted to Firebase..");
            let tx_digest = response.0;
            let response = response.1;
            let data = serde_json::to_value(response)?;
            self.handle_post_request(tx_digest, data).await?;
        }

        Ok(())
    }
}

impl AtomaOutputManager {
    async fn handle_post_request(
        &self,
        tx_digest: Digest,
        data: serde_json::Value,
    ) -> Result<(), AtomaOutputManagerError> {
        let client = Client::new();
        let mut url = self.firebase_uri.clone();
        let mut suffix = hex::encode(tx_digest);
        suffix.push_str(".json");
        url.push(suffix);
        info!("Firebase's output url: {:?}", url);
        debug!("Submitting to Firebase's real time storage, the data: {}", data);
        let response = client
            .post(url.to_str().unwrap())
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
}
