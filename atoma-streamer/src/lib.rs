use std::{path::PathBuf, sync::mpsc};

use atoma_types::{Digest, Response};
use reqwest::Client;
use thiserror::Error;
use tracing::{debug, error, info};

pub struct AtomaStreamer {
    firebase_uri: PathBuf,
    streamer_rx: mpsc::Receiver<(Digest, Response)>,
}

impl AtomaStreamer {
    pub fn new(firebase_uri: PathBuf, streamer_rx: mpsc::Receiver<(Digest, Response)>) -> Self {
        Self {
            firebase_uri,
            streamer_rx,
        }
    }

    pub async fn run(self) -> Result<(), AtomaOutputManagerError> {
        info!("Starting firebase service..");
        while let Ok((tx_digest, response)) = self.streamer_rx.recv() {
            info!("Received a new output to be submitted to Firebase..");
            let data = serde_json::to_value(response)?;
            self.handle_post_request(tx_digest, data).await?;
        }

        Ok(())
    }
}

impl AtomaStreamer {
    async fn handle_post_request(
        &self,
        tx_digest: Digest,
        data: serde_json::Value,
    ) -> Result<(), AtomaOutputManagerError> {
        let client = Client::new();
        let mut url = self.firebase_uri.clone();
        url.push(format!("{tx_digest}.json"));
        info!("Firebase's output url: {:?}", url);
        debug!(
            "Submitting to Firebase's real time storage, the data: {}",
            data
        );
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
