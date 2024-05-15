use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use atoma_types::Digest;
use config::AtomaFirebaseStreamerConfig;
use reqwest::Client;
use serde_json::json;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

mod config;

pub struct AtomaStreamer {
    firebase_uri: PathBuf,
    streamer_rx: mpsc::Receiver<(Digest, String)>,
    last_streamed_index: HashMap<Digest, usize>,
}

impl AtomaStreamer {
    pub fn new(firebase_uri: PathBuf, streamer_rx: mpsc::Receiver<(Digest, String)>) -> Self {
        Self {
            firebase_uri,
            streamer_rx,
            last_streamed_index: HashMap::new(),
        }
    }

    pub fn new_from_config<P: AsRef<Path>>(
        config_path: P,
        streamer_rx: mpsc::Receiver<(Digest, String)>,
    ) -> Self {
        let config = AtomaFirebaseStreamerConfig::from_file_path(config_path);
        Self {
            firebase_uri: config.firebase_uri(),
            streamer_rx,
            last_streamed_index: HashMap::new(),
        }
    }

    pub async fn run(mut self) -> Result<(), AtomaStreamerError> {
        info!("Starting firebase service..");
        while let Some((tx_digest, data)) = self.streamer_rx.recv().await {
            info!("Received a new output to be submitted to Firebase..");
            self.handle_streaming_request(tx_digest, data).await?;
        }

        Ok(())
    }
}

impl AtomaStreamer {
    async fn handle_streaming_request(
        &mut self,
        tx_digest: Digest,
        data: String,
    ) -> Result<(), AtomaStreamerError> {
        let client = Client::new();
        let mut url = self.firebase_uri.clone();
        url.push(format!("{tx_digest}.json"));
        info!("Firebase's output url: {:?}", url);
        debug!(
            "Submitting to Firebase's real time storage, the data: {}",
            data
        );

        let last_streamed_index = self.last_streamed_index.entry(tx_digest).or_insert(0);
        let response = client
            .patch(url.to_str().unwrap())
            .json(&json!({last_streamed_index.to_string(): data}))
            .send()
            .await?;
        let text = response.text().await?;
        info!("Received response with text: {text}");
        *last_streamed_index += 1;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum AtomaStreamerError {
    #[error("Deserialize JSON value error: `{0}`")]
    DeserializeError(#[from] serde_json::Error),
    #[error("Request error: `{0}`")]
    RequestError(#[from] reqwest::Error),
}
