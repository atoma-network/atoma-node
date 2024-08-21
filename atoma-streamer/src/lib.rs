use std::{collections::HashMap, path::Path};

use atoma_helpers::{Firebase, FirebaseAuth};
use atoma_types::Digest;
use config::AtomaFirebaseStreamerConfig;
use reqwest::{Client, Url};
use serde_json::json;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, instrument};

mod config;

/// `AtomaStreamer` instance
pub struct AtomaStreamer {
    /// Firebase url
    firebase_url: Url,
    /// A `mpsc::Receiver` channel, listening to newly
    /// AI generated outputs
    streamer_rx: mpsc::Receiver<(Digest, String)>,
    /// Last streamed index mapping, for each
    /// `Digest`
    last_streamed_index: HashMap<Digest, usize>,
    /// Firebase authentication
    auth: FirebaseAuth,
}

impl AtomaStreamer {
    /// Constructor
    pub fn new(
        firebase_url: Url,
        streamer_rx: mpsc::Receiver<(Digest, String)>,
        auth: FirebaseAuth,
    ) -> Self {
        Self {
            firebase_url,
            streamer_rx,
            last_streamed_index: HashMap::new(),
            auth,
        }
    }

    /// Creates a new `AtomaStreamer` instance from a configuration file
    pub async fn new_from_config<P: AsRef<Path>>(
        config_path: P,
        streamer_rx: mpsc::Receiver<(Digest, String)>,
        firebase: Firebase,
    ) -> Result<Self, AtomaStreamerError> {
        let config = AtomaFirebaseStreamerConfig::from_file_path(config_path);
        Ok(Self {
            firebase_url: config.firebase_url()?,
            streamer_rx,
            last_streamed_index: HashMap::new(),
            auth: firebase
                .add_user(
                    config.firebase_email(),
                    config.firebase_password(),
                    config.firebase_api_key(),
                )
                .await?,
        })
    }

    /// Runs main loop for `AtomaStreamer`
    #[instrument(skip_all)]
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
    /// Handles new streaming request
    #[instrument(skip_all)]
    async fn handle_streaming_request(
        &mut self,
        tx_digest: Digest,
        data: String,
    ) -> Result<(), AtomaStreamerError> {
        let client = Client::new();
        let mut url = self.firebase_url.clone();
        let token = self.auth.get_id_token().await?;
        let local_id = self.auth.get_local_id()?;
        {
            let mut path_segment = url
                .path_segments_mut()
                .map_err(|_| AtomaStreamerError::UrlError("URL is cannot-be-a-base".to_string()))?;
            path_segment.push("data");
            path_segment.push(&format!("{tx_digest}.json"));
        }
        url.set_query(Some(&format!("auth={token}")));
        info!("Firebase's output url: {:?}", url);
        debug!(
            "Submitting to Firebase's real time storage, the data: {}",
            data
        );

        let last_streamed_index = self.last_streamed_index.entry(tx_digest).or_insert(0);
        let index = last_streamed_index.to_string();
        tokio::spawn(async move {
            let response = client
                .patch(url)
                .json(&json!({index: data, "creatorUid": local_id}))
                .send()
                .await
                .unwrap();
            let text = response.text().await.unwrap();
            info!("Received response with text: {text}");
        });
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
    #[error("Firebase authentication error: `{0}`")]
    FirebaseAuthError(#[from] atoma_helpers::FirebaseAuthError),
    #[error("Url error: `{0}`")]
    UrlError(String),
    #[error("Url parse error: `{0}`")]
    UrlParseError(#[from] url::ParseError),
}
