use std::{collections::HashMap, sync::Arc};

use atoma_helpers::{Firebase, FirebaseAuth};
use atoma_types::AtomaStreamingData;
use reqwest::{Client, Url};
use serde_json::json;
use thiserror::Error;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, info, instrument};

/// `AtomaStreamer` instance
pub struct AtomaStreamer {
    /// Firebase url
    firebase_url: Url,
    /// A `mpsc::Receiver` channel, listening to newly
    /// AI generated outputs
    streamer_rx: mpsc::Receiver<AtomaStreamingData>,
    /// Last streamed index mapping, for each
    /// `Digest`
    last_streamed_index: HashMap<String, usize>,
    /// Firebase authentication
    auth: Arc<Mutex<FirebaseAuth>>,
}

impl AtomaStreamer {
    /// Creates a new `AtomaStreamer` instance from a firebase instance
    pub async fn new(
        streamer_rx: mpsc::Receiver<AtomaStreamingData>,
        firebase: Firebase,
    ) -> Result<Self, AtomaStreamerError> {
        Ok(Self {
            firebase_url: firebase.get_realtime_db_url(),
            streamer_rx,
            last_streamed_index: HashMap::new(),
            auth: firebase.get_auth(),
        })
    }

    /// Runs main loop for `AtomaStreamer`
    #[instrument(skip_all)]
    pub async fn run(mut self) -> Result<(), AtomaStreamerError> {
        info!("Starting firebase service..");
        while let Some(streaming_data) = self.streamer_rx.recv().await {
            info!("Received a new output to be submitted to Firebase..");
            self.handle_streaming_request(
                streaming_data.output_source_id().clone(),
                streaming_data.data().clone(),
            )
            .await?;
        }

        Ok(())
    }
}

impl AtomaStreamer {
    /// Handles new streaming request
    #[instrument(skip_all)]
    async fn handle_streaming_request(
        &mut self,
        request_id: String,
        data: String,
    ) -> Result<(), AtomaStreamerError> {
        let client = Client::new();
        let mut url = self.firebase_url.clone();
        let token = self.auth.lock().await.get_id_token().await?;
        {
            let mut path_segment = url
                .path_segments_mut()
                .map_err(|_| AtomaStreamerError::UrlError("URL is cannot-be-a-base".to_string()))?;
            path_segment.push("data");
            path_segment.push(&request_id);
            path_segment.push("response.json");
        }
        url.set_query(Some(&format!("auth={token}")));
        info!("Firebase's output url: {:?}", url);
        debug!(
            "Submitting to Firebase's real time storage, the data: {}",
            data
        );

        let last_streamed_index = self.last_streamed_index.entry(request_id).or_insert(0);
        let index = last_streamed_index.to_string();
        tokio::spawn(async move {
            let response = client
                .patch(url)
                .json(&json!({index: data}))
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
