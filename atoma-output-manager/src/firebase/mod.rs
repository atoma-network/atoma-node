use std::sync::Arc;

use atoma_helpers::{Firebase, FirebaseAuth};
use atoma_types::AtomaOutputMetadata;
use reqwest::Client;
use serde_json::json;
use tokio::sync::Mutex;
use tracing::{debug, info, instrument};
use url::Url;

use crate::AtomaOutputManagerError;

/// `FirebaseOutputManager` - Responsible for publishing
///     generated outputs to a Firebase storage. While this
///     approach consists of a centralized point of the Atoma
///     tech stack, it is fine for applications such as chat applications.
pub struct FirebaseOutputManager {
    /// The Atoma's firebase URL
    firebase_url: Url,
    auth: Arc<Mutex<FirebaseAuth>>,
}

impl FirebaseOutputManager {
    /// Constructor
    pub fn new(
        firebase: Firebase,
    ) -> Self {
        Self {
            auth: firebase.get_auth(),
            firebase_url:firebase.get_url(),
        }
    }

    /// Handles  a new post request. Encapsulates the logic necessary
    /// to post new requests, using `reqwest::Client`.
    #[instrument(skip_all)]
    pub async fn handle_post_request(
        &mut self,
        output_metadata: &AtomaOutputMetadata,
        output: String,
    ) -> Result<(), AtomaOutputManagerError> {
        let client = Client::new();
        let token = self.auth.lock().await.get_id_token().await?;
        let mut url = self.firebase_url.clone();
        {
            let mut path_segment = url
                .path_segments_mut()
                .map_err(|_| AtomaOutputManagerError::UrlError("URL is not valid".to_string()))?;
            path_segment.push("data");
            path_segment.push(&output_metadata.output_destination.request_id());
            path_segment.push("response.json");
        }
        url.set_query(Some(&format!("auth={token}")));
        info!("Firebase's output url: {:?}", url);
        debug!(
            "Submitting to Firebase's realtime database, with metadata: {:?}",
            output_metadata
        );
        let data = json!({
            "metadata": output_metadata,
            "data": output,
        });
        client.put(url).json(&data).send().await?;
        if output_metadata.tokens.len() > 0 {
            let mut url = self.firebase_url.clone();
            {
                let mut path_segment = url.path_segments_mut().map_err(|_| {
                    AtomaOutputManagerError::UrlError("URL is not valid".to_string())
                })?;
                path_segment.push("data");
                path_segment.push(&output_metadata.output_destination.request_id());
                path_segment.push("tokens.json");
            }
            url.set_query(Some(&format!("auth={token}")));
            let data = json!(output_metadata.tokens);
            client.put(url).json(&data).send().await?;
        }
        Ok(())
    }
}
