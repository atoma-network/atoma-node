use std::path::PathBuf;

use atoma_types::AtomaOutputMetadata;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{debug, info};

use crate::AtomaOutputManagerError;

/// `FirebaseOutputManager` - Responsible for publishing
///     generated outputs to a Firebase storage. While this
///     approach consists of a centralized point of the Atoma
///     tech stack, it is fine for applications such as chat applications.
pub struct FirebaseOutputManager {
    /// The Atoma's firebase URI
    firebase_uri: PathBuf,
    /// The node's firebase authentication token, to be able to perform write
    /// operations on the Atoma's firebase storage
    firebase_auth_token: String,
}

impl FirebaseOutputManager {
    /// Constructor
    pub fn new(firebase_uri: PathBuf, firebase_auth_token: String) -> Self {
        Self {
            firebase_uri,
            firebase_auth_token,
        }
    }

    /// Handles  a new post request. Encapsulates the logic necessary
    /// to post new requests, using `reqwest::Client`.
    pub async fn handle_post_request(
        &self,
        output_metadata: &AtomaOutputMetadata,
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
