use std::path::PathBuf;

use atoma_helpers::FirebaseAuth;
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
    auth: FirebaseAuth,
}

impl FirebaseOutputManager {
    /// Constructor
    pub fn new(firebase_uri: PathBuf, email: String, password: String, api_key: String) -> Self {
        Self {
            firebase_uri,
            auth: FirebaseAuth::new(email, password, api_key),
        }
    }

    /// Handles  a new post request. Encapsulates the logic necessary
    /// to post new requests, using `reqwest::Client`.
    pub async fn handle_post_request(
        &mut self,
        output_metadata: &AtomaOutputMetadata,
        output: Value,
    ) -> Result<(), AtomaOutputManagerError> {
        let client = Client::new();
        let token = self.auth.get_id_token().await?;
        let local_id = self.auth.get_local_id()?;
        let mut url = self.firebase_uri.clone();
        url.push(format!("{}.json", output_metadata.ticket_id));
        url.push(format!("?auth={token}"));
        info!("Firebase's output url: {:?}", url);
        debug!(
            "Submitting to Firebase's real time storage, with metadata: {:?}",
            output_metadata
        );
        let data = json!({
            "metadata": output_metadata,
            "data": output,
            "creatorUid": local_id,
        });
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
