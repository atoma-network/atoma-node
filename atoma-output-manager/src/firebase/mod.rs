use atoma_helpers::{Firebase, FirebaseAuth};
use atoma_types::{AtomaOutputMetadata, OutputType};
use reqwest::Client;
use serde_json::json;
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
    auth: FirebaseAuth,
}

impl FirebaseOutputManager {
    /// Constructor
    pub async fn new(
        firebase_url: String,
        email: String,
        password: String,
        api_key: String,
        firebase: Firebase,
        node_id: u64,
    ) -> Result<Self, AtomaOutputManagerError> {
        let firebase_url = Url::parse(&firebase_url)?;
        Ok(Self {
            auth: firebase
                .add_user(email, password, api_key, &firebase_url, node_id)
                .await?,
            firebase_url,
        })
    }

    /// Handles  a new post request. Encapsulates the logic necessary
    /// to post new requests, using `reqwest::Client`.
    #[instrument(skip_all)]
    pub async fn handle_post_request(
        &mut self,
        output_metadata: &AtomaOutputMetadata,
        output: Vec<u8>,
    ) -> Result<(), AtomaOutputManagerError> {
        let client = Client::new();
        let token = self.auth.get_id_token().await?;
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
            "Submitting to Firebase's real time storage, with metadata: {:?}",
            output_metadata
        );
        let data = match output_metadata.output_type {
            OutputType::Text => {
                let output = String::from_utf8(output)?;
                json!({
                    "data": {
                        "metadata": output_metadata,
                        "data": output,
                    },
                })
            }
            OutputType::Image => {
                let mut height_bytes_buffer = [0; 4];
                height_bytes_buffer.copy_from_slice(&output[output.len() - 4..output.len()]);

                let mut width_bytes_buffer = [0; 4];
                width_bytes_buffer.copy_from_slice(&output[output.len() - 8..output.len() - 4]);

                let height = u32::from_be_bytes(height_bytes_buffer) as usize;
                let width = u32::from_be_bytes(width_bytes_buffer) as usize;

                let output = output[..output.len() - 8].to_vec();
                json!({
                    "data": {
                        "metadata": output_metadata,
                        "data": (output, height, width),
                    }
                })
            }
        };
        let response = client.put(url).json(&data).send().await?;
        let text = response.text().await?;
        info!("Received response with text: {text}");
        Ok(())
    }
}
