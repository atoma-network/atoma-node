use atoma_helpers::Firebase;
use atoma_types::{AtomaOutputMetadata, OutputType};
use reqwest::{Client, StatusCode};
use serde_json::json;
use tracing::{debug, error, info, instrument};
use url::Url;

use crate::AtomaOutputManagerError;

/// `FirebaseOutputManager` - Responsible for publishing
///     generated outputs to a Firebase storage. While this
///     approach consists of a centralized point of the Atoma
///     tech stack, it is fine for applications such as chat applications.
pub struct FirebaseOutputManager {
    /// The Atoma's firebase URL
    firebase: Firebase,
}

impl FirebaseOutputManager {
    /// Constructor
    pub fn new(firebase: Firebase) -> Self {
        Self { firebase }
    }

    /// Handles  a new post request. Encapsulates the logic necessary
    /// to post new requests, using `reqwest::Client`.
    #[instrument(skip_all)]
    pub async fn handle_post_request(
        &mut self,
        output_metadata: &AtomaOutputMetadata,
        output: Vec<u8>,
        ipfs_cid: Option<String>,
    ) -> Result<(), AtomaOutputManagerError> {
        let client = Client::new();
        let token = self.firebase.get_id_token().await?;

        match output_metadata.output_type {
            OutputType::Text => {
                let mut url = self.firebase.get_realtime_db_url();
                {
                    let mut path_segment = url.path_segments_mut().map_err(|_| {
                        AtomaOutputManagerError::UrlError("URL is not valid".to_string())
                    })?;
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
                let output = String::from_utf8(output)?;
                let data = json!({
                    "metadata": output_metadata,
                    "data": output,
                });
                submit_put_request(&client, url, &data).await?;
                if !output_metadata.tokens.is_empty() {
                    let mut url = self.firebase.get_realtime_db_url();
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
                    submit_put_request(&client, url, &data).await?;
                }
            }
            OutputType::Image => {
                // First store the metadata
                let mut realtime_db_url = self.firebase.get_realtime_db_url();
                {
                    let mut path_segment = realtime_db_url.path_segments_mut().map_err(|_| {
                        AtomaOutputManagerError::UrlError("URL is not valid".to_string())
                    })?;
                    path_segment.push("data");
                    path_segment.push(&output_metadata.output_destination.request_id());
                    path_segment.push("response.json");
                }
                realtime_db_url.set_query(Some(&format!("auth={token}")));
                info!("Firebase's output url: {:?}", realtime_db_url);
                debug!(
                    "Submitting to Firebase's realtime database, with metadata: {:?}",
                    output_metadata
                );
                let data = match ipfs_cid {
                    Some(ipfs_cid) => {
                        json!({
                            "metadata": output_metadata,
                            "ipfs": ipfs_cid,
                        })
                    }
                    None => {
                        json!({
                            "metadata": output_metadata,
                        })
                    }
                };
                submit_put_request(&client, realtime_db_url, &data).await?;
                // Then store the image
                let mut storage_url = self.firebase.get_storage_url();
                storage_url.set_query(Some(&format!(
                    "name=images/{}.png",
                    output_metadata.output_destination.request_id()
                )));
                client
                    .post(storage_url)
                    .header("Content-Type", "image/png")
                    .header("Authorization", format!("Bearer {}", token))
                    .body(output)
                    .send()
                    .await?;
            }
        };
        Ok(())
    }
}

async fn submit_put_request(
    client: &Client,
    url: Url,
    data: &serde_json::Value,
) -> Result<(), AtomaOutputManagerError> {
    match client.put(url).json(data).send().await?.status() {
        StatusCode::OK => {
            info!("Output submitted to Firebase successfully");
        }
        status => {
            error!("Failed to submit to Firebase, with status: {:?}", status);
            return Err(AtomaOutputManagerError::FirebaseError(format!(
                "Failed to submit to Firebase, with status: {:?}",
                status
            )));
        }
    }
    Ok(())
}
