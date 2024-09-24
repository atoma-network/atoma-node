use atoma_helpers::Firebase;
use atoma_types::ModelInput;
use reqwest::Client;
use tracing::{info, instrument};

use crate::AtomaInputManagerError;

const NUMBER_OF_REQUESTS_TO_TRY: usize = 10;
const SLEEP_BETWEEN_REQUESTS_SEC: u64 = 1;

/// `FirebaseInputManager` - Responsible for getting the prompt from the user
pub struct FirebaseInputManager {
    /// The Atoma's firebase URL
    firebase: Firebase,
}

impl FirebaseInputManager {
    /// Constructor
    pub fn new(firebase: Firebase) -> Self {
        Self { firebase }
    }

    /// Handles  a new chat request. Encapsulates the logic necessary
    /// to post new requests, using `reqwest::Client`.
    #[instrument(skip_all)]
    pub async fn handle_chat_request(
        &mut self,
        request_id: String,
    ) -> Result<ModelInput, AtomaInputManagerError> {
        let client = Client::new();
        let token = self.firebase.get_id_token().await?;
        let mut url = self.firebase.get_realtime_db_url().clone();
        {
            let mut path_segment = url
                .path_segments_mut()
                .map_err(|_| AtomaInputManagerError::UrlError("URL is not valid".to_string()))?;
            path_segment.push("data");
            path_segment.push(&request_id);
            path_segment.push("prompt.json");
        }
        url.set_query(Some(&format!("auth={token}")));
        info!("Firebase's input url: {:?}", url);
        for _ in 0..NUMBER_OF_REQUESTS_TO_TRY {
            let response = client.get(url.clone()).send().await?;
            if response.status().is_success() {
                let text = response.text().await?;
                let json: serde_json::Value = serde_json::from_str(&text)?;
                let mut tokens = Vec::new();
                if let Some(previous_transaction) = json.get("previous_transaction") {
                    let previous_transaction = previous_transaction.as_str().unwrap();
                    // There is a previous transaction from which we can get the context tokens
                    let mut url = self.firebase.get_realtime_db_url().clone();
                    {
                        let mut path_segment = url.path_segments_mut().map_err(|_| {
                            AtomaInputManagerError::UrlError("URL is not valid".to_string())
                        })?;
                        path_segment.push("data");
                        path_segment.push(previous_transaction);
                        path_segment.push("tokens.json");
                    }
                    url.set_query(Some(&format!("auth={token}")));
                    let response = client.get(url).send().await?;
                    if response.status().is_success() {
                        let text = response.text().await?;
                        let json: serde_json::Value = serde_json::from_str(&text)?;
                        tokens = json
                            .as_array()
                            .unwrap_or(&vec![])
                            .iter()
                            .map(|value| value.as_u64().unwrap_or_default() as u32)
                            .collect();
                    }
                }
                return Ok(ModelInput::Chat((
                    json["data"].as_str().unwrap().to_string(),
                    tokens,
                )));
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(SLEEP_BETWEEN_REQUESTS_SEC)).await;
        }
        Err(AtomaInputManagerError::TimeoutError)
    }
}
