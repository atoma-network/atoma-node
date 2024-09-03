use atoma_helpers::{Firebase, FirebaseAuth};
use atoma_types::ModelInput;
use reqwest::Client;
use tracing::{info, instrument};
use url::Url;

use crate::AtomaInputManagerError;

const NUMBER_OF_REQUESTS_TO_TRY: usize = 5;
const SLEEP_BETWEEN_REQUESTS_SEC: u64 = 1;

/// `FirebaseInputManager` - Responsible for getting the prompt from the user
pub struct FirebaseInputManager {
    /// The Atoma's firebase URL
    firebase_url: Url,
    auth: FirebaseAuth,
}

impl FirebaseInputManager {
    /// Constructor
    pub async fn new(
        firebase_url: String,
        email: String,
        password: String,
        api_key: String,
        firebase: Firebase,
        node_id: u64,
    ) -> Result<Self, AtomaInputManagerError> {
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
    pub async fn handle_get_request(
        &mut self,
        request_id: String,
    ) -> Result<ModelInput, AtomaInputManagerError> {
        let client = Client::new();
        let token = self.auth.get_id_token().await?;
        let mut url = self.firebase_url.clone();
        {
            let mut path_segment = url
                .path_segments_mut()
                .map_err(|_| AtomaInputManagerError::UrlError("URL is not valid".to_string()))?;
            path_segment.push("data");
            path_segment.push(&request_id);
            path_segment.push("prompt");
            path_segment.push("data.json");
        }
        url.set_query(Some(&format!("auth={token}")));
        info!("Firebase's input url: {:?}", url);
        for _ in 0..NUMBER_OF_REQUESTS_TO_TRY {
            let response = client.get(url.clone()).send().await?;
            if response.status().is_success() {
                let text = response.text().await?;
                info!("Received response with text: {text}");
                return Ok(ModelInput::Text(text));
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(SLEEP_BETWEEN_REQUESTS_SEC)).await;
        }
        Err(AtomaInputManagerError::TimeoutError)
    }
}
