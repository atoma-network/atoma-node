use atoma_helpers::{Firebase, FirebaseAuth};
use atoma_types::AtomaInputMetadata;
use reqwest::Client;
use tracing::{debug, info, instrument};
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
        input_metadata: AtomaInputMetadata,
    ) -> Result<String, AtomaInputManagerError> {
        dbg!(&input_metadata);
        let client = Client::new();
        let token = self.auth.get_id_token().await?;
        dbg!(&token);
        let mut url = self.firebase_url.clone();
        dbg!(&url);
        {
            let mut path_segment = url
                .path_segments_mut()
                .map_err(|_| AtomaInputManagerError::UrlError("URL is not valid".to_string()))?;
            path_segment.push("prompts");
            path_segment.push(&input_metadata.user_id);
            path_segment.push(&input_metadata.node_id.to_string());
            path_segment.push(&input_metadata.ticket_id);
            path_segment.push("prompt.json");
        }
        url.set_query(Some(&format!("auth={token}")));
        dbg!(&url);
        info!("Firebase's input url: {:?}", url);
        debug!(
            "Submitting to Firebase's real time storage, with metadata: {:?}",
            input_metadata
        );
        for _ in 0..NUMBER_OF_REQUESTS_TO_TRY {
            let response = client.get(url.clone()).send().await?;
            if response.status().is_success() {
                let text = response.text().await?;
                info!("Received response with text: {text}");
                return Ok(text);
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(SLEEP_BETWEEN_REQUESTS_SEC)).await;
        }
        Err(AtomaInputManagerError::TimeoutError)
    }
}
