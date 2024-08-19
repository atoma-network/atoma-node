mod auth;
pub use auth::*;
use reqwest::{Client, Url};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct Firebase {
    add_user_lock: Arc<Mutex<()>>,
}

impl Default for Firebase {
    fn default() -> Self {
        Self::new()
    }
}

impl Firebase {
    pub fn new() -> Self {
        Self {
            add_user_lock: Arc::new(Mutex::new(())),
        }
    }

    pub async fn add_user(
        &self,
        email: String,
        password: String,
        api_key: String,
        fireabase_url: &Url,
        node_id: u64,
    ) -> Result<FirebaseAuth, FirebaseAuthError> {
        // This will prevent multiple calls to add_user from happening at the same time. Because in case the user doesn't exists it will trigger multiple signups.
        let _guard = self.add_user_lock.lock().await;
        let mut firebase_auth = FirebaseAuth::new(email, password, api_key).await?;
        let client = Client::new();
        let token = firebase_auth.get_id_token().await?;
        let mut url = fireabase_url.clone();
        {
            let mut path_segment = url.path_segments_mut().unwrap();
            path_segment.push("users");
            path_segment.push(&format!("{}.json", firebase_auth.get_local_id()?));
        }
        url.set_query(Some(&format!("auth={token}")));
        let data = json!({
            "node":node_id.to_string()
        });
        let response = client.put(url).json(&data).send().await?;
        let text = response.text().await?;

        Ok(firebase_auth)
    }
}
