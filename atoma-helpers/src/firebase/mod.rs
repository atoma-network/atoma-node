mod auth;
mod config;
use atoma_types::SmallId;
pub use auth::*;
pub use config::*;
use reqwest::{Client, Url};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct Firebase {
    auth: Arc<Mutex<FirebaseAuth>>,
    realtime_db_url: Url,
    storage_url: Url,
}

impl Firebase {
    pub async fn new(
        api_key: String,
        realtime_db_url: Url,
        storage_url: Url,
        node_id: SmallId,
    ) -> Result<Self, FirebaseAuthError> {
        let mut auth = FirebaseAuth::new(api_key).await?;
        let client = Client::new();
        let token = auth.get_id_token().await?;
        let mut add_node_url = realtime_db_url.clone();
        {
            let mut path_segment = add_node_url.path_segments_mut().unwrap();
            path_segment.push("nodes");
            path_segment.push(&format!("{}.json", auth.get_local_id()?));
        }
        add_node_url.set_query(Some(&format!("auth={token}")));
        let data = json!({
            "id":node_id.to_string()
        });
        client.put(add_node_url).json(&data).send().await?;

        Ok(Self {
            auth: Arc::new(Mutex::new(auth)),
            realtime_db_url,
            storage_url,
        })
    }

    pub fn get_auth(&self) -> Arc<Mutex<FirebaseAuth>> {
        Arc::clone(&self.auth)
    }

    pub fn get_realtime_db_url(&self) -> Url {
        self.realtime_db_url.clone()
    }

    pub fn get_storage_url(&self) -> Url {
        self.storage_url.clone()
    }
}
