use reqwest::{
    header::{AUTHORIZATION, CONTENT_TYPE},
    Client,
};
use serde_json::Value;
use thiserror::Error;
use url::Url;

/// Supabase auth struct
pub struct SupabaseAuth {
    /// The token is the token we will use to authenticate with Supabase requests
    pub token: String,
}

#[derive(Debug, Error)]
pub enum SupabaseAuthError {
    #[error("Reqwest error: `{0}`")]
    ReqwestError(#[from] reqwest::Error),
    #[error("Parse error: `{0}`")]
    ParseError(#[from] std::num::ParseIntError),
    #[error("No local id")]
    NoLocalIdError,
}

impl SupabaseAuth {
    pub(crate) async fn new(url: &Url, anon_key: &str) -> Result<Self, SupabaseAuthError> {
        let auth_url = format!("{url}/auth/v1/signup");
        let client = Client::new();
        let response = client
            .post(&auth_url)
            .header("apikey", anon_key)
            .header(AUTHORIZATION, format!("Bearer {anon_key}"))
            .header(CONTENT_TYPE, "application/json")
            .json(&serde_json::json!({}))
            .send()
            .await?;
        let auth_response = response.json::<Value>().await.unwrap();
        let token = auth_response.get("access_token").unwrap().as_str().unwrap();
        Ok(Self {
            token: token.to_string(),
        })
    }

    pub fn get_token(&self) -> String {
        self.token.clone()
    }
}
