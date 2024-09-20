use std::sync::{Arc, Mutex};

use reqwest::{
    header::{AUTHORIZATION, CONTENT_TYPE},
    Client, Response,
};
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror::Error;
use url::Url;

mod auth;
mod config;
mod tables;
pub use auth::*;
pub use config::*;
pub use tables::*;

#[derive(Clone)]
pub struct Supabase {
    auth: Arc<Mutex<SupabaseAuth>>,
    url: Url,
    anon_key: String,
}

impl Supabase {
    pub async fn new(url: Url, anon_key: String) -> Result<Self, SupabaseError> {
        let auth = SupabaseAuth::new(&url, &anon_key).await?;
        Ok(Self {
            auth: Arc::new(Mutex::new(auth)),
            url,
            anon_key,
        })
    }

    pub fn get_url(&self) -> Url {
        self.url.clone()
    }

    pub fn get_anon_key(&self) -> String {
        self.anon_key.clone()
    }

    pub async fn insert(&self, table: &str, data: &Value) -> Result<Response, SupabaseError> {
        let client = Client::new();
        let url = format!("{}/rest/v1/{table}", self.url.as_str());
        let response = client
            .post(url)
            .header(
                AUTHORIZATION,
                format!("Bearer {}", self.auth.lock().unwrap().get_token()),
            )
            .header("apikey", &self.anon_key)
            .header("Prefer", "return=representation")
            .json(data)
            .send()
            .await?;
        Ok(response)
    }

    pub async fn select<T: DeserializeOwned>(
        &self,
        table: &str,
        query: &str,
    ) -> Result<Vec<T>, SupabaseError> {
        let client = Client::new();
        let url = format!("{}/rest/v1/{table}?{query}", self.url.as_str());
        let response = client
            .get(url)
            .header(
                AUTHORIZATION,
                format!("Bearer {}", self.auth.lock().unwrap().get_token()),
            )
            .header("apikey", &self.anon_key)
            .header(CONTENT_TYPE, "application/json")
            .send()
            .await
            .expect("Failed to send request");
        dbg!(&response);
        Ok(response.json::<Vec<T>>().await?)
    }

    pub async fn select_single<T: DeserializeOwned>(
        &self,
        table: &str,
        query: &str,
    ) -> Result<Option<T>, SupabaseError> {
        self.select::<T>(
            table,
            // If the query is non-empty, append a "&" before the limit
            &format!("{query}{}limit=1", if query != "" { "&" } else { "" }),
        )
        .await
        .map(|mut v| v.pop())
    }
}

#[derive(Debug, Error)]
pub enum SupabaseError {
    #[error("Auth error: `{0}`")]
    AuthError(#[from] SupabaseAuthError),
    #[error("Reqwest error: `{0}`")]
    ReqwestError(#[from] reqwest::Error),
    #[error("No data found")]
    NoDataFound,
}
