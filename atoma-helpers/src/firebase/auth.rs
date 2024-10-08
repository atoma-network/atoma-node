use std::time::Instant;

use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use thiserror::Error;

/// If we are within this amount seconds of the token expiring, we will refresh it
const EXPIRATION_DELTA: usize = 10;
const SIGN_UP_URL: fn(&str) -> String = |api_key: &str| {
    format!(
        "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={}",
        api_key
    )
};
const REFRESH_URL: fn(&str) -> String = |api_key: &str| {
    format!(
        "https://securetoken.googleapis.com/v1/token?key={}",
        api_key
    )
};

/// Firebase Auth struct
pub struct FirebaseAuth {
    /// The id_token is the token we will use to authenticate with Firebase DB requests
    pub id_token: Option<String>,
    /// The refresh_token is used to get a new id_token when the current one expires
    pub refresh_token: Option<String>,
    /// The api_key is the key we use to authenticate with the Firebase Auth API
    pub api_key: String,
    /// The expires_in field is the time in seconds until the id_token expires
    pub expires_in: Option<usize>,
    /// The requested_at field is the time the id_token was last requested
    pub requested_at: Option<Instant>,
    /// This is the firebase uid, it's used to to create the createUid entry in the DB, so only this user can modify the data
    pub local_id: Option<String>,
}

#[derive(Debug, Error)]
pub enum FirebaseAuthError {
    #[error("Reqwest error: `{0}`")]
    ReqwestError(#[from] reqwest::Error),
    #[error("Parse error: `{0}`")]
    ParseError(#[from] std::num::ParseIntError),
    #[error("No local id")]
    NoLocalIdError,
}

#[derive(Debug, Deserialize)]
pub struct SignInResponse {
    /// These are not all the fields returned, but these are all the fields we need
    #[serde(rename = "idToken")]
    id_token: String,
    #[serde(rename = "refreshToken")]
    refresh_token: String,
    #[serde(rename = "expiresIn")]
    expires_in: String,
    #[serde(rename = "localId")]
    local_id: String,
}

enum Response {
    SignIn(SignInResponse),
    Refresh(RefreshResponse),
}

#[derive(Debug, Deserialize)]
pub struct RefreshResponse {
    /// These are not all the fields returned, but these are all the fields we need
    #[serde(rename = "id_token")]
    id_token: String,
    #[serde(rename = "refresh_token")]
    refresh_token: String,
    #[serde(rename = "expires_in")]
    expires_in: String,
}

impl FirebaseAuth {
    pub(crate) async fn new(api_key: String) -> Result<Self, FirebaseAuthError> {
        let mut res = Self {
            id_token: None,
            refresh_token: None,
            api_key,
            expires_in: None,
            requested_at: None,
            local_id: None,
        };
        // If we don't have an account yet, sign up. This will fail if the email is already in use, that means the password is probably wrong
        let response = res.sign_up().await?;
        res.set_from_response(response)?;
        Ok(res)
    }

    /// Sign up with email and password
    async fn sign_up(&self) -> Result<Response, FirebaseAuthError> {
        let client = Client::new();
        let url = SIGN_UP_URL(&self.api_key);
        let res = client
            .post(url)
            .json(&json!({"returnSecureToken": true}))
            .send()
            .await?;
        Ok(Response::SignIn(res.json::<SignInResponse>().await?))
    }

    // The token is about to expire (or it already has), refresh it
    async fn refresh(&mut self) -> Result<(), FirebaseAuthError> {
        let client = Client::new();
        let url = REFRESH_URL(&self.api_key);
        let res = client
            .post(url)
            .json(&json!({"grant_type": "refresh_token", "refresh_token": self.refresh_token}))
            .send()
            .await?;
        let response = if res.status().is_success() {
            Response::Refresh(res.json::<RefreshResponse>().await?)
        } else {
            // In rare occasions, the refresh token may expire, in which case we need to sign in again
            self.sign_up().await?
        };

        self.set_from_response(response)?;
        Ok(())
    }

    /// Set the fields from a firebase response
    fn set_from_response(&mut self, response: Response) -> Result<(), FirebaseAuthError> {
        match response {
            Response::SignIn(response) => {
                self.expires_in = Some(response.expires_in.parse()?);
                self.id_token = Some(response.id_token);
                self.refresh_token = Some(response.refresh_token);
                self.requested_at = Some(Instant::now());
                self.local_id = Some(response.local_id);
            }
            Response::Refresh(response) => {
                self.expires_in = Some(response.expires_in.parse()?);
                self.id_token = Some(response.id_token);
                self.refresh_token = Some(response.refresh_token);
                self.requested_at = Some(Instant::now());
            }
        };
        Ok(())
    }

    /// Get the id_token
    pub(crate) async fn get_id_token(&mut self) -> Result<String, FirebaseAuthError> {
        // If the id_token is None, we need to sign in
        if self.id_token.is_none() {
            let response = self.sign_up().await?;
            self.set_from_response(response)?;
        }
        // If the id_token is about to expire, we need to refresh it
        if self.requested_at.unwrap().elapsed().as_secs() as usize
            >= self.expires_in.unwrap() - EXPIRATION_DELTA
        {
            self.refresh().await?;
        }
        // Return the id_token that is valid at least `EXPIRATION_DELTA` seconds
        Ok(self.id_token.clone().unwrap())
    }

    /// Get the `local_id` which is the Firebase uid
    pub fn get_local_id(&self) -> Result<String, FirebaseAuthError> {
        self.local_id
            .clone()
            .ok_or(FirebaseAuthError::NoLocalIdError)
    }
}
