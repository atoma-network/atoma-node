use reqwest::{
    header::{self, HeaderMap},
    Client, IntoUrl, Url,
};
use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

pub struct HuggingFaceClient {
    client: Client,
    endpoint: Url,
    request_id: i64,
}

impl HuggingFaceClient {
    pub fn connect<T: IntoUrl>(endpoint: T) -> Result<Self, HuggingFaceError> {
        let client = Client::builder()
            .default_headers({
                let mut headers = HeaderMap::new();
                headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
                headers
            })
            .build()?;

        Ok(Self {
            client,
            endpoint: endpoint.into_url()?,
            request_id: 0,
        })
    }

    fn next_request_id(&mut self) -> i64 {
        self.request_id += 1;
        self.request_id
    }

    pub async fn send_request<T: Serialize, R: DeserializeOwned>(
        &mut self,
        method: &str,
        params: T,
    ) -> Result<R, HuggingFaceError> {
        let params = serde_json::to_value(params)?;
        let request_json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": self.next_request_id(),
            "method": method,
            "params`": params,
        });

        let response = self
            .client
            .post(self.endpoint.clone())
            .body(request_json.to_string())
            .send()
            .await?;

        let value = response.json().await?;
        let response = extract_json_result(value)?;

        Ok(serde_json::from_value(response)?)
    }
}

#[derive(Debug, Error)]
pub enum HuggingFaceError {
    #[error("Connection error: `{0}`")]
    ConnectionError(reqwest::Error),
    #[error("Serialization error: `{0}`")]
    SerializationError(serde_json::Error),
    #[error("Failed request with code `{code}` and message `{message}`")]
    RequestError { code: i64, message: String },
    #[error("Invalid response: `{message}`")]
    InvalidResponse { message: String },
}

impl From<reqwest::Error> for HuggingFaceError {
    fn from(error: reqwest::Error) -> Self {
        Self::ConnectionError(error)
    }
}

impl From<serde_json::Error> for HuggingFaceError {
    fn from(error: serde_json::Error) -> Self {
        Self::SerializationError(error)
    }
}

fn extract_json_result(val: serde_json::Value) -> Result<serde_json::Value, HuggingFaceError> {
    if let Some(err) = val.get("error") {
        let code = err.get("code").and_then(|c| c.as_i64()).unwrap_or(-1);
        let message = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Unknown error");
        return Err(HuggingFaceError::RequestError {
            code,
            message: message.to_string(),
        });
    }

    let result = val
        .get("result")
        .ok_or_else(|| HuggingFaceError::InvalidResponse {
            message: "Missing result field".to_string(),
        })?;
    Ok(result.clone())
}
