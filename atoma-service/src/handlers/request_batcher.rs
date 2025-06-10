use std::sync::Arc;

use futures::stream::{FuturesUnordered, StreamExt};
use reqwest::Client;
use serde_json::Value;
use tokio::sync::{
    mpsc::{self, UnboundedReceiver},
    oneshot, watch, RwLock,
};
use tracing::instrument;

use crate::{error::AtomaServiceError, handlers::request_counter::RequestCounter};

// This module provides a batcher for handling inference requests.
// It allows for batching requests to a specified URL with a given payload and sending the results back to the requester.
// The batcher processes requests at a specified interval, allowing for efficient handling of multiple requests.
// It uses `tokio` for asynchronous processing and `reqwest` for HTTP requests.
// The `InferenceBatcher` struct manages the batching of requests, while the `InferenceRequest` struct represents an individual request.
// The `InferenceBatcher` can be run with a specified shutdown signal, allowing it to gracefully handle shutdowns.
#[derive(Debug)]
pub struct InferenceRequest {
    pub url: String,
    pub payload: Value,
    pub result_sender: oneshot::Sender<Result<reqwest::Response, reqwest::Error>>,
}

/// The `InferenceBatcher` struct is responsible for batching inference requests.
/// It collects requests at a specified interval and sends them to a given URL with the provided payload.
pub struct RequestsBatcher {
    /// The interval at which the batcher processes requests.
    pub interval: std::time::Duration,
    /// The receiver channel for incoming inference requests.
    pub receiver: UnboundedReceiver<InferenceRequest>,
    /// The HTTP client used to send requests.
    /// It is wrapped in an `Arc<RwLock>` to allow for concurrent access.
    client: Arc<RwLock<Client>>,
}

#[derive(Debug, thiserror::Error)]
pub enum InferenceBatcherError {
    #[error("Failed to send the result back to the requester")]
    SendResultError,
    #[error("Failed to acquire the client lock")]
    ClientLockError,
    #[error("Failed to join the batcher task")]
    JoinError(#[from] tokio::task::JoinError),
}

impl RequestsBatcher {
    /// Creates a new `InferenceBatcher` instance.
    ///
    /// # Arguments
    /// * `interval` - The duration between processing batches of requests.
    /// * `receiver` - A channel receiver for incoming `InferenceRequest` objects.
    ///
    /// # Returns
    /// A new instance of `InferenceBatcher`.
    #[must_use]
    pub fn new(
        interval: std::time::Duration,
        receiver: UnboundedReceiver<InferenceRequest>,
    ) -> Self {
        Self {
            interval,
            receiver,
            client: Arc::new(RwLock::new(Client::new())),
        }
    }

    /// Runs the inference batcher, processing requests at the specified interval.
    ///
    /// # Arguments
    /// * `shutdown_receiver` - A watch receiver that signals when the batcher should shut down.
    ///
    /// # Returns
    /// A future that runs the batcher, processing requests and sending results.
    #[instrument(level = "debug", skip_all, fields(interval = %self.interval.as_secs()))]
    pub async fn run(
        mut self,
        mut shutdown_receiver: watch::Receiver<bool>,
    ) -> Result<(), InferenceBatcherError> {
        let (interval_sender, mut interval_receiver) = mpsc::channel::<()>(1);
        let interval = self.interval;
        let mut shutdown = shutdown_receiver.clone();
        let interval_trigger_handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = tokio::time::sleep(interval) => {
                        interval_sender.send(()).await.ok();
                    },
                    _ = shutdown.changed() => {
                        // If the shutdown signal is received, we exit the loop
                        break;
                    }
                }
            }
        });
        let batcher_handle = tokio::spawn(async move {
            let mut requests = Vec::new();

            loop {
                tokio::select! {
                    Some(request) = self.receiver.recv() => {
                        requests.push(request);
                    },
                    Some(()) = interval_receiver.recv() => {
                        self.process_requests(requests).await;
                        requests = Vec::new();
                    },
                    _ = shutdown_receiver.changed() => {
                        // If the shutdown signal is received, we exit the loop
                        break;
                    }
                }
            }
        });
        // Wait for both the interval trigger and the batcher to finish
        tokio::try_join!(interval_trigger_handle, batcher_handle)?;
        Ok(())
    }

    /// Processes a batch of inference requests by sending them asynchronously.
    /// This function is called when the batcher receives a signal to process requests.
    ///
    /// # Arguments
    /// * `requests` - A mutable reference to a vector of `InferenceRequest` objects.
    ///
    /// # Returns
    /// A `Result` indicating success or failure of processing the requests.
    #[instrument(level = "debug", skip_all, fields(requests_count = requests.len()))]
    pub async fn process_requests(&mut self, requests: Vec<InferenceRequest>) {
        if !requests.is_empty() {
            let mut tasks = FuturesUnordered::new();
            for request in requests {
                let client = Arc::clone(&self.client);
                tasks.push(tokio::spawn(async move {
                    Self::send_request(client, request).await
                }));
            }
            while let Some(result) = tasks.next().await {
                match result {
                    Ok(Ok(())) => {
                        // Successfully sent the request and received the response
                    }
                    Ok(Err(e)) => {
                        tracing::error!("Failed to send request: {}", e);
                    }
                    Err(e) => {
                        // This is a JoinError from tokio::spawn if the task panicked
                        tracing::error!("Task panicked or was cancelled: {}", e);
                    }
                }
            }
        }
    }

    /// Sends a single inference request and returns the result via the provided sender.
    /// This function is intended to be used within the batcher to handle individual requests.
    ///
    /// # Arguments
    /// * `request` - The `InferenceRequest` containing the URL, payload, and a sender for the result.
    ///
    /// # Returns
    /// A `Result` indicating success or failure of the request.
    #[instrument(level = "debug", skip_all)]
    async fn send_request(
        client: Arc<RwLock<Client>>,
        request: InferenceRequest,
    ) -> Result<(), InferenceBatcherError> {
        let result = client
            .read()
            .await
            .post(&request.url)
            .json(&request.payload)
            .send()
            .await;

        request
            .result_sender
            .send(result)
            .map_err(|_| InferenceBatcherError::SendResultError)
    }
}

/// Sends an inference request to the specified URL with the given payload.
/// This function is used to send requests to the inference service and handle the response.
/// It returns a `Result` containing the response or an error if the request fails.
///
/// # Arguments
/// * `request_batcher_sender` - The sender channel for the request batcher.
/// * `url` - The URL to which the request should be sent.
/// * `path` - The path to append to the URL for the request.
/// * `payload` - The payload to send with the request, serialized as JSON.
/// * `running_num_requests` - A reference to a shared counter for tracking running requests.
/// * `payload_hash` - A hash of the payload for tracking purposes.
/// * `stack_small_id` - An optional identifier for the stack small.
/// * `endpoint` - The endpoint for which the request is being sent, used for error reporting.
///
/// # Returns
/// A `Result` containing the response from the inference service or an `AtomaServiceError` if an error occurs.
#[instrument(level = "debug", skip_all, fields(url = %url, payload_hash = ?payload_hash, stack_small_id = ?stack_small_id))]
#[allow(clippy::too_many_arguments)]
pub async fn send_and_receive_request(
    request_batcher_sender: &mpsc::UnboundedSender<InferenceRequest>,
    url: &str,
    path: &str,
    payload: &Value,
    running_num_requests: &Arc<RequestCounter>,
    payload_hash: &[u8; 32],
    stack_small_id: Option<i64>,
    endpoint: &str,
) -> Result<reqwest::Response, AtomaServiceError> {
    let (result_sender, result_receiver) = tokio::sync::oneshot::channel();
    request_batcher_sender.send(InferenceRequest {
            url: format!("{}{}", url, path),
            payload:payload.clone(),
            result_sender
        }).map_err(|e| {
                running_num_requests
                .decrement(url);
            AtomaServiceError::InternalError {
                message: format!(
                    "Error sending request to inference service, for request with payload hash: {:?}, and stack small id: {:?}, with error: {}",
                    payload_hash,
                    stack_small_id,
                    e
                ),
                endpoint: endpoint.to_string(),
            }
        })?;
    let response = result_receiver
            .await
            .map_err(|e| {
                running_num_requests
                    .decrement(url);
                AtomaServiceError::InternalError {
                    message: format!(
                        "Error receiving response from inference service, for request with payload hash: {:?}, and stack small id: {:?}, with error: {}",
                        payload_hash,
                        stack_small_id,
                        e
                    ),
                    endpoint: endpoint.to_string(),
                }
            })?.map_err(|e| {
                running_num_requests
                    .decrement(url);
                AtomaServiceError::InternalError {
                    message: format!(
                        "Error processing response from inference service, for request with payload hash: {:?}, and stack small id: {:?}, with error: {}",
                        payload_hash,
                        stack_small_id,
                        e
                    ),
                    endpoint: endpoint.to_string(),
                }
            })?;
    Ok(response)
}
