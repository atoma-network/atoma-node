use std::sync::Arc;

use atoma_types::Request;
use axum::{http::StatusCode, routing::post, Extension, Json, Router};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, instrument};

use crate::model_thread::{ThreadRequest, ThreadResponse};

pub type RequestSender = mpsc::Sender<(ThreadRequest, oneshot::Sender<ThreadResponse>)>;

pub async fn run(sender: RequestSender, port: u64) {
    let (shutdown_signal_sender, mut shutdown_signal_receiver) = mpsc::channel::<()>(1);
    let app = Router::new()
        .route("/", post(jrpc_call))
        .route("/healthz", post(healthz))
        .layer(Extension(Arc::new(sender)))
        .layer(Extension(Arc::new(shutdown_signal_sender)));

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move { shutdown_signal_receiver.recv().await.unwrap() })
        .await
        .unwrap();
}

async fn healthz() -> Json<Value> {
    Json(json!({
        "status": "ok"
    }))
}

async fn jrpc_call(
    Extension(sender): Extension<Arc<RequestSender>>,
    Extension(shutdown): Extension<Arc<mpsc::Sender<()>>>,
    Json(input): Json<Value>,
) -> Json<Value> {
    match inner_jrpc_call(sender, input, shutdown).await {
        Ok(response) => Json(json!({
                "result":response
        })),
        Err(err) => Json(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
                "message": err
            }
        })),
    }
}

#[instrument(skip_all)]
async fn inner_jrpc_call(
    sender: Arc<RequestSender>,
    input: Value,
    shutdown: Arc<mpsc::Sender<()>>,
) -> Result<Value, String> {
    match input.get("request") {
        Some(request) => {
            let (one_sender, one_receiver) = oneshot::channel();
            info!("Sending request to model service");
            let request = match serde_json::from_value::<Request>(request.clone())
                .map_err(|e| e.to_string())
            {
                Ok(req) => req,
                Err(e) => {
                    error!("Failed to deserialize `Request`, with error: {e}");
                    return Err(e.to_string());
                }
            };
            sender
                .send((ThreadRequest::Inference(request), one_sender))
                .await
                .map_err(|e| {
                    error!("Failed to send request to Model Service");
                    e.to_string()
                })?;
            match one_receiver.await {
                Ok(response) => match response {
                    ThreadResponse::Inference(response) => Ok(response.response()),
                    _ => Err("The request failed".to_string()),
                },
                Err(e) => {
                    error!("Failed to generate response, with error: {e}");
                    Err("The request failed".to_string())
                }
            }
        }
        None => {
            info!("Shutting down JRPC server...");
            shutdown.send(()).await.unwrap();
            Ok(serde_json::Value::Null)
        }
    }
}
