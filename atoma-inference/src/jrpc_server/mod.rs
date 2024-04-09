use std::{net::Shutdown, sync::Arc};

use axum::{extract::State, http::StatusCode, routing::post, Extension, Json, Router};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

pub type RequestSender = mpsc::Sender<(Value, oneshot::Sender<Value>)>;

pub async fn run(sender: RequestSender) {
    let (shutdown_signal_sender, mut shutdown_signal_receiver) = mpsc::channel::<()>(1);
    let app = Router::new()
        .route("/", post(jrpc_call))
        .route("/healthz", post(healthz))
        .layer(Extension(Arc::new(sender)))
        .layer(Extension(Arc::new(shutdown_signal_sender)));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:21212")
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

async fn inner_jrpc_call(
    sender: Arc<RequestSender>,
    input: Value,
    shutdown: Arc<mpsc::Sender<()>>,
) -> Result<Value, String> {
    match input.get("request") {
        Some(request) => {
            let (one_sender, one_receiver) = oneshot::channel();
            sender
                .send((request.clone(), one_sender))
                .await
                .map_err(|e| e.to_string())?;
            if let Ok(response) = one_receiver.await {
                Ok(response)
            } else {
                Err("The request failed".to_string())
            }
        }
        None => {
            shutdown.send(()).await.unwrap();
            Ok(serde_json::Value::Null)
        }
    }
}
