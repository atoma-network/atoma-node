use std::sync::Arc;

use axum::{extract::State, routing::post, Json, Router};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

pub type RequestSender = mpsc::Sender<(Value, oneshot::Sender<Value>)>;

pub async fn run(sender: RequestSender) {
    let app = Router::new()
        .route("/", post(jrpc_call))
        .with_state(Arc::new(sender));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn jrpc_call(
    State(sender): State<Arc<RequestSender>>,
    Json(input): Json<Value>,
) -> Json<Value> {
    match inner_jrpc_call(sender, input).await {
        Ok(response) => Json(json!({
                "result":response
        })),
        Err(err) => Json(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32600,
                "message": err
            }
        })),
    }
}

async fn inner_jrpc_call(sender: Arc<RequestSender>, input: Value) -> Result<Value, String> {
    let request = input.get("request").expect("Request not found");
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
