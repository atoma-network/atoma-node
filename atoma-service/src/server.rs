use axum::{routing::get, Router};
use sqlx::SqlitePool;
use tokenizers::Tokenizer;
use tokio::{net::TcpListener, signal, sync::watch::Sender};
use tracing::info;

#[derive(Clone)]
pub struct AppState {
    pub state: SqlitePool,
    pub tokenizer: Tokenizer,
}

pub fn create_router(app_state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .with_state(app_state)
}

pub async fn health_check() -> &'static str {
    "OK"
}

pub async fn run_server(
    app_state: AppState,
    tcp_listener: TcpListener,
    shutdown_sender: Sender<bool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_router(app_state);
    let shutdown_signal = async {
        signal::ctrl_c()
            .await
            .expect("Failed to parse Ctrl+C signal");
        info!("Shutting down server...");
    };
    let server =
        axum::serve(tcp_listener, app.into_make_service()).with_graceful_shutdown(shutdown_signal);
    server.await?;

    shutdown_sender.send(true)?;

    Ok(())
}
