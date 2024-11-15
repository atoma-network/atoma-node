use anyhow::{Context, Result};
use tokio::sync::watch;

/// Spawns a task that will automatically trigger shutdown if it encounters an error
///
/// This helper function wraps a future in a tokio task that monitors its execution.
/// If the wrapped future returns an error, it will automatically trigger a shutdown
/// signal through the provided sender.
///
/// # Arguments
///
/// * `f` - The future to execute, which must return a `Result<()>`
/// * `shutdown_sender` - A channel sender used to signal shutdown to other parts of the application
///
/// # Returns
///
/// Returns a `JoinHandle` for the spawned task
///
/// # Example
///
/// ```
/// let (shutdown_tx, shutdown_rx) = watch::channel(false);
/// let handle = spawn_with_shutdown(some_fallible_task(), shutdown_tx);
/// ```
pub fn spawn_with_shutdown<F>(
    f: F,
    shutdown_sender: watch::Sender<bool>,
) -> tokio::task::JoinHandle<Result<()>>
where
    F: std::future::Future<Output = Result<()>> + Send + 'static,
{
    tokio::task::spawn(async move {
        let res = f.await;
        if res.is_err() {
            // Only send shutdown signal if the task failed
            shutdown_sender
                .send(true)
                .context("Failed to send shutdown signal")?;
        }
        res
    })
}
