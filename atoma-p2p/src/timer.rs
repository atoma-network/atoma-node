use chrono::Utc;
use std::time::Duration;
use tokio::{sync::mpsc::UnboundedSender, task::JoinHandle};
use tracing::{error, instrument};

use crate::service::AtomaP2pNodeError;

/// The interval at which the node will publish usage metrics to the gossipsub topic
const USAGE_METRICS_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);

/// Spawns an asynchronous task that periodically sends usage metrics for nodes
///
/// This function creates a background task that:
/// - For nodes: Periodically collects and sends usage metrics at a fixed interval
/// - For clients: Immediately returns Ok(()) as clients don't need to publish metrics
///
/// The task runs in an infinite loop for nodes, calculating the next heartbeat time
/// and sending metrics through the provided channel.
///
/// # Arguments
///
/// * `is_client` - Boolean indicating whether this is a client (true) or node (false)
/// * `node_public_url` - The public URL of the node
/// * `node_small_id` - The small ID of the node
/// * `country` - The country where the node is located
/// * `usage_metrics_tx` - Unbounded channel sender for transmitting usage metrics
///
/// # Returns
///
/// Returns a JoinHandle that can be used to monitor or cancel the task
///
/// # Errors
///
/// Returns AtomaP2pNodeError::UsageMetricsSendError if sending metrics fails
#[instrument(
    level = "info",
    fields(
        is_client = %is_client,
        event = "usage_metrics_timer_task",
        name = "atoma-p2p", 
    ),
    skip_all
)]
pub fn usage_metrics_timer_task(
    is_client: bool,
    node_public_url: String,
    node_small_id: u64,
    country: String,
    usage_metrics_tx: UnboundedSender<NodeUsageMetrics>,
) -> JoinHandle<Result<(), AtomaP2pNodeError>> {
    tokio::spawn(async move {
        // NOTE: We only publish usage metrics for nodes, clients do not need to publish any usage metrics
        if !is_client {
            loop {
                tokio::time::sleep(USAGE_METRICS_HEARTBEAT_INTERVAL).await;
                let usage_metrics =
                    get_usage_metrics(node_public_url.clone(), node_small_id, country.clone());
                usage_metrics_tx.send(usage_metrics).map_err(|e| {
                    error!(
                        target = "atoma-p2p",
                        event = "usage_metrics_send_error",
                        error = %e,
                        "Failed to send usage metrics"
                    );
                    AtomaP2pNodeError::UsageMetricsSendError
                })?;
            }
        }
        Ok(())
    })
}

/// Represents usage metrics collected from a node in the network
///
/// This struct contains essential information about a node's status and location
/// that is periodically published to the network for monitoring and analytics purposes.
pub struct NodeUsageMetrics {
    /// The public URL where the node can be accessed
    /// This is typically the endpoint that other nodes and clients use to connect
    pub node_public_url: String,

    /// A unique identifier for the node (smaller representation of the full ID)
    /// Used for quick identification and reference in the network
    pub node_small_id: u64,

    /// The country where the node is located (ISO country code)
    /// Used for geographical distribution analysis and latency optimization
    pub country: String,

    /// Unix timestamp indicating when the metrics were collected
    /// Helps track the freshness of the metrics and synchronize data across nodes
    pub timestamp: u64,
}

/// Returns the usage metrics for the node
fn get_usage_metrics(
    node_public_url: String,
    node_small_id: u64,
    country: String,
) -> NodeUsageMetrics {
    NodeUsageMetrics {
        node_public_url,
        node_small_id,
        country,
        timestamp: Utc::now()
            .timestamp()
            .try_into()
            .expect("Failed to convert timestamp to u64, timestamp should never be negative"),
    }
}
