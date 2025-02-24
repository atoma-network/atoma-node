use chrono::Utc;
use std::{collections::HashMap, time::Duration};
use tokio::{sync::mpsc::UnboundedSender, task::JoinHandle};
use tracing::{error, instrument};

use crate::{
    broadcast_metrics::compute_node_metrics,
    errors::AtomaP2pNodeError,
    types::{NodeMessage, NodeP2pMetadata},
};

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
    country: Option<String>,
    metrics_endpoints: HashMap<String, (String, String)>,
    is_client: bool,
    node_public_url: Option<String>,
    node_small_id: Option<u64>,
    usage_metrics_tx: UnboundedSender<NodeMessage>,
) -> JoinHandle<Result<(), AtomaP2pNodeError>> {
    tokio::spawn(async move {
        // NOTE: We only publish usage metrics for nodes, clients do not need to publish any usage metrics
        if !is_client {
            if node_public_url.is_none() || node_small_id.is_none() || country.is_none() {
                error!(
                    target = "atoma-p2p",
                    event = "invalid_config",
                    "Invalid config, either public_url, node_small_id or country is not set, this should never happen"
                );
                return Err(AtomaP2pNodeError::InvalidConfig(
                    "Invalid config, either public_url, node_small_id or country is not set, this should never happen".to_string(),
                ));
            }
            let node_public_url = node_public_url.unwrap();
            let node_small_id = node_small_id.unwrap();
            let country = country.unwrap();
            loop {
                tokio::time::sleep(USAGE_METRICS_HEARTBEAT_INTERVAL).await;
                let node_metadata =
                    get_node_metadata(node_public_url.clone(), node_small_id, country.clone());
                let node_metrics = compute_node_metrics(&metrics_endpoints).await?;
                let node_message = NodeMessage {
                    node_metadata,
                    node_metrics,
                };
                usage_metrics_tx.send(node_message).map_err(|e| {
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

/// Returns metadata for the node
///
/// This metadata is used to identify the node and its location in the network
/// and is published to the network for monitoring and analytics purposes.
///
/// # Arguments
///
/// * `node_public_url` - The public URL where the node can be accessed
/// * `node_small_id` - The small ID of the node
/// * `country` - The country where the node is located
///
/// # Returns
///
/// Returns a `NodeP2pMetadata` struct containing the node's metadata
fn get_node_metadata(
    node_public_url: String,
    node_small_id: u64,
    country: String,
) -> NodeP2pMetadata {
    NodeP2pMetadata {
        node_public_url,
        node_small_id,
        country,
        timestamp: Utc::now()
            .timestamp()
            .try_into()
            .expect("Failed to convert timestamp to u64, timestamp should never be negative"),
    }
}
