use std::sync::LazyLock;
use opentelemetry::{
    global,
    metrics::{Counter, Gauge, Histogram, Meter, UpDownCounter},
};
use sysinfo::Networks;

// Add global metrics
static GLOBAL_METER: LazyLock<Meter> = LazyLock::new(|| global::meter("atoma-node"));

/// Counter metric that tracks the total number of dial attempts.
///
/// This metric counts the number of dial attempts,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_dials_attempted`
/// - Type: Counter
/// - Labels: `peer_id`
/// - Unit: requests (count)
pub static TOTAL_DIALS_ATTEMPTED: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("total_dials")
        .with_description("The total number of dials attempted")
        .with_unit("dials")
        .build()
});

/// Counter metric that tracks the total number of dial failures.
///
/// This metric counts the number of dial failures,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_dials_failed`
/// - Type: Counter
/// - Labels: `peer_id`
/// - Unit: dials (count)
pub static TOTAL_DIALS_FAILED: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("total_dials_failed")
        .with_description("The total number of dials failed")
        .with_unit("dials")
        .build()
});

/// Gauge metric that tracks the total number of connections.
///
/// This metric counts the number of connections,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_connections`
/// - Type: Gauge
/// - Labels: `peer_id`
/// - Unit: connections (count)
pub static TOTAL_CONNECTIONS: LazyLock<Gauge<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_gauge("total_connections")
        .with_description("The total number of connections")
        .with_unit("connections")
        .build()
});

/// Gauge metric that tracks the total number of peers connected.
///
/// This metric counts the number of peers connected,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_peers_connected`
/// - Type: Gauge
/// - Labels: `peer_id`
/// - Unit: peers (count)
pub static PEERS_CONNECTED: LazyLock<Gauge<i64>> = LazyLock::new(|| {
    GLOBAL_METER
        .i64_gauge("peers_connected")
        .with_description("The number of peers connected")
        .with_unit("peers")
        .build()
});

/// Counter metric that tracks the total number of gossipsub subscriptions.
///
/// This metric counts the number of gossipsub subscriptions,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_gossipsub_subscriptions`
/// - Type: `UpDownCounter`
/// - Labels: `peer_id`
/// - Unit: `subscriptions` (count)
pub static TOTAL_GOSSIPSUB_SUBSCRIPTIONS: LazyLock<UpDownCounter<i64>> = LazyLock::new(|| {
    GLOBAL_METER
        .i64_up_down_counter("total_gossipsub_subscriptions")
        .with_description("The total number of gossipsub subscriptions")
        .with_unit("subscriptions")
        .build()
});

/// Counter metric that tracks the total number of invalid gossipsub messages received.
///
/// This metric counts the number of invalid gossipsub messages received,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_invalid_gossipsub_messages_received`
/// - Type: Counter
/// - Labels: `peer_id`
/// - Unit: messages (count)
pub static TOTAL_INVALID_GOSSIPSUB_MESSAGES_RECEIVED: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("total_invalid_gossipsub_messages_received")
        .with_description("The total number of invalid gossipsub messages received")
        .with_unit("messages")
        .build()
});

/// Counter metric that tracks the total number of gossipsub messages forwarded.
///
/// This metric counts the number of gossipsub messages forwarded,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_gossipsub_messages_forwarded`
/// - Type: Counter
/// - Labels: `peer_id`
/// - Unit: messages (count)
pub static TOTAL_GOSSIPSUB_MESSAGES_FORWARDED: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("total_gossipsub_messages_forwarded")
        .with_description("The total number of gossipsub messages forwarded")
        .with_unit("messages")
        .build()
});

/// Counter metric that tracks the total number of gossipsub publishes.
///
/// This metric counts the number of gossipsub publishes,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_gossipsub_publishes`
/// - Type: Counter
/// - Labels: `peer_id`
/// - Unit: messages (count)
pub static TOTAL_GOSSIPSUB_PUBLISHES: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("total_gossipsub_publishes")
        .with_description("The total number of gossipsub publishes")
        .with_unit("messages")
        .build()
});

/// Counter metric that tracks the total number of failed gossipsub messages.
///
/// This metric counts the number of failed gossipsub messages,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_failed_gossipsub_messages`
/// - Type: Counter
/// - Labels: `peer_id`
/// - Unit: messages (count)
pub static TOTAL_FAILED_GOSSIPSUB_PUBLISHES: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("total_failed_gossipsub_messages")
        .with_description("The total number of failed gossipsub messages")
        .with_unit("messages")
        .build()
});

/// Gauge metric that tracks the total number of incoming connections.
///
/// This metric counts the number of incoming connections,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_incoming_connections`
/// - Type: Gauge
/// - Labels: `peer_id`
/// - Unit: connections (count)
pub static TOTAL_INCOMING_CONNECTIONS: LazyLock<Gauge<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_gauge("total_incoming_connections")
        .with_description("The total number of incoming connections")
        .with_unit("connections")
        .build()
});

/// Gauge metric that tracks the total number of outgoing connections.
///
/// This metric counts the number of outgoing connections,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_outgoing_connections`
/// - Type: Gauge
/// - Labels: `peer_id`
/// - Unit: connections (count)
pub static TOTAL_OUTGOING_CONNECTIONS: LazyLock<Gauge<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_gauge("total_outgoing_connections")
        .with_description("The total number of outgoing connections")
        .with_unit("connections")
        .build()
});

/// Counter metric that tracks the total number of mDNS discoveries.
///
/// This metric counts the number of mDNS discoveries,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_mdns_discoveries`
/// - Type: Counter
/// - Labels: `peer_id`
/// - Unit: discoveries (count)
pub static TOTAL_MDNS_DISCOVERIES: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("total_mdns_discoveries")
        .with_description("The total number of mDNS discoveries")
        .with_unit("discoveries")
        .build()
});

/// Gauge metric that tracks the total number of stream bandwidth.
///
/// This metric counts the number of stream bandwidth,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_stream_incoming_bandwidth`
/// - Type: Gauge
/// - Labels: `peer_id`
/// - Unit: bytes (count)
pub static TOTAL_STREAM_INCOMING_BANDWIDTH: LazyLock<Gauge<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_gauge("total_stream_bandwidth")
        .with_description("The total number of stream bandwidth")
        .with_unit("bytes")
        .build()
});

/// Gauge metric that tracks the total number of stream outgoing bandwidth.
///
/// This metric counts the number of stream outgoing bandwidth,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_stream_outgoing_bandwidth`
/// - Type: Gauge
/// - Labels: `peer_id`
/// - Unit: bytes (count)
pub static TOTAL_STREAM_OUTGOING_BANDWIDTH: LazyLock<Gauge<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_gauge("total_stream_outgoing_bandwidth")
        .with_description("The total number of stream outgoing bandwidth")
        .with_unit("bytes")
        .build()
});

/// Histogram metric that tracks the histogram of gossip scores.
///
/// This metric counts the histogram of gossip scores,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_gossip_score_histogram`
/// - Type: Histogram
/// - Labels: `peer_id`
/// - Unit: score (count)
pub static GOSSIP_SCORE_HISTOGRAM: LazyLock<Histogram<f64>> = LazyLock::new(|| {
    GLOBAL_METER
        .f64_histogram("gossip_score_histogram")
        .with_description("The histogram of gossip scores")
        .with_unit("score")
        .build()
});

/// Gauge metric that tracks the size of the Kademlia routing table.
///
/// This metric counts the size of the Kademlia routing table,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
pub static KAD_ROUTING_TABLE_SIZE: LazyLock<Gauge<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_gauge("kad_routing_table_size")
        .with_description("The size of the Kademlia routing table")
        .with_unit("size")
        .build()
});

/// Structure to store the network metrics.
///
/// This data is collected from the system
/// and is used to update the network metrics.
pub struct NetworkMetrics {
    networks: Networks,
    bytes_received: Gauge<u64>,
    bytes_transmitted: Gauge<u64>,
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        let networks = Networks::new_with_refreshed_list();

        let bytes_received = GLOBAL_METER
            .u64_gauge("total_bytes_received")
            .with_description("The total number of bytes received")
            .with_unit("bytes")
            .build();

        let bytes_transmitted = GLOBAL_METER
            .u64_gauge("total_bytes_transmitted")
            .with_description("The total number of bytes transmitted")
            .with_unit("bytes")
            .build();

        Self {
            networks,
            bytes_received,
            bytes_transmitted,
        }
    }
}

// Network metrics implementation
impl NetworkMetrics {
    pub fn update_metrics(&mut self) {
        self.networks.refresh(true);

        let total_received = self
            .networks
            .values()
            .map(sysinfo::NetworkData::total_received)
            .sum();

        let total_transmitted = self
            .networks
            .values()
            .map(sysinfo::NetworkData::total_transmitted)
            .sum();

        self.bytes_received.record(total_received, &[]); // Empty attributes array is fine since we're just recording a global metric
        self.bytes_transmitted.record(total_transmitted, &[]);
    }
}
