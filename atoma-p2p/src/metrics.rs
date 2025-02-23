use nvml_wrapper::{
    enum_wrappers::device::TemperatureSensor,
    struct_wrappers::device::{MemoryInfo, Utilization},
    Nvml,
};
use serde::{Deserialize, Serialize};
use sysinfo::{Networks, System};
use thiserror::Error;
use tracing::instrument;

use once_cell::sync::Lazy;
use opentelemetry::{
    global,
    metrics::{Counter, Gauge, Histogram, Meter, UpDownCounter},
};

// Add global metrics
static GLOBAL_METER: Lazy<Meter> = Lazy::new(|| global::meter("atoma-node"));

/// Structure to store the usage metrics for the node
///
/// This data is collected from the system and the GPU
/// to be sent across the p2p network, for efficient request routing.
#[derive(Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct NodeMetrics {
    /// The CPU usage of the node
    pub cpu_usage: f32,
    /// The average frequency of the CPUs in the system
    pub cpu_frequency: u64,
    /// The amount of RAM used
    pub ram_used: u64,
    /// The total amount of RAM in the system
    pub ram_total: u64,
    /// The amount of RAM used in swap
    pub ram_swap_used: u64,
    /// The total amount of swap memory in the system
    pub ram_swap_total: u64,
    /// The number of CPUs in the system
    pub num_cpus: u32,
    /// The total number of bytes received from the network
    pub network_rx: u64,
    /// The total number of bytes transmitted to the network
    pub network_tx: u64,
    /// The number of GPUs in the system
    pub num_gpus: u32,
    /// The usage metrics for each GPU
    pub gpus: Vec<GpuMetrics>,
}

/// Structure to store the usage metrics for each GPU
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GpuMetrics {
    /// The amount of memory used by the GPU
    pub memory_used: u64,
    /// The total amount of memory on the GPU
    pub memory_total: u64,
    /// The amount of free memory on the GPU
    pub memory_free: u64,
    /// The percentage of time the GPU was reading or writing
    pub percentage_time_read_write: u32,
    /// The percentage of time the GPU was executing
    pub percentage_time_gpu_execution: u32,
    /// The temperature of the GPU in Celsius
    pub temperature: u32,
    /// The power usage of the GPU in milliwatts
    pub power_usage: u32,
}

/// Returns the usage metrics for the node
#[instrument(level = "info", target = "metrics")]
pub fn compute_usage_metrics(mut sys: System) -> Result<NodeMetrics, NodeMetricsError> {
    let nvml = Nvml::init()?;

    let device_count = nvml.device_count()?;
    let mut gpus = Vec::new();
    for i in 0..device_count {
        let device = nvml.device_by_index(i)?;
        let Utilization { gpu, memory } = device.utilization_rates()?;
        let MemoryInfo { used, total, free } = device.memory_info()?;
        let temperature = device.temperature(TemperatureSensor::Gpu)?;
        let power_usage = device.power_usage()?;
        gpus.push(GpuMetrics {
            memory_used: used,
            memory_total: total,
            memory_free: free,
            percentage_time_read_write: memory,
            percentage_time_gpu_execution: gpu,
            temperature,
            power_usage,
        });
    }

    // Refresh the system information so we can get the latest metrics
    sys.refresh_all();
    let cpu_usage = sys.global_cpu_usage();
    let cpu_frequency =
        sys.cpus().iter().map(sysinfo::Cpu::frequency).sum::<u64>() / sys.cpus().len() as u64;
    let ram_used = sys.used_memory();
    let ram_total = sys.total_memory();
    let ram_swap_used = sys.used_swap();
    let ram_swap_total = sys.total_swap();
    let num_cpus = sys.cpus().len();
    let networks = Networks::new_with_refreshed_list();
    let mut network_rx = 0;
    let mut network_tx = 0;
    for (_interface, data) in &networks {
        network_rx += data.received();
        network_tx += data.transmitted();
    }

    Ok(NodeMetrics {
        cpu_usage,
        cpu_frequency,
        ram_used,
        ram_total,
        ram_swap_used,
        ram_swap_total,
        num_cpus: u32::try_from(num_cpus)?,
        network_rx,
        network_tx,
        num_gpus: device_count,
        gpus,
    })
}

#[derive(Debug, Error)]
pub enum NodeMetricsError {
    #[error("Nvml error: {0}")]
    NvmlError(#[from] nvml_wrapper::error::NvmlError),
    #[error("Failed to convert number of CPUs to u32: {0}")]
    TryFromIntError(#[from] std::num::TryFromIntError),
}

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
pub static TOTAL_DIALS_ATTEMPTED: Lazy<Counter<u64>> = Lazy::new(|| {
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
pub static TOTAL_DIALS_FAILED: Lazy<Counter<u64>> = Lazy::new(|| {
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
pub static TOTAL_CONNECTIONS: Lazy<Gauge<u64>> = Lazy::new(|| {
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
pub static PEERS_CONNECTED: Lazy<Gauge<i64>> = Lazy::new(|| {
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
pub static TOTAL_GOSSIPSUB_SUBSCRIPTIONS: Lazy<UpDownCounter<i64>> = Lazy::new(|| {
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
pub static TOTAL_INVALID_GOSSIPSUB_MESSAGES_RECEIVED: Lazy<Counter<u64>> = Lazy::new(|| {
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
pub static TOTAL_GOSSIPSUB_MESSAGES_FORWARDED: Lazy<Counter<u64>> = Lazy::new(|| {
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
pub static TOTAL_GOSSIPSUB_PUBLISHES: Lazy<Counter<u64>> = Lazy::new(|| {
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
pub static TOTAL_FAILED_GOSSIPSUB_PUBLISHES: Lazy<Counter<u64>> = Lazy::new(|| {
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
pub static TOTAL_INCOMING_CONNECTIONS: Lazy<Gauge<u64>> = Lazy::new(|| {
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
pub static TOTAL_OUTGOING_CONNECTIONS: Lazy<Gauge<u64>> = Lazy::new(|| {
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
pub static TOTAL_MDNS_DISCOVERIES: Lazy<Counter<u64>> = Lazy::new(|| {
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
pub static TOTAL_STREAM_INCOMING_BANDWIDTH: Lazy<Gauge<u64>> = Lazy::new(|| {
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
pub static TOTAL_STREAM_OUTGOING_BANDWIDTH: Lazy<Gauge<u64>> = Lazy::new(|| {
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
pub static GOSSIP_SCORE_HISTOGRAM: Lazy<Histogram<f64>> = Lazy::new(|| {
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
pub static KAD_ROUTING_TABLE_SIZE: Lazy<Gauge<u64>> = Lazy::new(|| {
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
