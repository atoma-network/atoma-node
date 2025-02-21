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

pub static TOTAL_DIALS_ATTEMPTED: Lazy<Counter<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_counter("total_dials")
        .with_description("The total number of dials attempted")
        .with_unit("dials")
        .build()
});

pub static TOTAL_DIALS_FAILED: Lazy<Counter<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_counter("total_dials_failed")
        .with_description("The total number of dials failed")
        .with_unit("dials")
        .build()
});

pub static TOTAL_CONNECTIONS: Lazy<Gauge<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_gauge("total_connections")
        .with_description("The total number of connections")
        .with_unit("connections")
        .build()
});

pub static PEERS_CONNECTED: Lazy<Gauge<i64>> = Lazy::new(|| {
    GLOBAL_METER
        .i64_gauge("peers_connected")
        .with_description("The number of peers connected")
        .with_unit("peers")
        .build()
});

pub static TOTAL_GOSSIPSUB_SUBSCRIPTIONS: Lazy<UpDownCounter<i64>> = Lazy::new(|| {
    GLOBAL_METER
        .i64_up_down_counter("total_gossipsub_subscriptions")
        .with_description("The total number of gossipsub subscriptions")
        .with_unit("subscriptions")
        .build()
});

pub static TOTAL_VALID_GOSSIPSUB_MESSAGES_RECEIVED: Lazy<Counter<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_counter("total_valid_gossipsub_messages_received")
        .with_description("The total number of valid gossipsub messages received")
        .with_unit("messages")
        .build()
});

pub static TOTAL_GOSSIPSUB_MESSAGES_FORWARDED: Lazy<Counter<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_counter("total_gossipsub_messages_forwarded")
        .with_description("The total number of gossipsub messages forwarded")
        .with_unit("messages")
        .build()
});

pub static TOTAL_FAILED_GOSSIPSUB_MESSAGES: Lazy<Counter<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_counter("total_failed_gossipsub_messages")
        .with_description("The total number of failed gossipsub messages")
        .with_unit("messages")
        .build()
});

pub static TOTAL_INCOMING_CONNECTIONS: Lazy<Gauge<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_gauge("total_incoming_connections")
        .with_description("The total number of incoming connections")
        .with_unit("connections")
        .build()
});

pub static TOTAL_STREAM_INCOMING_BANDWIDTH: Lazy<Gauge<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_gauge("total_stream_bandwidth")
        .with_description("The total number of stream bandwidth")
        .with_unit("bytes")
        .build()
});

pub static TOTAL_STREAM_OUTGOING_BANDWIDTH: Lazy<Gauge<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_gauge("total_stream_outgoing_bandwidth")
        .with_description("The total number of stream outgoing bandwidth")
        .with_unit("bytes")
        .build()
});

pub static GOSSIP_SCORE_HISTOGRAM: Lazy<Histogram<f64>> = Lazy::new(|| {
    GLOBAL_METER
        .f64_histogram("gossip_score_histogram")
        .with_description("The histogram of gossip scores")
        .with_unit("score")
        .build()
});

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
