use nvml_wrapper::{
    enum_wrappers::device::TemperatureSensor,
    struct_wrappers::device::{MemoryInfo, Utilization},
    Nvml,
};
use reqwest;
use serde::{Deserialize, Serialize};
use sysinfo::{Networks, System};
use thiserror::Error;
use tracing::instrument;

use crate::constants::{PROMETHEUS_URL, QUERIES};

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
#[derive(Debug, PartialEq, Serialize, Deserialize, Default)]
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
    /// Maximum power limit in milliwatts
    pub max_power_limit: u32,
    /// Default power limit in milliwatts
    pub default_power_limit: u32,
    /// Maximum operating temperature in Celsius
    pub max_temperature: u32,
    /// Target operating temperature in Celsius
    pub energy_consumption: u64,
    /// Chat completion latency
    pub chat_completion_latency: f64,
    /// Time to first token
    pub time_to_first_token: f64,
    /// Inter token generation time
    pub inter_token_generation_time: f64,
    /// Decoding time
    pub decoding_time: f64,
    /// Image generation latency
    pub image_generation_latency: f64,
    /// Text embeddings latency
    pub text_embeddings_latency: f64,
    /// Total requests
    pub total_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
}

#[derive(Debug, Deserialize)]
struct PrometheusResponse {
    status: String,
    data: PrometheusData,
}

#[derive(Debug, Deserialize)]
struct PrometheusData {
    result: Vec<PrometheusResult>,
}

#[derive(Debug, Deserialize)]
struct PrometheusResult {
    value: (i64, String), // timestamp and value
}

#[derive(Debug, Default)]
pub struct MetricsResponse {
    pub chat_latency: f64,
    pub first_token_time: f64,
    pub inter_token_time: f64,
    pub decoding_time: f64,
    pub image_gen_latency: f64,
    pub text_emb_latency: f64,
    pub total_requests: u64,
    pub failed_requests: u64,
}

async fn get_prometheus_metrics() -> Result<MetricsResponse, NodeMetricsError> {
    async fn query_metric(client: &reqwest::Client, query: &str) -> Result<f64, NodeMetricsError> {
        let timeout = std::time::Duration::from_secs(5);
        let response: PrometheusResponse = client
            .get(format!("{PROMETHEUS_URL}/api/v1/query"))
            .query(&[("query", query)])
            .timeout(timeout)
            .send()
            .await?
            .json()
            .await?;

        if response.status != "success" {
            return Err(NodeMetricsError::PrometheusError(format!(
                "Query failed with status: {}",
                response.status
            )));
        }

        // Get the first result's value, or default to 0.0
        let value = response
            .data
            .result
            .first()
            .and_then(|r| r.value.1.parse::<f64>().ok())
            .unwrap_or(0.0);

        Ok(value)
    }

    let client = reqwest::Client::new();
    let mut metrics = MetricsResponse::default();

    // Execute queries concurrently using futures
    let results = futures::future::join_all(
        QUERIES
            .iter()
            .map(|(_, query)| query_metric(&client, query)),
    )
    .await;

    // Assign results to the appropriate fields
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    for ((name, _), result) in QUERIES.iter().zip(results.into_iter()) {
        match (*name, result) {
            ("chat_latency", Ok(val)) => metrics.chat_latency = val,
            ("first_token_time", Ok(val)) => metrics.first_token_time = val,
            ("inter_token_time", Ok(val)) => metrics.inter_token_time = val,
            ("decoding_time", Ok(val)) => metrics.decoding_time = val,
            ("image_gen_latency", Ok(val)) => metrics.image_gen_latency = val,
            ("text_emb_latency", Ok(val)) => metrics.text_emb_latency = val,
            ("total_requests", Ok(val)) => metrics.total_requests = val as u64,
            ("failed_requests", Ok(val)) => metrics.failed_requests = val as u64,
            (_, Err(e)) => {
                tracing::warn!("Failed to fetch metric {}: {}", name, e);
            }
            (_, Ok(_)) => {} // Add this catch-all for unmatched metric names
        }
    }

    Ok(metrics)
}

/// Returns the usage metrics for the node
#[instrument(level = "info", target = "metrics")]
pub async fn compute_usage_metrics(mut sys: System) -> Result<NodeMetrics, NodeMetricsError> {
    let nvml = Nvml::init()?;
    let device_count = nvml.device_count()?;
    let mut gpus = Vec::new();

    // Get metrics from Prometheus
    let metrics = get_prometheus_metrics().await?;

    for i in 0..device_count {
        let device = nvml.device_by_index(i)?;
        let Utilization { gpu, memory } = device.utilization_rates()?;
        let MemoryInfo { used, total, free } = device.memory_info()?;
        let temperature = device.temperature(TemperatureSensor::Gpu)?;
        let power_usage = device.power_usage()?;

        // Get power limits
        let max_power_limit = device.power_management_limit()?;
        let default_power_limit = device.enforced_power_limit()?;

        // Get temperature thresholds
        let max_temperature = device.temperature_threshold(
            nvml_wrapper::enum_wrappers::device::TemperatureThreshold::GpuMax,
        )?;
        let energy_consumption = device.total_energy_consumption()?;

        gpus.push(GpuMetrics {
            memory_used: used,
            memory_total: total,
            memory_free: free,
            percentage_time_read_write: memory,
            percentage_time_gpu_execution: gpu,
            temperature,
            power_usage,
            max_power_limit,
            default_power_limit,
            max_temperature,
            energy_consumption,
            chat_completion_latency: metrics.chat_latency,
            time_to_first_token: metrics.first_token_time,
            inter_token_generation_time: metrics.inter_token_time,
            decoding_time: metrics.decoding_time,
            image_generation_latency: metrics.image_gen_latency,
            text_embeddings_latency: metrics.text_emb_latency,
            total_requests: metrics.total_requests,
            failed_requests: metrics.failed_requests,
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
    #[error("Failed to fetch telemetry metrics: {0}")]
    PrometheusMetricsError(#[from] PrometheusMetricsError),
    #[error("Failed to fetch Prometheus metrics: {0}")]
    PrometheusError(String),
    #[error("Request failed: {0}")]
    RequestError(#[from] reqwest::Error),
}

#[derive(Debug, Error)]
pub enum PrometheusMetricsError {
    #[error("Failed to parse Prometheus response: {0}")]
    ParseError(#[from] serde_json::Error),
}
