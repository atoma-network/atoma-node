use std::{process, sync::LazyLock};

use anyhow::{Context, Result};
use opentelemetry::{global, trace::TracerProvider, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    metrics::{self as sdkmetrics},
    trace::{self as sdktrace, RandomIdGenerator, Sampler},
    Resource,
};

use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::{
    non_blocking,
    rolling::{RollingFileAppender, Rotation},
};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan, time::UtcTime},
    layer::SubscriberExt,
    prelude::*,
    util::SubscriberInitExt,
    EnvFilter, Registry,
};
use url::Url;

/// The directory where the logs are stored.
const LOGS: &str = "./logs";
/// The log file name for the node service.
const NODE_LOG_FILE: &str = "atoma-node.log";
/// The log file name for the daemon service.
const DAEMON_LOG_FILE: &str = "atoma-daemon.log";

// Default Grafana OTLP endpoint if not specified in environment
// Override this to be the localhost for local development if not using Docker
const DEFAULT_OTLP_ENDPOINT: &str = "http://otel-collector:4317";
const DEFAULT_LOKI_ENDPOINT: &str = "http://loki:3100";

static RESOURCE: LazyLock<Resource> =
    LazyLock::new(|| Resource::new(vec![KeyValue::new("service_name", "atoma-node")]));

/// Initialize metrics with OpenTelemetry SDK
fn init_metrics(otlp_endpoint: &str) -> sdkmetrics::SdkMeterProvider {
    let metrics_exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_tonic()
        .with_endpoint(otlp_endpoint)
        .build()
        .unwrap();

    let reader =
        sdkmetrics::PeriodicReader::builder(metrics_exporter, opentelemetry_sdk::runtime::Tokio)
            .with_interval(std::time::Duration::from_secs(3))
            .with_timeout(std::time::Duration::from_secs(10))
            .build();

    sdkmetrics::SdkMeterProvider::builder()
        .with_reader(reader)
        .with_resource(RESOURCE.clone())
        .build()
}

/// Initialize tracing with OpenTelemetry SDK
fn init_traces(otlp_endpoint: &str) -> Result<sdktrace::Tracer> {
    let tracing_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(otlp_endpoint)
        .build()?;

    let tracer_provider = opentelemetry_sdk::trace::TracerProvider::builder()
        .with_batch_exporter(tracing_exporter, opentelemetry_sdk::runtime::Tokio)
        .with_sampler(Sampler::AlwaysOn)
        .with_id_generator(RandomIdGenerator::default())
        .with_max_events_per_span(64)
        .with_max_attributes_per_span(16)
        .with_max_events_per_span(16)
        .with_resource(RESOURCE.clone())
        .build();

    let tracer = tracer_provider.tracer("atoma-node");
    global::set_tracer_provider(tracer_provider);

    Ok(tracer)
}

/// Configure logging with JSON formatting, file output, and console output
///
/// # Errors
///
/// Returns an error if:
/// - Failed to create logs directory
/// - Failed to initialize OpenTelemetry tracing
/// - Failed to set global default subscriber
///
/// # Panics
///
/// This function may panic if:
/// - LOKI_ENDPOINT is set but contains an invalid URL
/// - Failed to parse process ID
pub fn setup_logging() -> Result<(WorkerGuard, WorkerGuard)> {
    let otlp_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .unwrap_or_else(|_| DEFAULT_OTLP_ENDPOINT.to_string());

    let loki_endpoint =
        std::env::var("LOKI_ENDPOINT").unwrap_or_else(|_| DEFAULT_LOKI_ENDPOINT.to_string());

    let (layer, task) = tracing_loki::builder()
        .label("service_name", "atoma-node")?
        .extra_field("pid", format!("{}", process::id()))?
        .build_url(Url::parse(&loki_endpoint).unwrap())?;

    // The background task needs to be spawned so the logs actually get
    // delivered.
    tokio::spawn(task);

    // Create logs directory if it doesn't exist
    std::fs::create_dir_all(LOGS).context("Failed to create logs directory")?;

    // Set up metrics
    let metrics_provider = init_metrics(&otlp_endpoint);
    global::set_meter_provider(metrics_provider);

    // Set up file appenders with rotation for both services
    let node_appender = RollingFileAppender::new(Rotation::DAILY, LOGS, NODE_LOG_FILE);
    let daemon_appender = RollingFileAppender::new(Rotation::DAILY, LOGS, DAEMON_LOG_FILE);

    // Create non-blocking writers and keep the guards
    let (node_non_blocking, node_guard) = non_blocking(node_appender);
    let (daemon_non_blocking, daemon_guard) = non_blocking(daemon_appender);

    // Initialize OpenTelemetry tracing
    let tracer = init_traces(&otlp_endpoint)?;
    let opentelemetry_layer = OpenTelemetryLayer::new(tracer);

    let logs_exporter = opentelemetry_otlp::LogExporter::builder()
        .with_tonic()
        .with_endpoint(otlp_endpoint)
        .build()?;

    let _logger_provider = opentelemetry_sdk::logs::LoggerProvider::builder()
        .with_batch_exporter(logs_exporter, opentelemetry_sdk::runtime::Tokio)
        .with_resource(RESOURCE.clone())
        .build();

    // Create all layers
    let node_layer = fmt::layer()
        .json()
        .with_timer(UtcTime::rfc_3339())
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_target(true)
        .with_line_number(true)
        .with_file(true)
        .with_current_span(true)
        .with_span_list(true)
        .with_writer(node_non_blocking)
        .with_filter(EnvFilter::new("info,atoma_node=info,atoma_p2p=info"));

    let daemon_layer = fmt::layer()
        .json()
        .with_timer(UtcTime::rfc_3339())
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_target(true)
        .with_line_number(true)
        .with_file(true)
        .with_current_span(true)
        .with_span_list(true)
        .with_writer(daemon_non_blocking)
        .with_filter(EnvFilter::new("info,atoma_daemon=info,atoma_p2p=info"));

    let console_layer = fmt::layer()
        .pretty()
        .with_target(true)
        .with_thread_ids(true)
        .with_line_number(true)
        .with_file(true)
        .with_span_events(FmtSpan::ENTER);

    // Use a single env_filter for everything
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,atoma_daemon=info,atoma_p2p=info"));

    // Apply the filter at the registry level only
    Registry::default()
        .with(env_filter)
        .with(console_layer)
        .with(node_layer)
        .with(daemon_layer)
        .with(opentelemetry_layer)
        .with(sentry::integrations::tracing::layer())
        .with(layer)
        .try_init()
        .context("Failed to set global default subscriber")?;

    Ok((node_guard, daemon_guard))
}

/// Ensure all spans are exported before shutdown
pub fn shutdown() {
    global::shutdown_tracer_provider();
}
