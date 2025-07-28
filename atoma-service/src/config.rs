use std::{collections::HashMap, path::Path};

use config::{Config, File};
use serde::Deserialize;

/// Configuration for the Atoma Service.
///
/// This struct holds the configuration options for the Atoma Service,
/// including URLs for various services and a list of models.
#[derive(Debug, Deserialize)]
pub struct AtomaServiceConfig {
    /// URL for the chat completions service with maximum concurrency settings.
    ///
    /// This field specifies the endpoint for the chat completions service used by the Atoma Service.
    pub chat_completions_service_urls: HashMap<String, Vec<(String, String, usize)>>,

    /// URL for the embeddings service.
    ///
    /// This is an optional field that, if provided, specifies the endpoint
    /// for the embeddings service used by the Atoma Service.
    pub embeddings_service_url: Option<String>,

    /// URL for the image generations service.
    ///
    /// This is an optional field that, if provided, specifies the endpoint
    /// for the image generations service used by the Atoma Service.
    pub image_generations_service_url: Option<String>,

    /// List of model names.
    ///
    /// This field contains a list of model names that are deployed by the Atoma Service,
    /// on behalf of the node.
    pub models: Vec<String>,

    /// List of model revisions.
    ///
    /// This field contains a list of the associated model revisions, for each
    /// model that is currently supported by the Atoma Service.
    pub revisions: Vec<String>,

    /// Bind address for the Atoma Service.
    ///
    /// This field specifies the address and port on which the Atoma Service will bind.
    pub service_bind_address: String,

    /// The URL of the heartbeat service.
    pub heartbeat_url: String,

    /// The DSN for the Sentry service.
    pub sentry_dsn: Option<String>,

    /// The environment for the Atoma Service.
    pub environment: Option<String>,

    /// The interval for updating the metrics.
    pub metrics_update_interval: Option<u64>,

    /// List of allowed sui addresses for fiat payments.
    pub whitelist_sui_addresses_for_fiat: Vec<String>,

    /// The timeout for the too many requests error in milliseconds.
    pub too_many_requests_timeout_ms: u64,

    ///Lower threshold for memory usage, if the memory usage goes below this value, the service will not be considered overloaded
    pub memory_lower_threshold: f64,

    /// Upper threshold for memory usage, if the memory usage goes above this value, the service will be considered overloaded
    pub memory_upper_threshold: f64,

    /// The maximum number of queued requests for each inference service.
    pub max_num_queued_requests: f64,

    /// Batching interval for requests in milliseconds.
    pub requests_batcher_interval_ms: u64,

    /// The maximum number of requests that can be processed in an interval.
    pub limit_number_of_requests_per_interval: usize,

    /// Interval in milliseconds for limiting requests.
    pub limit_request_interval_ms: u64,

    /// Proxy service URL for the Atoma Service.
    pub proxy_service_url: String,
}

impl AtomaServiceConfig {
    /// Creates a new `AtomaServiceConfig` instance from a configuration file.
    ///
    /// # Arguments
    ///
    /// * `config_file_path` - Path to the configuration file. The file should be in a format
    ///   supported by the `config` crate (e.g., YAML, JSON, TOML) and contain an "atoma-service"
    ///   section with the required configuration fields.
    ///
    /// # Returns
    ///
    /// Returns a new `AtomaServiceConfig` instance populated with values from the config file.
    ///
    /// # Panics
    ///
    /// This method will panic if:
    /// * The configuration file cannot be read or parsed
    /// * The "atoma-service" section is missing from the configuration
    /// * The configuration format doesn't match the expected structure
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder()
            .add_source(File::with_name(config_file_path.as_ref().to_str().unwrap()))
            .add_source(
                config::Environment::with_prefix("ATOMA_SERVICE")
                    .keep_prefix(true)
                    .separator("__"),
            );
        let config = builder
            .build()
            .expect("Failed to generate atoma-service configuration file");
        config
            .get::<Self>("atoma_service")
            .expect("Failed to generate configuration instance")
    }
}
