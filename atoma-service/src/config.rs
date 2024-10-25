use std::path::Path;

use config::{Config, File};
use serde::Deserialize;

/// Configuration for the Atoma Service.
///
/// This struct holds the configuration options for the Atoma Service,
/// including URLs for various services and a list of models.
#[derive(Debug, Deserialize)]
pub struct AtomaServiceConfig {
    /// URL for the inference service.
    ///
    /// This is an optional field that, if provided, specifies the endpoint
    /// for the inference service used by the Atoma Service.
    pub inference_service_url: Option<String>,

    /// URL for the embeddings service.
    ///
    /// This is an optional field that, if provided, specifies the endpoint
    /// for the embeddings service used by the Atoma Service.
    pub embeddings_service_url: Option<String>,

    /// URL for the multimodal service.
    ///
    /// This is an optional field that, if provided, specifies the endpoint
    /// for the multimodal service used by the Atoma Service.
    pub multimodal_service_url: Option<String>,

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
            .add_source(File::with_name(config_file_path.as_ref().to_str().unwrap()));
        let config = builder
            .build()
            .expect("Failed to generate atoma-service configuration file");
        config
            .get::<Self>("atoma-service")
            .expect("Failed to generate configuration instance")
    }
}