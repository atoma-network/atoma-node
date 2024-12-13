use config::{Config, File};
use serde::Deserialize;
use std::path::Path;

/// Configuration for the proxy server
///
/// This struct holds the configuration parameters needed to connect to proxy server.
#[derive(Debug, Deserialize)]
pub struct ProxyConfig {
    pub proxy_address: String,
    pub node_public_address: String,
    pub country: String,
}

impl ProxyConfig {
    /// Creates a new ProxyConfig instance from a configuration file
    ///
    /// # Arguments
    ///
    /// * `config_file_path` - Path to the configuration file. The file should be in a format
    ///                        supported by the `config` crate (e.g., TOML, JSON, YAML) and
    ///                        contain an "proxy_server" section with the required configuration
    ///                        parameters.
    ///
    /// # Returns
    ///
    /// Returns a new `ProxyConfig` instance populated with values from the config file.
    ///
    /// # Panics
    ///
    /// This method will panic if:
    /// * The configuration file cannot be read or parsed
    /// * The "proxy_server" section is missing from the configuration
    /// * The configuration format doesn't match the expected structure
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder()
            .add_source(File::with_name(config_file_path.as_ref().to_str().unwrap()))
            .add_source(
                config::Environment::with_prefix("PROXY_SERVER")
                    .keep_prefix(true)
                    .separator("__"),
            );
        let config = builder
            .build()
            .expect("Failed to generate atoma-daemon configuration file");
        config
            .get::<Self>("proxy_server")
            .expect("Failed to generate configuration instance")
    }
}
