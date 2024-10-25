use config::{Config, File};
use serde::Deserialize;
use std::path::Path;
/// Configuration for the Atoma daemon service
///
/// This struct holds the configuration parameters needed to run the Atoma daemon,
/// including service binding information and node badge definitions.
#[derive(Debug, Deserialize)]
pub struct AtomaDaemonConfig {
    /// The address and port where the service will listen for connections
    /// Format: "host:port" (e.g., "127.0.0.1:8080")
    pub service_bind_address: String,

    /// List of node badges, where each badge is a tuple of (badge_id, value)
    /// The badge_id is a unique identifier string and value is the associated numeric value
    pub node_badges: Vec<(String, u64)>,
}

impl AtomaDaemonConfig {
    /// Creates a new AtomaDaemonConfig instance from a configuration file
    ///
    /// # Arguments
    ///
    /// * `config_file_path` - Path to the configuration file. The file should be in a format
    ///                        supported by the `config` crate (e.g., TOML, JSON, YAML) and
    ///                        contain an "atoma-daemon" section with the required configuration
    ///                        parameters.
    ///
    /// # Returns
    ///
    /// Returns a new `AtomaDaemonConfig` instance populated with values from the config file.
    ///
    /// # Panics
    ///
    /// This method will panic if:
    /// * The configuration file cannot be read or parsed
    /// * The "atoma-daemon" section is missing from the configuration
    /// * The configuration format doesn't match the expected structure
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder()
            .add_source(File::with_name(config_file_path.as_ref().to_str().unwrap()));
        let config = builder
            .build()
            .expect("Failed to generate atoma-daemon configuration file");
        config
            .get::<Self>("atoma-daemon")
            .expect("Failed to generate configuration instance")
    }
}
