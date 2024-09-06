use std::path::Path;

use config::Config;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
/// Atoma's Firebase configuration values
pub struct AtomaOutputManagerConfig {
    /// The node's Firebase api key
    /// The node's Gateway api key
    pub gateway_api_key: String,
    /// The node's Gateway's bearer token
    pub gateway_bearer_token: String,
}

impl AtomaOutputManagerConfig {
    /// Constructs a new instance of `Self` from a configuration file path
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder().add_source(config::File::with_name(
            config_file_path.as_ref().to_str().unwrap(),
        ));
        let config = builder
            .build()
            .expect("Failed to generate Atoma Sui output manager configuration file");
        config
            .get::<Self>("output_manager")
            .expect("Failed to generated Atoma output manager config file")
    }
}
