use std::path::Path;

use config::Config;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
/// Atoma's Firebase configuration values
pub struct AtomaInputManagerConfig {
    /// The IPFS host
    pub ipfs_host: Option<String>,
    /// The IPFS port
    pub ipfs_port: u16,
}

impl AtomaInputManagerConfig {
    /// Constructs a new instance of `Self` from a configuration file path
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder().add_source(config::File::with_name(
            config_file_path.as_ref().to_str().unwrap(),
        ));
        let config = builder
            .build()
            .expect("Failed to generate Atoma Sui input manager configuration file");
        config
            .get::<Self>("input_manager")
            .expect("Failed to generated Atoma input manager config file")
    }
}
