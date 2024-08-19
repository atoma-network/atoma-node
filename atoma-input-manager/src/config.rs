use std::path::Path;

use config::Config;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
/// Atoma's Firebase configuration values
pub struct AtomaInputManagerConfig {
    /// The Atoma's Firebase authentication URL
    pub firebase_url: String,
    /// Currently we use email/password authentication for Firebase
    /// Email
    pub firebase_email: String,
    /// Password
    pub firebase_password: String,
    /// The node's Firebase api key
    pub firebase_api_key: String,
    pub small_id: u64,
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
