use std::path::{Path, PathBuf};

use config::Config;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
/// Atoma's Firebase configuration values
pub struct AtomaOutputManagerConfig {
    /// The Atoma's Firebase authentication URI
    pub firebase_uri: PathBuf,
    /// Currently we use email/password authentication for Firebase
    /// Email
    pub firebase_email: String,
    /// Password
    pub firebase_password: String,
    /// The node's Firebase api key
    pub firebase_api_key: String,
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
            .expect("Failed to generate Atoma Sui client configuration file");
        config
            .get::<Self>("output_manager")
            .expect("Failed to generated Atoma Sui client config file")
    }
}
