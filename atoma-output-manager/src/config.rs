use std::path::Path;

use config::Config;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
/// Atoma's Firebase configuration values
pub struct AtomaOutputManagerConfig {
    /// The Atoma's Firebase authentication URL
    pub firebase_url: String,
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
    /// The node's IPFS api url
    pub ipfs_api_url: String,
    /// The node's IPFS username
    pub ipfs_username: String,
    /// The node's IPFS password
    pub ipfs_password: String,
    /// The node's small id
    pub small_id: u64,
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
