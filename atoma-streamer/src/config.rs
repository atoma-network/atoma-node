use std::path::{Path, PathBuf};

use config::Config;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct AtomaFirebaseStreamerConfig {
    firebase_uri: String,
    firebase_email: String,
    firebase_password: String,
    firebase_api_key: String,
}

impl AtomaFirebaseStreamerConfig {
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder().add_source(config::File::with_name(
            config_file_path.as_ref().to_str().unwrap(),
        ));
        let config = builder
            .build()
            .expect("Failed to generate Atoma Sui client configuration file");
        config
            .get::<Self>("streamer")
            .expect("Failed to generated Atoma Firebase streamer config file")
    }

    /// Get the firebase_uri from the config
    pub fn firebase_uri(&self) -> PathBuf {
        self.firebase_uri.clone().into()
    }

    /// Get the email from the config
    pub fn firebase_email(&self) -> String {
        self.firebase_email.clone()
    }

    /// Get the password from the config
    pub fn firebase_password(&self) -> String {
        self.firebase_password.clone()
    }

    /// Get the api_key from the config
    pub fn firebase_api_key(&self) -> String {
        self.firebase_api_key.clone()
    }
}
