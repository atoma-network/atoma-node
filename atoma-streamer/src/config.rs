use std::path::{Path, PathBuf};

use config::Config;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct AtomaFirebaseStreamerConfig {
    firebase_uri: String,
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
            .expect("Failed to generated Atoma Sui client config file")
    }

    pub fn firebase_uri(&self) -> PathBuf {
        self.firebase_uri.clone().into()
    }
}
