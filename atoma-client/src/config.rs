use std::{path::Path, time::Duration};

use config::Config;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct AtomaSuiClientConfig {
    config_path: String,
    request_timeout: Duration,
    max_concurrent_requests: u64,
}

impl AtomaSuiClientConfig {
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder().add_source(config::File::with_name(
            config_file_path.as_ref().to_str().unwrap(),
        ));
        let config = builder
            .build()
            .expect("Failed to generate Atoma Sui client configuration file");
        config
            .try_deserialize::<Self>()
            .expect("Failed to generated Atoma Sui client config file")
    }

    pub fn config_path(&self) -> String {
        self.config_path.clone()
    }

    pub fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    pub fn max_concurrent_requests(&self) -> u64 {
        self.max_concurrent_requests
    }
}
