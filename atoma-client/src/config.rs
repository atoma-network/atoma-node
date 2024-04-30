use std::{path::Path, time::Duration};

use atoma_types::SmallId;
use config::Config;
use serde::Deserialize;
use sui_sdk::types::base_types::ObjectID;

#[derive(Debug, Deserialize)]
pub struct AtomaSuiClientConfig {
    config_path: String,
    node_badge_id: ObjectID,
    small_id: SmallId,
    package_id: ObjectID,
    atoma_db_id: ObjectID,
    max_concurrent_requests: u64,
    request_timeout: Duration,
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

    pub fn node_badge_id(&self) -> ObjectID {
        self.node_badge_id
    }

    pub fn small_id(&self) -> SmallId {
        self.small_id
    }

    pub fn package_id(&self) -> ObjectID {
        self.package_id
    }

    pub fn atoma_db_id(&self) -> ObjectID {
        self.atoma_db_id
    }

    pub fn max_concurrent_requests(&self) -> u64 {
        self.max_concurrent_requests
    }

    pub fn request_timeout(&self) -> Duration {
        self.request_timeout
    }
}
