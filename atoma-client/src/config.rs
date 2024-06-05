use std::{path::Path, time::Duration};

use atoma_types::SmallId;
use config::Config;
use serde::Deserialize;
use sui_sdk::types::base_types::ObjectID;

#[derive(Debug, Deserialize)]
/// `AtomaSuiClientConfig` - Configuration parameters that are necessary for the `AtomaSuiClientConfig`
pub struct AtomaSuiClientConfig {
    /// Configuration path.
    config_path: String,
    /// Node's own badge id (obtained once the node registers on the Atoma's smart contract, on the Sui blockchain).
    node_badge_id: ObjectID,
    /// Node's unique identifier (obtained once the node registers on the Atoma's smart contract, on the Sui blockchain).
    small_id: SmallId,
    /// Atoma's smart contract package id, on the Sui blockchain.
    package_id: ObjectID,
    /// The Atoma's DB id, this is generated once, once the Atoma's smart contract is first deployed.
    /// Should be the same for every node on the network.
    atoma_db_id: ObjectID,
    /// Maximum number of concurrent requests, for the Sui client.
    max_concurrent_requests: u64,
    /// Request timeout, for the Sui client.
    request_timeout: Duration,
}

impl AtomaSuiClientConfig {
    /// Constructs a new instance of `AtomaSuiClientConfig` from a file path.
    ///
    /// It deserializes the file content into a new `AtomaSuiClientConfig` instance.
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder().add_source(config::File::with_name(
            config_file_path.as_ref().to_str().unwrap(),
        ));
        let config = builder
            .build()
            .expect("Failed to generate Atoma Sui client configuration file");
        config
            .get::<Self>("client")
            .expect("Failed to generated Atoma Sui client config file")
    }

    /// Getter for the `config_path`
    pub fn config_path(&self) -> String {
        self.config_path.clone()
    }

    /// Getter for `node_badge_id`
    pub fn node_badge_id(&self) -> ObjectID {
        self.node_badge_id
    }

    /// Getter for `small_id`
    pub fn small_id(&self) -> SmallId {
        self.small_id
    }

    /// Getter for `package_id`
    pub fn package_id(&self) -> ObjectID {
        self.package_id
    }

    /// Getter for `atoma_db_id`
    pub fn atoma_db_id(&self) -> ObjectID {
        self.atoma_db_id
    }

    /// Getter for `max_concurrent_requests`
    pub fn max_concurrent_requests(&self) -> u64 {
        self.max_concurrent_requests
    }

    /// Getter for `request_timeout`
    pub fn request_timeout(&self) -> Duration {
        self.request_timeout
    }
}
