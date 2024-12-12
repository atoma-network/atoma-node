use config::{Config, File};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// Configuration settings for a P2P Atoma Node.
///
/// This struct holds timing-related configuration parameters that control
/// the behavior of peer-to-peer connections in an Atoma node.

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct P2pAtomaNodeConfig {
    /// The address to listen on for incoming tcp connections.
    ///
    /// This is the address that the node will use to listen for incoming connections.
    /// It is a string in the format of "/ip4/x.x.x.x/tcp/x".
    pub listen_addr: String,

    /// The interval at which heartbeat messages are sent to peers.
    ///
    /// Heartbeats are used to verify that connections are still alive and
    /// to maintain the connection state with peers.
    pub heartbeat_interval: Duration,

    /// The maximum duration a connection can remain idle before it is closed.
    ///
    /// If no messages are exchanged within this duration, the connection
    /// will be terminated to free up resources.
    pub idle_connection_timeout: Duration,
}

impl P2pAtomaNodeConfig {
    /// Creates a new `P2pAtomaNodeConfig` instance from a configuration file.
    ///
    /// This method loads configuration settings from both a file and environment variables:
    /// - File: Reads the specified configuration file
    /// - Environment: Reads variables prefixed with `ATOMA_P2P__`
    ///
    /// # Arguments
    ///
    /// * `config_file_path` - Path to the configuration file
    ///
    /// # Returns
    ///
    /// Returns a new `P2pAtomaNodeConfig` instance with the loaded configuration.
    ///
    /// # Panics
    ///
    /// This method will panic if:
    /// - The configuration file cannot be read or parsed
    /// - Required configuration values are missing
    /// - The configuration format is invalid
    ///
    /// # Example
    ///
    /// ```
    /// use atoma_p2p::config::P2pAtomaNodeConfig;
    ///
    /// let config = P2pAtomaNodeConfig::from_file_path("config/atoma.toml");
    /// ```
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder()
            .add_source(File::with_name(config_file_path.as_ref().to_str().unwrap()))
            .add_source(
                config::Environment::with_prefix("ATOMA_P2P")
                    .keep_prefix(true)
                    .separator("__"),
            );
        let config = builder
            .build()
            .expect("Failed to generate atoma-p2p configuration file");
        config
            .get::<Self>("atoma_p2p")
            .expect("Failed to generate configuration instance")
    }
}
