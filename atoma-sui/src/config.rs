use std::{path::Path, time::Duration};

use config::Config;
use serde::{Deserialize, Serialize};
use sui_sdk::types::base_types::ObjectID;

/// Configuration for the Sui Event Subscriber
///
/// This struct holds the necessary configuration parameters for connecting to and
/// interacting with a Sui network, including URLs, package ID, timeout, and small IDs.
#[derive(Debug, Deserialize, Serialize)]
pub struct SuiEventSubscriberConfig {
    /// The HTTP URL for a Sui RPC node, to which the subscriber will connect
    /// This is used for making HTTP requests to the Sui RPC node
    http_rpc_node_addr: String,

    /// The WebSocket URL for a Sui RPC node, to which the subscriber will connect
    /// This is used for establishing WebSocket connections for real-time events
    ws_rpc_node_addr: String,

    /// The Atoma's package ID on the Sui network
    /// This identifies the specific package (smart contract) to interact with
    package_id: ObjectID,

    /// The timeout duration for requests
    /// This sets the maximum time to wait for a response from the Sui network
    request_timeout: Duration,

    /// Optional value to limit the number of dynamic fields to be retrieved for each iteration
    /// of the event subscriber loop
    limit: Option<usize>,

    /// A list of node small IDs
    /// These are values used to identify the Atoma's nodes that are under control by
    /// current Sui wallet
    small_ids: Vec<u64>,
}

impl SuiEventSubscriberConfig {
    /// Constructor
    pub fn new(
        http_rpc_node_addr: String,
        ws_rpc_node_addr: String,
        package_id: ObjectID,
        request_timeout: Duration,
        limit: Option<usize>,
        small_ids: Vec<u64>,
    ) -> Self {
        Self {
            http_rpc_node_addr,
            ws_rpc_node_addr,
            package_id,
            request_timeout,
            limit,
            small_ids,
        }
    }

    /// Getter for `http_url`
    pub fn http_rpc_node_addr(&self) -> String {
        self.http_rpc_node_addr.clone()
    }

    /// Getter for `ws_url`
    pub fn ws_rpc_node_addr(&self) -> String {
        self.ws_rpc_node_addr.clone()
    }

    /// Getter for `limit`
    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Getter for `package_id`
    pub fn package_id(&self) -> ObjectID {
        self.package_id
    }

    /// Getter for `request_timeout`
    pub fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    /// Getter for `small_id`
    pub fn small_ids(&self) -> Vec<u64> {
        self.small_ids.clone()
    }

    /// Constructs a new `SuiEventSubscriberConfig` instance from a configuration file path.
    ///
    /// # Arguments
    ///
    /// * `config_file_path` - A path-like object representing the location of the configuration file.
    ///
    /// # Returns
    ///
    /// Returns a new `SuiEventSubscriberConfig` instance populated with values from the configuration file.
    ///
    /// # Panics
    ///
    /// This method will panic if:
    /// - The configuration file cannot be read or parsed.
    /// - The "event_subscriber" section is missing from the configuration file.
    /// - The configuration values cannot be deserialized into a `SuiEventSubscriberConfig` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use atoma_sui::config::SuiEventSubscriberConfig;
    /// use std::path::Path;
    ///
    /// let config = SuiEventSubscriberConfig::from_file_path("config.toml");
    /// ```
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder().add_source(config::File::with_name(
            config_file_path.as_ref().to_str().unwrap(),
        ));
        let config = builder
            .build()
            .expect("Failed to generate inference configuration file");
        config
            .get::<Self>("event_subscriber")
            .expect("Failed to generated config file")
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config = SuiEventSubscriberConfig::new(
            "".to_string(),
            "".to_string(),
            "0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e"
                .parse()
                .unwrap(),
            Duration::from_secs(5 * 60),
            Some(10),
            vec![0, 1, 2],
        );

        let toml_str = toml::to_string(&config).unwrap();
        let should_be_toml_str = "http_url = \"\"\nws_url = \"\"\npackage_id = \"0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e\"\nlimit = 10\nsmall_ids = [0, 1, 2]\n\n[request_timeout]\nsecs = 300\nnanos = 0\n";
        assert_eq!(toml_str, should_be_toml_str);
    }
}
