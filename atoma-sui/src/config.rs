use std::{path::Path, time::Duration};

use config::Config;
use serde::{Deserialize, Serialize};
use sui_sdk::types::base_types::ObjectID;

/// Configuration for the Sui Event Subscriber
///
/// This struct holds the necessary configuration parameters for connecting to and
/// interacting with a Sui network, including URLs, package ID, timeout, and small IDs.
#[derive(Debug, Deserialize, Serialize)]
pub struct AtomaSuiConfig {
    /// The HTTP URL for a Sui RPC node, to which the subscriber will connect
    /// This is used for making HTTP requests to the Sui RPC node
    http_rpc_node_addr: String,

    /// The WebSocket URL for a Sui RPC node, to which the subscriber will connect
    /// This is used for establishing WebSocket connections for real-time events
    ws_rpc_node_addr: String,

    /// The Atoma's DB object ID on the Sui network
    /// This identifies the specific Atoma's DB object to interact with
    atoma_db: ObjectID,

    /// The Atoma's package ID on the Sui network
    /// This identifies the specific package (smart contract) to interact with
    atoma_package_id: ObjectID,

    /// The Atoma's TOMA token package ID on the Sui network
    /// This identifies the specific package (smart contract) to interact with
    /// for TOMA token payments
    toma_package_id: ObjectID,

    /// The timeout duration for requests
    /// This sets the maximum time to wait for a response from the Sui network
    request_timeout: Option<Duration>,

    /// The number of concurrent tasks to run
    /// This sets the maximum number of tasks to run concurrently
    num_concurrent_tasks: Option<usize>,

    /// The maximum number of concurrent requests to the Sui client
    max_concurrent_requests: Option<u64>,

    /// Optional value to limit the number of dynamic fields to be retrieved for each iteration
    /// of the event subscriber loop
    limit: Option<usize>,

    /// A list of node small IDs
    /// These are values used to identify the Atoma's nodes that are under control by
    /// current Sui wallet
    small_ids: Vec<u64>,

    /// Sui's config path
    sui_config_path: String,
}

impl AtomaSuiConfig {
    /// Constructor
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        http_rpc_node_addr: String,
        ws_rpc_node_addr: String,
        atoma_db: ObjectID,
        atoma_package_id: ObjectID,
        toma_package_id: ObjectID,
        request_timeout: Option<Duration>,
        limit: Option<usize>,
        small_ids: Vec<u64>,
        num_concurrent_tasks: Option<usize>,
        max_concurrent_requests: Option<u64>,
        sui_config_path: String,
    ) -> Self {
        Self {
            http_rpc_node_addr,
            ws_rpc_node_addr,
            atoma_db,
            atoma_package_id,
            toma_package_id,
            request_timeout,
            limit,
            small_ids,
            num_concurrent_tasks,
            max_concurrent_requests,
            sui_config_path,
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
    pub fn atoma_package_id(&self) -> ObjectID {
        self.atoma_package_id
    }

    /// Getter for `toma_package_id`
    pub fn toma_package_id(&self) -> ObjectID {
        self.toma_package_id
    }

    /// Getter for `atoma_db`
    pub fn atoma_db(&self) -> ObjectID {
        self.atoma_db
    }

    /// Getter for `request_timeout`
    pub fn request_timeout(&self) -> Option<Duration> {
        self.request_timeout
    }

    /// Getter for `small_id`
    pub fn small_ids(&self) -> Vec<u64> {
        self.small_ids.clone()
    }

    /// Getter for `num_concurrent_tasks`
    pub fn num_concurrent_tasks(&self) -> Option<usize> {
        self.num_concurrent_tasks
    }

    /// Getter for `max_concurrent_requests`
    pub fn max_concurrent_requests(&self) -> Option<u64> {
        self.max_concurrent_requests
    }

    /// Getter for `keystore_path`
    pub fn sui_config_path(&self) -> String {
        self.sui_config_path.clone()
    }

    /// Constructs a new `AtomaSuiConfig` instance from a configuration file path.
    ///
    /// # Arguments
    ///
    /// * `config_file_path` - A path-like object representing the location of the configuration file.
    ///
    /// # Returns
    ///
    /// Returns a new `AtomaSuiConfig` instance populated with values from the configuration file.
    ///
    /// # Panics
    ///
    /// This method will panic if:
    /// - The configuration file cannot be read or parsed.
    /// - The "atoma-sui" section is missing from the configuration file.
    /// - The configuration values cannot be deserialized into a `AtomaSuiConfig` instance.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use atoma_sui::config::AtomaSuiConfig;
    /// use std::path::Path;
    ///
    /// let config = AtomaSuiConfig::from_file_path("config.toml");
    /// ```
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder().add_source(config::File::with_name(
            config_file_path.as_ref().to_str().unwrap(),
        ));
        let config = builder
            .build()
            .expect("Failed to generate atoma-sui configuration file");
        config
            .get::<Self>("atoma-sui")
            .expect("Failed to generate configuration instance")
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config = AtomaSuiConfig::new(
            "".to_string(),
            "".to_string(),
            "0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e"
                .parse()
                .unwrap(),
            "0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e"
                .parse()
                .unwrap(),
            "0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e"
                .parse()
                .unwrap(),
            Some(Duration::from_secs(5 * 60)),
            Some(10),
            vec![0, 1, 2],
            Some(10),
            Some(10),
            "".to_string(),
        );

        let toml_str = toml::to_string(&config).unwrap();
        let should_be_toml_str = "http_rpc_node_addr = \"\"\nws_rpc_node_addr = \"\"\natoma_db = \"0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e\"\natoma_package_id = \"0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e\"\ntoma_package_id = \"0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e\"\nnum_concurrent_tasks = 10\nmax_concurrent_requests = 10\nlimit = 10\nsmall_ids = [0, 1, 2]\nsui_config_path = \"\"\n\n[request_timeout]\nsecs = 300\nnanos = 0\n";
        assert_eq!(toml_str, should_be_toml_str);
    }
}
