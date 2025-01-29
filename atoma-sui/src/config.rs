use std::{path::Path, time::Duration};

use config::Config as RustConfig;
use serde::{Deserialize, Serialize};
use sui_sdk::types::base_types::ObjectID;

/// Configuration for Sui blockchain interactions.
///
/// This struct holds the necessary configuration parameters for connecting to and
/// interacting with a Sui network, including URLs, package ID, timeout, and small IDs.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Config {
    /// The HTTP URL for a Sui RPC node, to which the subscriber will connect
    /// This is used for making HTTP requests to the Sui RPC node
    http_rpc_node_addr: String,

    /// The Atoma's DB object ID on the Sui network
    /// This identifies the specific Atoma's DB object to interact with
    atoma_db: ObjectID,

    /// The Atoma's package ID on the Sui network
    /// This identifies the specific package (smart contract) to interact with
    atoma_package_id: ObjectID,

    /// The USDC token package ID on the Sui network
    /// This identifies the specific package (smart contract) to interact with
    /// for USDC token payments
    usdc_package_id: ObjectID,

    /// The timeout duration for requests
    /// This sets the maximum time to wait for a response from the Sui network
    request_timeout: Option<Duration>,

    /// The maximum number of concurrent requests to the Sui client
    max_concurrent_requests: Option<u64>,

    /// Optional value to limit the number of dynamic fields to be retrieved for each iteration
    /// of the event subscriber loop
    limit: Option<usize>,

    /// A list of node small IDs
    /// These are values used to identify the Atoma's nodes that are under control by
    /// current Sui wallet
    node_small_ids: Option<Vec<u64>>,

    /// A list of task small IDs
    /// These are values used to identify the Atoma's tasks that are under control by
    /// current Sui wallet
    task_small_ids: Option<Vec<u64>>,

    /// Sui's config path
    sui_config_path: String,

    /// Sui's keystore path
    sui_keystore_path: String,

    /// Path to the cursor file where the cursor is stored
    cursor_path: String,
}

impl Config {
    /// Gets the HTTP RPC node address
    #[must_use]
    pub fn http_rpc_node_addr(&self) -> String {
        self.http_rpc_node_addr.clone()
    }

    /// Getter for `limit`
    #[must_use]
    pub const fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Getter for `package_id`
    #[must_use]
    pub const fn atoma_package_id(&self) -> ObjectID {
        self.atoma_package_id
    }

    /// Getter for `usdc_package_id`
    #[must_use]
    pub const fn usdc_package_id(&self) -> ObjectID {
        self.usdc_package_id
    }

    /// Getter for `atoma_db`
    #[must_use]
    pub const fn atoma_db(&self) -> ObjectID {
        self.atoma_db
    }

    /// Getter for `request_timeout`
    #[must_use]
    pub const fn request_timeout(&self) -> Option<Duration> {
        self.request_timeout
    }

    /// Getter for `small_id`
    #[must_use]
    pub fn node_small_ids(&self) -> Option<Vec<u64>> {
        self.node_small_ids.clone()
    }

    /// Getter for `task_small_ids`
    #[must_use]
    pub fn task_small_ids(&self) -> Option<Vec<u64>> {
        self.task_small_ids.clone()
    }

    /// Getter for `max_concurrent_requests`
    #[must_use]
    pub const fn max_concurrent_requests(&self) -> Option<u64> {
        self.max_concurrent_requests
    }

    /// Getter for `keystore_path`
    #[must_use]
    pub fn sui_config_path(&self) -> String {
        self.sui_config_path.clone()
    }

    /// Getter for `sui_keystore_path`
    #[must_use]
    pub fn sui_keystore_path(&self) -> String {
        self.sui_keystore_path.clone()
    }

    /// Getter for `cursor_path`
    #[must_use]
    pub fn cursor_path(&self) -> String {
        self.cursor_path.clone()
    }

    /// Constructs a new `Config` instance from a configuration file path.
    ///
    /// # Arguments
    ///
    /// * `config_file_path` - A path-like object representing the location of the configuration file.
    ///
    /// # Returns
    ///
    /// Returns a new `Config` instance populated with values from the configuration file.
    ///
    /// # Panics
    ///
    /// This method will panic if:
    /// - The configuration file cannot be read or parsed.
    /// - The "atoma-sui" section is missing from the configuration file.
    /// - The configuration values cannot be deserialized into a `Config` instance.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use atoma_sui::config::Config;
    /// use std::path::Path;
    ///
    /// let config = Config::from_file_path("config.toml");
    /// ```
    #[must_use]
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = RustConfig::builder()
            .add_source(config::File::with_name(
                config_file_path.as_ref().to_str().unwrap(),
            ))
            .add_source(
                config::Environment::with_prefix("ATOMA_SUI")
                    .keep_prefix(true)
                    .separator("__"),
            );

        let config = builder
            .build()
            .expect("Failed to generate atoma-sui configuration file");
        config
            .get::<Self>("atoma_sui")
            .expect("Failed to generate configuration instance")
    }
}

/// Builder for creating Config instances
/// Builder pattern implementation for creating `Config` instances.
///
/// This struct provides a flexible way to construct `Config` objects by allowing optional
/// setting of individual configuration parameters. Each field is wrapped in an `Option`
/// to track whether it has been explicitly set.
///
/// # Fields
///
/// * `http_rpc_node_addr` - Optional HTTP URL for the Sui RPC node
/// * `atoma_db` - Optional Atoma's DB object ID on the Sui network
/// * `atoma_package_id` - Optional Atoma's package ID on the Sui network
/// * `usdc_package_id` - Optional USDC token package ID on the Sui network
/// * `request_timeout` - Optional timeout duration for requests
/// * `max_concurrent_requests` - Optional maximum number of concurrent requests
/// * `limit` - Optional limit on number of dynamic fields per iteration
/// * `node_small_ids` - Optional list of node small IDs under control
/// * `task_small_ids` - Optional list of task small IDs under control
/// * `sui_config_path` - Optional path to Sui config file
/// * `sui_keystore_path` - Optional path to Sui keystore
/// * `cursor_path` - Optional path to cursor file
///
/// # Example
///
/// ```rust,ignore
/// let config = Builder::new()
///     .http_rpc_node_addr("http://localhost:9000".to_string())
///     .atoma_db(object_id)
///     .build();
/// ```
pub struct Builder {
    http_rpc_node_addr: Option<String>,
    atoma_db: Option<ObjectID>,
    atoma_package_id: Option<ObjectID>,
    usdc_package_id: Option<ObjectID>,
    request_timeout: Option<Duration>,
    max_concurrent_requests: Option<u64>,
    limit: Option<usize>,
    node_small_ids: Option<Vec<u64>>,
    task_small_ids: Option<Vec<u64>>,
    sui_config_path: Option<String>,
    sui_keystore_path: Option<String>,
    cursor_path: Option<String>,
}

impl Builder {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            http_rpc_node_addr: None,
            atoma_db: None,
            atoma_package_id: None,
            usdc_package_id: None,
            request_timeout: None,
            max_concurrent_requests: None,
            limit: None,
            node_small_ids: None,
            task_small_ids: None,
            sui_config_path: None,
            sui_keystore_path: None,
            cursor_path: None,
        }
    }

    #[must_use]
    pub fn http_rpc_node_addr(mut self, addr: String) -> Self {
        self.http_rpc_node_addr = Some(addr);
        self
    }

    #[must_use]
    pub const fn atoma_db(mut self, db: ObjectID) -> Self {
        self.atoma_db = Some(db);
        self
    }

    #[must_use]
    pub const fn atoma_package_id(mut self, package_id: ObjectID) -> Self {
        self.atoma_package_id = Some(package_id);
        self
    }

    #[must_use]
    pub const fn usdc_package_id(mut self, package_id: ObjectID) -> Self {
        self.usdc_package_id = Some(package_id);
        self
    }

    #[must_use]
    pub const fn request_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.request_timeout = timeout;
        self
    }

    #[must_use]
    pub const fn max_concurrent_requests(mut self, requests: Option<u64>) -> Self {
        self.max_concurrent_requests = requests;
        self
    }

    #[must_use]
    pub const fn limit(mut self, limit: Option<usize>) -> Self {
        self.limit = limit;
        self
    }

    #[must_use]
    pub fn node_small_ids(mut self, ids: Option<Vec<u64>>) -> Self {
        self.node_small_ids = ids;
        self
    }

    #[must_use]
    pub fn task_small_ids(mut self, ids: Option<Vec<u64>>) -> Self {
        self.task_small_ids = ids;
        self
    }

    #[must_use]
    pub fn sui_config_path(mut self, path: String) -> Self {
        self.sui_config_path = Some(path);
        self
    }

    #[must_use]
    pub fn sui_keystore_path(mut self, path: String) -> Self {
        self.sui_keystore_path = Some(path);
        self
    }

    #[must_use]
    pub fn cursor_path(mut self, path: String) -> Self {
        self.cursor_path = Some(path);
        self
    }

    /// Builds the final Config from the builder
    ///
    /// # Returns
    /// A new `Config` instance with the configured values
    ///
    /// # Panics
    /// This function will panic if:
    /// - `atoma_db` is not set
    /// - `atoma_package_id` is not set
    /// - `usdc_package_id` is not set
    #[must_use]
    pub fn build(self) -> Config {
        Config {
            http_rpc_node_addr: self.http_rpc_node_addr.unwrap_or_default(),
            atoma_db: self.atoma_db.expect("atoma_db is required"),
            atoma_package_id: self.atoma_package_id.expect("atoma_package_id is required"),
            usdc_package_id: self.usdc_package_id.expect("usdc_package_id is required"),
            request_timeout: self.request_timeout,
            max_concurrent_requests: self.max_concurrent_requests,
            limit: self.limit,
            node_small_ids: self.node_small_ids,
            task_small_ids: self.task_small_ids,
            sui_config_path: self.sui_config_path.unwrap_or_default(),
            sui_keystore_path: self.sui_keystore_path.unwrap_or_default(),
            cursor_path: self.cursor_path.unwrap_or_default(),
        }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config = Builder::new()
            .http_rpc_node_addr(String::new())
            .atoma_db(
                "0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e"
                    .parse()
                    .unwrap(),
            )
            .atoma_package_id(
                "0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e"
                    .parse()
                    .unwrap(),
            )
            .usdc_package_id(
                "0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e"
                    .parse()
                    .unwrap(),
            )
            .request_timeout(Some(Duration::from_secs(5 * 60)))
            .limit(Some(10))
            .node_small_ids(Some(vec![0, 1, 2]))
            .task_small_ids(Some(vec![3, 4, 5]))
            .max_concurrent_requests(Some(10))
            .sui_config_path(String::new())
            .sui_keystore_path(String::new())
            .cursor_path(String::new())
            .build();

        let toml_str = toml::to_string(&config).unwrap();
        let should_be_toml_str = "http_rpc_node_addr = \"\"\natoma_db = \"0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e\"\natoma_package_id = \"0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e\"\nusdc_package_id = \"0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e\"\nmax_concurrent_requests = 10\nlimit = 10\nnode_small_ids = [0, 1, 2]\ntask_small_ids = [3, 4, 5]\nsui_config_path = \"\"\nsui_keystore_path = \"\"\ncursor_path = \"\"\n\n[request_timeout]\nsecs = 300\nnanos = 0\n";
        assert_eq!(toml_str, should_be_toml_str);
    }
}
