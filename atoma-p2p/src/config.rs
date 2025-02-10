use config::{Config, File};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use validator::{Validate, ValidationError};

/// Configuration settings for a P2P Atoma Node.
///
/// This struct holds timing-related configuration parameters that control
/// the behavior of peer-to-peer connections in an Atoma node.

#[derive(Clone, Debug, Deserialize, Serialize, Validate)]
pub struct AtomaP2pNodeConfig {
    /// Country of origin of the node
    #[validate(custom(function = validate_country_code))]
    pub country: String,

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

    /// The address to listen on for incoming QUIC connections.
    ///
    /// This is the address that the node will use to listen for incoming connections.
    /// It is a string in the format of "/ip4/x.x.x.x/udp/x/quic-v1".
    pub listen_addr: String,

    /// The public URL of the node.
    ///
    /// This is the URL that the node will use to send requests to the network.
    /// It is a string in the format of "https://x.x.x.x:x".
    #[validate(custom(function = validate_public_url))]
    pub public_url: String,

    /// The node's small id (assigned by the Atoma smart contract)
    pub node_small_id: u64,

    /// The list of bootstrap nodes to connect to.
    ///
    /// Bootstrap nodes are nodes that the node will use to bootstrap its network connection.
    /// They are a list of strings in the format of "/ip4/x.x.x.x/udp/x/quic-v1/p2p/QmHash...".
    pub bootstrap_nodes: Vec<String>,
}

impl AtomaP2pNodeConfig {
    /// Creates a new `AtomaP2pNodeConfig` instance from a configuration file.
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
    /// Returns a new `AtomaP2pNodeConfig` instance with the loaded configuration.
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
    /// ```rust,ignore
    /// use atoma_p2p::config::AtomaP2pNodeConfig;
    ///
    /// let config = AtomaP2pNodeConfig::from_file_path("config/atoma.toml");
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
        let config = config
            .get::<Self>("atoma_p2p")
            .expect("Failed to generate configuration instance");
        config.validate().expect("Configuration is invalid");
        config
    }
}

/// Validates the country code of the node.
///
/// This function validates the country code of the node by checking if it is a valid ISO 3166-1 alpha-2 country code.
///
/// # Arguments
///
/// * `country` - The country code of the node.
///
/// # Returns
///
/// Returns a `Result<()>`, which is `Ok(())` if the country code is valid, or an error if it is invalid.
fn validate_country_code(country: &str) -> Result<(), ValidationError> {
    isocountry::CountryCode::for_alpha2(country)
        .map_err(|_| ValidationError::new("Country code is invalid"))?;
    Ok(())
}

/// Validates the public URL of the node.
///
/// This function validates the public URL of the node by parsing it and checking if it is a valid URL.
///
/// # Arguments
///
/// * `public_url` - The public URL of the node.
///
/// # Returns
///
/// Returns a `Result<()>`, which is `Ok(())` if the public URL is valid, or an error if it is invalid.
fn validate_public_url(public_url: &str) -> Result<(), ValidationError> {
    url::Url::parse(public_url).map_err(|_| ValidationError::new("Public URL is invalid"))?;
    Ok(())
}
