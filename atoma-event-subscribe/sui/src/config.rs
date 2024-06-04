use std::{path::Path, time::Duration};

use atoma_types::SmallId;
use config::Config;
use serde::{Deserialize, Serialize};
use sui_sdk::types::base_types::ObjectID;

#[derive(Debug, Deserialize, Serialize)]
/// `SuiSubscriberConfig` - Configuration parameters for a `SuiSubscriber`
pub struct SuiSubscriberConfig {
    /// The http url
    http_url: String,
    /// The websocket address
    ws_url: String,
    /// Atoma smart contract package id
    package_id: ObjectID,
    /// Duration of a request timeout
    request_timeout: Duration,
    /// Node's own small id (a unique identifier that is generated once the node register
    ///  on the Atoma smart contract, on Sui's blockchain)
    small_id: u64,
}

impl SuiSubscriberConfig {
    /// Constructor
    pub fn new(
        http_url: String,
        ws_url: String,
        package_id: ObjectID,
        request_timeout: Duration,
        small_id: u64,
    ) -> Self {
        Self {
            http_url,
            ws_url,
            package_id,
            request_timeout,
            small_id,
        }
    }

    /// Getter for `http_url`
    pub fn http_url(&self) -> String {
        self.http_url.clone()
    }

    /// Getter for `ws_url`
    pub fn ws_url(&self) -> String {
        self.ws_url.clone()
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
    pub fn small_id(&self) -> SmallId {
        self.small_id
    }

    /// Constructs a new `SuiSubscriberConfig` instance, from a configuration file path.
    ///
    /// It loads the file contents in memeory and deserializes it into an `SuiSubscriberConfig` instance.
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
        let config = SuiSubscriberConfig::new(
            "".to_string(),
            "".to_string(),
            "0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e"
                .parse()
                .unwrap(),
            Duration::from_secs(5 * 60),
            0,
        );

        let toml_str = toml::to_string(&config).unwrap();
        let should_be_toml_str = "http_url = \"\"\nws_url = \"\"\npackage_id = \"0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e\"\nsmall_id = 0\n\n[request_timeout]\nsecs = 300\nnanos = 0\n";
        assert_eq!(toml_str, should_be_toml_str);
    }
}
