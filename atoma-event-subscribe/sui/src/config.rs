use std::{path::Path, time::Duration};

use config::Config;
use serde::{Deserialize, Serialize};
use sui_sdk::types::base_types::ObjectID;

#[derive(Debug, Deserialize, Serialize)]
pub struct SuiSubscriberConfig {
    http_url: String,
    ws_url: String,
    object_id: ObjectID,
    request_timeout: Duration,
}

impl SuiSubscriberConfig {
    pub fn new(
        http_url: String,
        ws_url: String,
        object_id: ObjectID,
        request_timeout: Duration,
    ) -> Self {
        Self {
            http_url,
            ws_url,
            object_id,
            request_timeout,
        }
    }

    pub fn http_url(&self) -> String {
        self.http_url.clone()
    }

    pub fn ws_url(&self) -> String {
        self.ws_url.clone()
    }

    pub fn object_id(&self) -> ObjectID {
        self.object_id
    }

    pub fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder().add_source(config::File::with_name(
            config_file_path.as_ref().to_str().unwrap(),
        ));
        let config = builder
            .build()
            .expect("Failed to generate inference configuration file");
        config
            .try_deserialize::<Self>()
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
        );

        let toml_str = toml::to_string(&config).unwrap();
        let should_be_toml_str = "http_url = \"\"\nws_url = \"\"\nobject_id = \"0x8d97f1cd6ac663735be08d1d2b6d02a159e711586461306ce60a2b7a6a565a9e\"\n\n[request_timeout]\nsecs = 300\nnanos = 0\n";
        assert_eq!(toml_str, should_be_toml_str);
    }
}