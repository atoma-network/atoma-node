use std::path::Path;

use config::Config;
use serde::Deserialize;
use sui_sdk::types::base_types::ObjectID;

#[derive(Debug, Deserialize)]
pub struct SuiSubscriberConfig {
    http_url: String,
    ws_url: String,
    object_id: ObjectID,
}

impl SuiSubscriberConfig {
    pub fn new(http_url: String, ws_url: String, object_id: ObjectID) -> Self {
        Self {
            http_url,
            ws_url,
            object_id,
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
