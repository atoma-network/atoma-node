use serde::Deserialize;
use std::path::Path;

use config::Config;

#[derive(Deserialize)]
pub struct SuiConfig {
    atoma_contract_address: String,
    address: String,
    http_addr: String,
    ws_addr: String,
}

impl SuiConfig {
    pub fn new(
        address: String,
        atoma_contract_address: String,
        http_addr: String,
        ws_addr: String,
    ) -> Self {
        Self {
            atoma_contract_address,
            address,
            http_addr,
            ws_addr,
        }
    }

    pub fn address(&self) -> String {
        self.address.clone()
    }

    pub fn atoma_contract_address(&self) -> String {
        self.atoma_contract_address.clone()
    }

    pub fn http_addr(&self) -> String {
        self.http_addr.clone()
    }

    pub fn ws_addr(&self) -> String {
        self.ws_addr.clone()
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
