use std::path::Path;

use config::Config;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AtomaConfidentialComputeConfig {
    pub device_indices: Vec<u16>,
    pub task_small_ids: Vec<u64>,
}

impl AtomaConfidentialComputeConfig {
    /// Creates a new configuration with the given device indices and task small ids
    ///
    /// # Panics
    ///
    /// This function will panic if the `device_indices` and `task_small_ids` have different lengths.
    #[must_use]
    pub fn new(device_indices: Vec<u16>, task_small_ids: Vec<u64>) -> Self {
        assert!(
            device_indices.len() == task_small_ids.len(),
            "device_indices and task_small_ids must have the same length"
        );
        Self {
            device_indices,
            task_small_ids,
        }
    }
}

impl AtomaConfidentialComputeConfig {
    /// Loads the configuration from a file path
    ///
    /// # Panics
    ///
    /// This function will panic if the configuration file is not found or if the configuration is invalid,
    /// including the case where the `device_indices` and `task_small_ids` have different lengths.
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder()
            .add_source(config::File::with_name(
                config_file_path.as_ref().to_str().unwrap(),
            ))
            .add_source(
                config::Environment::with_prefix("ATOMA_CONFIDENTIAL_COMPUTE")
                    .keep_prefix(true)
                    .separator("__"),
            );

        let config = builder
            .build()
            .expect("Failed to generate atoma confidential compute configuration file");
        let cc_config = config
            .get::<Self>("atoma_confidential_compute")
            .expect("Failed to generate configuration instance");
        assert!(
            cc_config.device_indices.len() == cc_config.task_small_ids.len(),
            "device indices and task small ids must have the same length"
        );
        cc_config
    }
}
