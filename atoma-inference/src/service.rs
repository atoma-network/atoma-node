use async_trait::async_trait;
use ed25519_consensus::SigningKey as PrivateKey;

use crate::{config::InferenceConfig, InferenceServiceTrait};

pub struct InferenceService {
    config: InferenceConfig,
    private_key: PrivateKey,
}

impl InferenceService {
    pub fn new(config: InferenceConfig, private_key: PrivateKey) -> Self {
        Self {
            config,
            private_key,
        }
    }
}

#[async_trait]
impl InferenceServiceTrait for InferenceService {
    async fn start(&mut self) {}

    async fn shutdown(self) {}

    async fn is_running(&self) -> bool {
        todo!()
    }
}
