pub mod config;
pub mod core;
pub mod service;
pub mod specs;
pub mod types;

use async_trait::async_trait;

#[async_trait]
pub trait InferenceServiceTrait {
    async fn start(&mut self);

    async fn shutdown(self);

    async fn is_running(&self) -> bool;
}
