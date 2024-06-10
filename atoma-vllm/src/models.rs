use async_trait::async_trait;

use crate::sampling_params::SamplingParams;

#[async_trait]
pub trait ModelLoader {
    type Error;

    async fn fetch();
    async fn load() -> Result<Self, Self::Error>
    where
        Self: Sized;
}

#[async_trait]
pub trait ModelExecutor: ModelLoader {
    type Input;
    type Logits;
    type Output;

    async fn forward(&mut self, input: Self::Input) -> Result<Self::Logits, Self::Error>;
    async fn sample(
        &mut self,
        input: Self::Logits,
        sampling_params: SamplingParams,
    ) -> Result<Self::Output, Self::Error>;
}
