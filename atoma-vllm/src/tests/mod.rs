use std::path::PathBuf;

use async_trait::async_trait;
use rand::Rng;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use crate::{
    config::{CacheConfig, SchedulerConfig},
    llm_service::LlmService,
    model_executor::{ModelExecutor, ModelLoader},
    tokenizer::TokenizerWorker,
    validation::{NextTokenChooserParameters, Validation},
};

const BLOCK_SIZE: usize = 16;
const MAX_STOP_SEQUENCES: usize = 1;
const MAX_TOP_N_TOKENS: u32 = 0;
const MAX_INPUT_LENGTH: usize = 64;
const MAX_TOTAL_TOKENS: u32 = 512;
const NUM_CPU_BLOCKS: usize = 4096;
const NUM_GPU_BLOCKS: usize = 4096;
const TOTAL_NUM_TOKENS: usize = 512;
const EOS_TOKEN_ID: u32 = 2048;

struct MockModel {}

#[async_trait]
impl ModelLoader for MockModel {
    type FilePaths = ();
    type Error = String;

    async fn fetch() -> Result<Self::FilePaths, Self::Error> {
        Ok(())
    }

    async fn load() -> Result<Self, Self::Error> {
        Ok(Self {})
    }

    fn cache_dir(&self) -> PathBuf {
        "./cache/".into()
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(EOS_TOKEN_ID)
    }
}

#[async_trait]
impl ModelExecutor for MockModel {
    type Input = Vec<u32>;
    type Logits = Vec<(u32, f32)>;
    type Output = u32;

    async fn forward(&mut self, input: Self::Input) -> Result<Self::Logits, Self::Error> {
        let mut rng = rand::thread_rng();
        Ok(input
            .into_iter()
            .map(|u| (u, rng.gen_range(0.0..1.0)))
            .collect())
    }

    async fn sample(
        &mut self,
        mut logits: Self::Logits,
        sampling_params: NextTokenChooserParameters,
    ) -> Result<Self::Output, Self::Error> {
        let top_k = sampling_params.top_k;

        logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_k_values: Vec<_> = logits.into_iter().take(top_k as usize).collect();

        if top_k_values.is_empty() {
            panic!("Empty top k tokens array")
        }

        // Randomly sample one token from the top_k selected values
        let mut rng = rand::thread_rng();
        let sampled_index = rng.gen_range(0..top_k_values.len());

        Ok(top_k_values[sampled_index].0)
    }
}

#[tokio::test]
async fn test_llm_engine() {
    let (atoma_client_sender, atoma_client_receiver) = mpsc::unbounded_channel();
    let (atoma_event_subscriber_sender, atoma_event_subscriber_receiver) =
        mpsc::unbounded_channel();

    let cache_config = CacheConfig::new(
        BLOCK_SIZE,
        1.0,
        1,
        "auto".to_string(),
        None,
        None,
        NUM_CPU_BLOCKS,
        NUM_GPU_BLOCKS,
    )
    .expect("Failed to create cache config");

    let scheduler_config = SchedulerConfig::new(512, 4, 512, 0.0, false, 0)
        .expect("Failed to create scheduler config");

    let current_dir = std::env::current_dir().unwrap();

    let tokenizer_path = current_dir.join("src/tests/tokenizer.json");
    let tokenizer =
        Tokenizer::from_file(tokenizer_path).expect("Failed to read tokenizer from file");

    let (tokenizer_sender, tokenizer_receiver) = mpsc::unbounded_channel();
    let validation = Validation::new(
        1,
        MAX_STOP_SEQUENCES,
        MAX_TOP_N_TOKENS,
        MAX_INPUT_LENGTH,
        MAX_TOTAL_TOKENS,
        tokenizer_sender,
    );

    let tokenizer_clone = tokenizer.clone();
    let _tokenizer_handle = tokio::spawn(async move {
        let _tokenizer_worker = TokenizerWorker::start(tokenizer_clone, tokenizer_receiver, 2)
            .await
            .expect("Failed to start tokenizer");
    });

    let model = MockModel::load()
        .await
        .expect("Failed to create mock model");

    let service = LlmService::start(
        atoma_event_subscriber_receiver,
        atoma_client_sender,
        cache_config,
        true,
        scheduler_config,
        model,
        tokenizer,
        validation,
    )
    .await
    .expect("Failed to start LLM service");
}
