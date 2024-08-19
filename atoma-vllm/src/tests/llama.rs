use crate::{llm_service::LlmService, validation::Validation, config::{CacheConfig, SchedulerConfig}};
use candle_core::{cuda::cudarc::driver::result::device, DType, Device};
use std::path::PathBuf;

const BLOCK_SIZE: usize = 16;
const MAX_STOP_SEQUENCES: usize = 1;
const MAX_TOP_N_TOKENS: u32 = 4;
const MAX_INPUT_LENGTH: usize = 512;
const MAX_NUM_SEQUENCES: usize = 32;
const MAX_TOTAL_TOKENS: u32 = 2048;

#[tokio::test]
async fn test_llama_model() {
    let api_key = "".to_string();
    let cache_dir: PathBuf = "./test_llama_cache_dir/".into();
    let model_name = "llama_tiny_llama_1_1b_chat".to_string();
    let device = Device::new_cuda(0).expect("Failed to create new CUDA device");
    let dtype = DType::BF16;
    let num_tokenizer_workers = 2;
    let revision = "main".to_string();
    let (shutdown_signal_sender, shutdown_signal_receiver) = oneshot::channel();

    let (atoma_event_subscriber_sender, atoma_event_subscriber_receiver) =
        tokio::sync::mpsc::unbounded_channel();
    let (atoma_client_sender, atoma_client_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (tokenizer_sender, tokenizer_receiver) = tokio::sync::mpsc::unbounded_channel();

    let cache_config = CacheConfig::new(
        BLOCK_SIZE,
        1.0,
        1,
        "auto".to_string(),
        None,
        None,
        100,
        100,
    ).expect("Failed to create cache config");

    let scheduler_config = SchedulerConfig::new(512, MAX_NUM_SEQUENCES, 512, 0.0, false, 0)
        .expect("Failed to create scheduler config");

    let validation_service = Validation::new(
        1,
        MAX_STOP_SEQUENCES,
        MAX_TOP_N_TOKENS,
        MAX_INPUT_LENGTH,
        MAX_TOTAL_TOKENS,
        tokenizer_sender,
    );

    let llm_service = LlmService::start(
        api_key,
        atoma_event_subscriber_receiver,
        atoma_client_sender,
        cache_config,
        cache_dir,
        device,
        dtype,
        true,
        model_name,
        num_tokenizer_workers,
        revision,
        scheduler_config,
        validation_service,
        shutdown_signal_receiver,
    )
    .await
    .expect("Failed to start LLM service");
}
