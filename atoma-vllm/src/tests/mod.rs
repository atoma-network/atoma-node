use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use rand::Rng;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::info;

use crate::{
    config::{CacheConfig, SchedulerConfig},
    llm_service::LlmService,
    model_executor::{ModelExecutor, ModelExecutorError, ModelLoader, ModelLoaderError},
    sequence::ExecuteModelRequest,
    tokenizer::TokenizerWorker,
    types::{GenerateParameters, GenerateRequest},
    validation::{NextTokenChooserParameters, StoppingCriteriaParameters, Validation},
};

const BLOCK_SIZE: usize = 16;
const MAX_STOP_SEQUENCES: usize = 1;
const MAX_TOP_N_TOKENS: u32 = 0;
const MAX_INPUT_LENGTH: usize = 16;
const MAX_TOTAL_TOKENS: u32 = 32;
const NUM_CPU_BLOCKS: usize = 4096;
const NUM_GPU_BLOCKS: usize = 4096;
const EOS_TOKEN_ID: u32 = 2048;

struct MockModel {}

#[async_trait]
impl ModelLoader for MockModel {
    type FilePaths = ();

    async fn fetch() -> Result<Self::FilePaths, ModelLoaderError> {
        Ok(())
    }

    async fn load() -> Result<Self, ModelLoaderError> {
        Ok(Self {})
    }

    fn cache_dir(&self) -> PathBuf {
        "./cache/".into()
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(EOS_TOKEN_ID)
    }
}

impl From<ExecuteModelRequest> for Vec<u32> {
    fn from(value: ExecuteModelRequest) -> Self {
        value
            .sequence_groups_metadata
            .first()
            .unwrap()
            .sequence_data
            .values()
            .next()
            .unwrap()
            .get_token_ids()
    }
}

#[async_trait]
impl ModelExecutor for MockModel {
    type Input = Vec<u32>;
    type Logits = Vec<(u32, f32)>;
    type Output = u32;

    fn forward(&mut self, input: Self::Input) -> Result<Self::Logits, ModelExecutorError> {
        let mut rng = rand::thread_rng();
        std::thread::sleep(Duration::from_secs(2)); // mimic forward pass
        Ok(input
            .into_iter()
            .map(|u| (u, rng.gen_range(0.0..1.0)))
            .collect())
    }

    fn sample(
        &mut self,
        mut logits: Self::Logits,
        next_token_params: NextTokenChooserParameters,
        stopping_params: StoppingCriteriaParameters,
    ) -> Result<Self::Output, ModelExecutorError> {
        let top_k = next_token_params.top_k;

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
    init_tracing();

    const NUM_REQUESTS: usize = 128;
    const MAX_NUM_SEQUENCES: usize = 32;
    const NUM_RUNS: usize = NUM_REQUESTS / MAX_NUM_SEQUENCES;

    let (atoma_client_sender, mut atoma_client_receiver) = mpsc::unbounded_channel();
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

    let scheduler_config = SchedulerConfig::new(512, MAX_NUM_SEQUENCES, 512, 0.0, false, 0)
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

    let mut service = LlmService::start(
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

    tokio::spawn(async move {
        service.run().await.expect("Fail to run llm service");
    });

    info!("Sending request through atoma_event_subscriber_sender");

    let requests = (0..NUM_REQUESTS).map(|i| GenerateRequest {
        request_id: format!("{}", i),
        inputs: "Hello world, from the Caribbean".to_string(),
        parameters: GenerateParameters {
            best_of: None,
            temperature: Some(1.2),
            repetition_penalty: Some(1.1),
            frequency_penalty: Some(1.1),
            repeat_last_n: Some(8),
            top_k: Some(8),
            top_p: Some(0.8),
            typical_p: None,
            do_sample: true,
            max_new_tokens: Some(16),
            return_full_text: Some(true),
            stop: vec!["STOP".to_string()],
            truncate: None,
            decoder_input_details: true,
            random_seed: Some(42),
            top_n_tokens: None,
            n: 1,
        },
    });

    for request in requests {
        atoma_event_subscriber_sender
            .send(request)
            .expect("Failed to send request");
    }

    let mut number_of_responses = 0;

    let start = Instant::now();
    let mut elapsed_times = Vec::with_capacity(100);

    for _ in 0..(NUM_RUNS) {
        let responses = atoma_client_receiver.recv().await.unwrap();
        elapsed_times.push(start.elapsed());
        for response in responses.iter() {
            number_of_responses += 1;
            info!("Got new response: {response:?}");
        }
        info!("Number of responses {number_of_responses}")
    }

    info!("Elapsed times: {elapsed_times:?}");

    assert_eq!(number_of_responses, NUM_REQUESTS);
    assert_eq!(elapsed_times.len(), NUM_RUNS);

    for i in 0..(NUM_RUNS - 1) {
        let left_run_time = elapsed_times[i];
        let right_run_time = elapsed_times[i + 1];
        assert!(right_run_time - left_run_time <= elapsed_times[0] + Duration::from_secs(5)); // Give enough variability time for different machines
        assert!(right_run_time - left_run_time <= elapsed_times[0] - Duration::from_secs(5));
    }
}

#[tokio::test]
async fn test_llm_engine_with_enable_chunking() {
    init_tracing();

    const NUM_REQUESTS: usize = 128;
    const MAX_NUM_SEQUENCES: usize = 32;
    const NUM_RUNS: usize = NUM_REQUESTS / MAX_NUM_SEQUENCES;

    let (atoma_client_sender, mut atoma_client_receiver) = mpsc::unbounded_channel();
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

    let scheduler_config = SchedulerConfig::new(512, MAX_NUM_SEQUENCES, 512, 0.0, true, 0)
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

    let mut service = LlmService::start(
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

    tokio::spawn(async move {
        service.run().await.expect("Fail to run llm service");
    });

    info!("Sending request through atoma_event_subscriber_sender");

    let requests = (0..NUM_REQUESTS).map(|i| GenerateRequest {
        request_id: format!("{}", i),
        inputs: "Hello world, from the Caribbean".to_string(),
        parameters: GenerateParameters {
            best_of: None,
            temperature: Some(1.2),
            repetition_penalty: Some(1.1),
            frequency_penalty: Some(1.1),
            repeat_last_n: Some(8),
            top_k: Some(8),
            top_p: Some(0.8),
            typical_p: None,
            do_sample: true,
            max_new_tokens: Some(16),
            return_full_text: Some(true),
            stop: vec!["STOP".to_string()],
            truncate: None,
            decoder_input_details: true,
            random_seed: Some(42),
            top_n_tokens: None,
            n: 1,
        },
    });

    for request in requests {
        atoma_event_subscriber_sender
            .send(request)
            .expect("Failed to send request");
    }

    let mut number_of_responses = 0;

    let start = Instant::now();
    let mut elapsed_times = Vec::with_capacity(100);

    for _ in 0..(NUM_RUNS) {
        let responses = atoma_client_receiver.recv().await.unwrap();
        elapsed_times.push(start.elapsed());
        for response in responses.iter() {
            number_of_responses += 1;
            info!("Got new response: {response:?}");
        }
        info!("Number of responses {number_of_responses}")
    }

    info!("Elapsed times: {elapsed_times:?}");

    assert_eq!(number_of_responses, NUM_REQUESTS);
    assert_eq!(elapsed_times.len(), NUM_RUNS);

    for i in 0..(NUM_RUNS - 1) {
        let left_run_time = elapsed_times[i];
        let right_run_time = elapsed_times[i + 1];
        assert!(left_run_time - right_run_time - elapsed_times[0] <= Duration::from_secs(5));
        // Give enough variability time for different machines
    }
}

pub fn init_tracing() {
    let _ = tracing_subscriber::fmt::try_init();
}
