use std::{collections::HashMap, sync::Arc};

use super::{
    cache::{CacheEngine, CacheEngineError},
    config::ModelConfig,
    model_executor::{ModelExecutor, ModelExecutorError, ModelLoaderError},
    sequence::{ExecuteModelRequest, SequenceMetadata, SequenceOutput},
    tokenizer::TokenizerWorker,
};
use candle::{DType, DTypeParseError, Device, Error as CandleError, Tensor};
use thiserror::Error;
use tracing::{error, info, info_span, instrument, warn, Span};

const PAD_SLOT_ID: i64 = -1;

/// `ModelInput` - Input for LLM model
/// forward pass
pub struct ModelInput {
    /// Input tokens tensor
    input_tokens_tensor: Tensor,
    /// Input positions tensor
    input_positions: Tensor,
    /// Number of decoded tokens
    num_decode_tokens: usize,
    /// Number of prefills
    num_prefills: usize,
    /// Cumulative query lengths, of size `batch_size + 1`
    cu_query_lengths: Tensor,
}

/// `ModelWorker` - Responsible for running a LLM model
/// instance (or a partition of it).
///
/// Each worker is associated with a single GPU. The worker is responsible for
/// maintaining the KV cache and executing the model on the GPU. In case of
/// distributed inference, each worker is assigned a partition of the model.
pub struct ModelWorker<M: ModelExecutor> {
    /// Cache engine
    cache_engine: CacheEngine,
    /// Device,
    device: Device,
    /// Enable chunked prefill (boolean)
    enable_chunked_prefill: bool,
    /// Model runner instance
    model: M,
    /// Initial GPU available memory
    initial_gpu_memory: usize,
    /// Tracing Span
    span: Span,
}

impl<M> ModelWorker<M>
where
    M: ModelExecutor,
{
    /// Constructor
    #[instrument(skip_all)]
    pub fn new(
        device: Device,
        dtype: DType,
        model: M,
        enable_chunked_prefill: bool,
    ) -> Result<Self, ModelWorkerError> {
        let span = info_span!("model-worker");
        let _span = span.clone();
        let _enter = _span.enter();

        info!("Starting a new `ModelWorker` instance");
        let cache_engine = CacheEngine::new(
            device.clone(),
            dtype,
            model.alibi_slopes(),
            model.hidden_size() / model.num_attention_heads(),
            model.num_attention_heads(),
            model.num_attention_heads(),
            model.num_kv_heads(),
            model.softmax_scale(),
            model.sliding_window(),
        )?;

        // TODO:
        // 1. Check cuda is available (error otherwise);
        // 2. Access initial GPU memory (using cudarc)
        Ok(Self {
            cache_engine,
            device,
            enable_chunked_prefill,
            model,
            initial_gpu_memory: 0, // TODO 2.
            span,
        })
    }

    /// Determines the number of available GPU blocks
    pub fn num_available_gpu_blocks(&self) -> usize {
        todo!()
    }

    /// Executes model's forward pass
    #[instrument(skip_all)]
    pub fn execute_model(
        &mut self,
        request: ExecuteModelRequest,
    ) -> Result<Vec<SequenceOutput>, ModelWorkerError> {
        info!("Executing model on new request..");

        let span = self.span.clone();
        let _enter = span.enter();

        let ExecuteModelRequest {
            sequences_metadata,
            ..
        } = request;

        let num_sequence_groups = sequences_metadata.len();

        // NOTE: Number of sequence groups should not be zero,
        // as we don't schedule empty sequences, for now.
        if num_sequence_groups == 0 {
            warn!("Number of sequence groups to run model on should not be empty");
            return Ok(vec![]);
        }

        let ModelInput {
            input_tokens_tensor,
            input_positions,
            cu_query_lengths,
            ..
        } = self.prepare_input_tensors(&sequences_metadata)?;

        let selected_token_indices = utils::compute_selected_token_indices(&cu_query_lengths)?;

        let kv_cache = self.cache_engine.gpu_cache.iter_mut().collect();
        let logits = self.model.forward(
            &input_tokens_tensor,
            &input_positions,
            &selected_token_indices,
        )?;

        let sampled_outputs = self.model.sample(&logits, &sequences_metadata)?;

        Ok(sampled_outputs)
    }

    /// Swaps cached KV tensors, if needed
    #[instrument(skip_all)]
    pub fn cache_swap(
        &mut self,
    ) -> Result<(), ModelWorkerError> {
        todo!()
    }

    /// Prepares input tensors for model forward run, based
    /// on availabe sequence groups metadata.
    ///
    /// The API assumes seq_group_metadata_list is sorted by prefill -> decode.
    ///
    /// The result tensors and data structure also batches input in prefill
    /// -> decode order. For example,
    ///
    /// - input_tokens.i(..num_prefill_tokens) contains prefill tokens.
    /// - input_tokens.i(num_prefill_tokens..) contains decode tokens.
    ///
    #[instrument(skip_all)]
    pub fn prepare_input_tensors(
        &self,
        sequence_groups_metadata: &Vec<Arc<SequenceMetadata>>,
    ) -> Result<ModelInput, ModelWorkerError> {
        let _enter = self.span.enter();
        info!("Preparing input tensors for new inference request..");

        let mut input_tokens = Vec::<u32>::new();
        let mut input_positions = Vec::new();
        let mut sequence_lengths = Vec::new();
        let mut prefill_sequence_lengths = Vec::new();
        let mut decode_sequence_lengths = Vec::new();
        let mut context_lengths = Vec::new();
        let mut query_lengths = Vec::new();

        let mut num_prefills = 0;
        let mut num_prefill_tokens = 0;
        let mut num_decode_tokens = 0;

        for sequence_group_metadata in sequence_groups_metadata.iter() {
            let is_prompt = sequence_group_metadata.is_prompt;

            for (sequence_id, sequence_data) in sequence_group_metadata.sequence_data.iter() {
                // 1. Context length
                let context_length = if is_prompt {
                    sequence_data.get_num_computed_tokens()
                } else {
                    // NOTE: If we ever want to introduce speculative
                    // decoding in the future, this is invalid
                    // for it, so one needs to introduce additional
                    // logic.
                    sequence_data.length() - 1
                };

                // 2. Sequence length
                let sequence_length = sequence_data
                    .length()
                    .min(context_length + sequence_group_metadata.token_chunk_size);

                // 3. Tokens
                let tokens = if is_prompt {
                    &sequence_data.get_token_ids()[context_length..sequence_length]
                } else {
                    if sequence_data.get_last_token_id().is_none() {
                        error!("Empty prompts should not be received in `ModelWorker`");
                        return Err(ModelWorkerError::EmptyPrompt(
                            "Empty prompts should not be received in `ModelWorker`".into(),
                        ));
                    }
                    // DON'T PANIC: we should not receive empty prompts
                    let last_token_id = sequence_data.get_last_token_id().unwrap();
                    &[last_token_id]
                };

                // 4. Query length
                let query_length = if is_prompt {
                    sequence_length - context_length
                } else {
                    1
                };

                // 5. Update previous values if sliding window is used

                // These are seq_len/context_len capped to the sliding window.
                // They are passed to decode kernel.
                // We still need original seq_len/context_len to compute slot
                // mapping (and input position) below.
                let mut sliding_sequence_length = sequence_length;
                let sliding_context_length = context_length;

                // This is a hack to make sliding window work with
                // Paged Attention. We can remove it if we make paged attn kernel
                // to properly handle sliding window attention.
                if self.model.sliding_window().is_some() && !is_prompt {
                    // DON'T PANIC: by the branch check
                    sliding_sequence_length =
                        self.model.sliding_window().unwrap().min(sequence_length);
                }

                // 6. Update intermediate states
                sequence_lengths.push(sliding_sequence_length as u32);
                context_lengths.push(sliding_context_length as u32);

                query_lengths.push(query_length as u32);
                input_tokens.extend(tokens);
                input_positions.extend((context_length as i64)..(sequence_length as i64));

                // 7. Update intermediate states depending on the type of the sequence
                //    (prompt or decode)
                if is_prompt {
                    debug_assert_eq!(
                        sequence_group_metadata.sequence_data.keys().len(),
                        1,
                        "Prompt should have only one sequence ID"
                    );
                    num_prefills += 1;
                    num_prefill_tokens += tokens.len();
                    prefill_sequence_lengths.push(sequence_length);
                } else {
                    debug_assert_eq!(
                        query_length, 1,
                        "Invalid query length: seq_len: {}, context_len: {}, query_len: {}",
                        sequence_length, context_length, query_length
                    );
                    num_decode_tokens += query_length;
                    decode_sequence_lengths.push(sliding_sequence_length);
                }

                // 8. Compute the slot mapping.
                let block_table = sequence_group_metadata
                    .block_tables
                    .get(sequence_id)
                    .expect("Block table should exist for a sequence on decoding phase");
            }
        }

        // 9. Build the required tensors for attention metadata
        let max_query_len = *query_lengths.iter().max().unwrap_or(&0) as usize;
        let max_prefill_seq_len = *prefill_sequence_lengths.iter().max().unwrap_or(&0) as usize;
        let max_decode_seq_len = *decode_sequence_lengths.iter().max().unwrap_or(&0);

        let seq_lens_tensor = Tensor::new(sequence_lengths, &self.device)?;
        let mut seq_start_loc =
            Tensor::zeros(seq_lens_tensor.dims1()? + 1, DType::U32, &self.device)?;

        let cumsum_seq_lens = seq_lens_tensor
            .to_dtype(DType::F32)?
            .cumsum(0)?
            .to_dtype(DType::U32)?;
        seq_start_loc = seq_start_loc.slice_assign(&[1..], &cumsum_seq_lens)?;

        let input_tokens_tensor = Tensor::new(input_tokens, &self.device)?.unsqueeze(0)?;
        let input_positions_tensor = Tensor::new(input_positions, &self.device)?.unsqueeze(0)?;

        let context_lens_tensor = Tensor::new(context_lengths, &self.device)?;
        let query_lens_tensor = Tensor::new(query_lengths, &self.device)?.to_dtype(DType::F32)?;
        let mut query_start_loc =
            Tensor::zeros(query_lens_tensor.dims1()? + 1, DType::U32, &self.device)?;

        let cumsum_query_lens = query_lens_tensor.cumsum(0)?.to_dtype(DType::U32)?;
        query_start_loc = query_start_loc.slice_assign(&[1..], &cumsum_query_lens)?;

        Ok(ModelInput {
            input_tokens_tensor,
            input_positions: input_positions_tensor,
            num_decode_tokens,
            num_prefills,
            cu_query_lengths: query_start_loc,
        })
    }
}

#[derive(Debug, Error)]
pub enum ModelWorkerError {
    #[error("Candle error: `{0}`")]
    CandleError(#[from] CandleError),
    #[error("Model loader error: `{0}`")]
    ModelLoader(#[from] ModelLoaderError),
    #[error("Cache engine error: `{0}`")]
    CacheEngineError(#[from] CacheEngineError),
    #[error("Prefill chunked error: `{0}`")]
    InvalidChunkedPrefill(String),
    #[error("Empty prompt error: `{0}`")]
    EmptyPrompt(String),
    #[error("Invalid number sequences error: `{0}`")]
    InvalidNumberSequences(String),
    #[error("Model executor error: `{0}`")]
    ModelExecutorError(#[from] ModelExecutorError),
}

#[cfg(test)]
mod tests {
    use candle::{DType, Device};

    use super::*;

    struct MockModelExecuter {}

    // fn create_model_worker() -> ModelWorker<MockModelExecuter> {
    //     ModelWorker {
    //         cache_engine: CacheEngine::new(
    //             16,
    //             Device::Cpu,
    //             DType::BF16,
    //             None,
    //             64,
    //             32,
    //             16,
    //             4,
    //             128,
    //             128,
    //             1.,
    //             None,
    //         )
    //         .unwrap(),
    //         device: Device::Cpu,
    //         cache_config: CacheConfig::new(
    //             16,
    //             0.8,
    //             1_024,
    //             String::from("bf16"),
    //             None,
    //             None,
    //             128,
    //             128,
    //         )
    //         .unwrap(),
    //         enable_chunked_prefill: false,
    //         model: MockModelExecuter {},
    //         initial_gpu_memory: 1_024,
    //         span: info_span!("mock-model-worker"),
    //     }
    // }
}