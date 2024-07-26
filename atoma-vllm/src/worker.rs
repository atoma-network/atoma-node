use std::sync::Arc;

use crate::{
    config::{CacheConfig, SchedulerConfig},
    model_executor::{ModelExecutor, ModelLoaderError},
    sequence::{ExecuteModelRequest, SequenceGroupMetadata},
};
use atoma_paged_attention::flash_attention::{
    FlashAttention, FlashAttentionDecodingMetadata, FlashAttentionMetadata,
    FlashAttentionPrefillMetadata,
};
use candle::{DType, DTypeParseError, Device, Error as CandleError, Tensor};
use thiserror::Error;
use tracing::{error, info_span, instrument, warn, Span};

const PAD_SLOT_ID: i64 = -1;

/// `ModelInput` - Input for LLM model
/// forward pass
pub struct ModelInput {
    /// Input tokens tensor
    input_tokens_tensor: Tensor,
    /// Input positions tensor
    input_positions: Tensor,
    /// Attention Metadata
    attention_metadata: FlashAttentionMetadata,
    /// Number of decoded tokens
    num_decode_tokens: usize,
    /// Number of prefills
    num_prefills: usize,
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
    /// Cache configuration
    cache_config: CacheConfig,
    /// Scheduler configuration
    scheduler_config: SchedulerConfig,
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
    pub fn new(
        api_key: String,
        cache_config: CacheConfig,
        device: Device,
        dtype: DType,
        model_name: String,
        revision: String,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, ModelWorkerError> {
        // NOTE: for now we use a synchronous model loader
        let file_paths = M::fetch(api_key, model_name, revision)?;
        let model = M::load(file_paths)?;
        let cache_engine = CacheEngine::new(
            &cache_config,
            device.clone(),
            dtype,
            model.head_size(),
            model.num_attention_heads(),
            model.num_layers(),
            model.num_kv_heads(),
            model.sliding_window(),
        )?;

        // TODO:
        // 1. Check cuda is available (error otherwise);
        // 2. Access initial GPU memory (using cudarc)
        Ok(Self {
            cache_engine,
            device,
            cache_config,
            scheduler_config,
            model,
            initial_gpu_memory: 0, // TODO 2.
            span: info_span!("model-worker"),
        })
    }

    /// Determines the number of available GPU blocks
    pub fn num_available_gpu_blocks(&self) -> usize {
        todo!()
    }

    /// Executes model's forward pass
    #[instrument(skip_all)]
    pub fn execute_model(
        &self,
        request: ExecuteModelRequest,
    ) -> Result<M::Output, ModelWorkerError> {
        let _enter = self.span.enter();

        let ExecuteModelRequest {
            sequence_groups_metadata,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            running_queue_size,
        } = request;

        let num_sequence_groups = sequence_groups_metadata.len();

        // TODO: pass these simply as `HashMap<i64, i64>` swap blocks
        let blocks_to_swap_in = blocks_to_swap_in
            .into_iter()
            .flat_map(|(i, j)| [i, j])
            .collect::<Vec<_>>();
        // `blocks_to_swap_in` and `blocks_to_swap_out` are CPU tensors
        let blocks_to_swap_in =
            Tensor::new(blocks_to_swap_in, &candle::Device::Cpu)?.reshape(((), 2))?;

        let blocks_to_swap_out = blocks_to_swap_out
            .into_iter()
            .flat_map(|(i, j)| [i, j])
            .collect::<Vec<_>>();
        let blocks_to_swap_out =
            Tensor::new(blocks_to_swap_out, &candle::Device::Cpu)?.reshape(((), 2))?;

        // `blocks_to_copy` is a gpu tensor. The source and target of
        // blocks to copy are in the same device, and `blocks_to_copy`
        // can be used directly within cuda kernels.
        let blocks_to_copy = blocks_to_copy
            .into_iter()
            .flat_map(|(i, j)| [i, j])
            .collect::<Vec<_>>();
        let blocks_to_copy = Tensor::new(blocks_to_copy, &self.device)?.reshape(((), 2))?;

        // At this point we need to perform cache swap operations
        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)?;

        // NOTE: Number of sequence groups should not be zero,
        // as we don't schedule empty sequences, for now.
        if num_sequence_groups == 0 {
            warn!("Number of sequence groups to run model on should not be empty");
            return Ok(());
        }

        let next_tokens_params: Vec<_> = sequence_groups_metadata
            .iter()
            .map(|metadata| metadata.next_token_chooser_params.clone())
            .collect();
        let stopping_params: Vec<_> = sequence_groups_metadata
            .iter()
            .map(|metadata| metadata.stopping_criteria_params.clone())
            .collect();

        let ModelInput {
            input_tokens_tensor,
            input_positions,
            attention_metadata,
            num_decode_tokens,
            num_prefills,
        } = self.prepare_input_tensors(sequence_groups_metadata)?;

        let hidden_states = self.model.forward(
            &input_tokens_tensor,
            &input_positions,
            &self.cache_engine.gpu_cache,
            &attention_metadata,
        )?;

        let logits = self.model.compute_logits(&hidden_states)?;

        let sampled_outputs = self
            .model
            .sample(&logits, next_tokens_params, stopping_params)?;

        Ok(sampled_outputs)
    }

    /// Swaps cached blocks
    #[instrument(skip_all)]
    pub fn cache_swap(
        &self,
        blocks_to_swap_in: Tensor,
        blocks_to_swap_out: Tensor,
        blocks_to_copy: Tensor,
    ) -> Result<(), ModelWorkerError> {
        if blocks_to_swap_in.elem_count() > 0 {
            self.cache_engine.swap_in(blocks_to_swap_in)?
        }
        if blocks_to_swap_out.elem_count() > 0 {
            self.cache_engine.swap_out(blocks_to_swap_out)?
        }
        if blocks_to_copy.elem_count() > 0 {
            self.cache_engine.copy_blocks(blocks_to_copy)?
        }
        Ok(())
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
        sequence_groups_metadata: Vec<Arc<SequenceGroupMetadata>>,
    ) -> Result<ModelInput, ModelWorkerError> {
        let mut input_tokens = Vec::new();
        let mut input_positions = Vec::new();
        let mut slot_mapping = Vec::new();
        let mut sequence_lengths = Vec::new();
        let mut prefill_sequence_lengths = Vec::new();
        let mut decode_sequence_lengths = Vec::new();
        let mut context_lengths = Vec::new();
        let mut query_lengths = Vec::new();
        let mut block_tables = Vec::new();

        let mut decode_only = true;
        let mut num_prefills = 0;
        let mut num_prefill_tokens = 0;
        let mut num_decode_tokens = 0;

        let mut sliding_window_blocks = self
            .model
            .sliding_window()
            .map(|sw| (sw + self.cache_engine.block_size - 1) / self.cache_engine.block_size);

        for sequence_group_metadata in sequence_groups_metadata.iter() {
            let is_prompt = sequence_group_metadata.is_prompt;

            for (sequence_id, sequence_data) in sequence_group_metadata.sequence_data.iter() {
                let context_length = if is_prompt {
                    sequence_data.get_num_computed_tokens()
                } else {
                    // NOTE: If we ever want to introduce speculative
                    // decoding in the future, this is invalid
                    // for it, so one needs to introduce additional
                    // logic.
                    sequence_data.length() - 1
                };

                let sequence_length = sequence_data
                    .length()
                    .min(context_length + sequence_group_metadata.token_chunk_size);

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

                // These are seq_len/context_len capped to the sliding window.
                // They are passed to decode kernel.
                // We still need original seq_len/context_len to compute slot
                // mapping (and input position) below.
                let mut sliding_sequence_length = sequence_data.length();
                let mut sliding_context_length = context_length;

                // This is a hack to make sliding window work with
                // Paged Attention. We can remove it if we make paged attn kernel
                // to properly handle sliding window attention.
                if self.model.sliding_window().is_some() && !is_prompt {
                    // DON'T PANIC: by the branch check
                    sliding_sequence_length =
                        self.model.sliding_window().unwrap().min(sequence_length);
                    sliding_context_length = sliding_sequence_length - 1;
                }

                let block_table = if self.scheduler_config.enable_chunked_prefill() || !is_prompt {
                    // DON'T PANIC: Unwrap is safe here because block_tables
                    // should have allocated a block table for this sequence
                    let mut block_table = sequence_group_metadata
                        .block_tables
                        .get(sequence_id)
                        .expect("Block table should be allocated for sequence on decoding phase")
                        .clone();

                    if let Some(current_sw_blocks) = self.model.sliding_window() {
                        let start = block_table.len().saturating_sub(current_sw_blocks);
                        block_table = block_table[start..].to_vec();
                    }

                    block_table
                } else {
                    // Prefill without chunked prefill
                    vec![]
                };

                block_tables.push(block_table);
                sequence_lengths.push(sliding_sequence_length);
                context_lengths.push(sliding_context_length as u32);

                let query_length = sliding_sequence_length - sliding_context_length;
                query_lengths.push(query_length as u32);
                input_tokens.extend(tokens);
                input_positions.extend((context_length as i64)..(sequence_length as i64));

                if is_prompt {
                    debug_assert_eq!(
                        sequence_group_metadata.sequence_data.keys().len(),
                        1,
                        "Prompt should have only one sequence ID"
                    );
                    num_prefills += 1;
                    num_prefill_tokens += tokens.len();
                    decode_only = false;
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

                if sequence_group_metadata.block_tables.is_empty() {
                    // During memory profiling, the block tables are not
                    // initialized yet. In this case, we just use a dummy
                    // slot mapping.
                    // In embeddings, the block tables are {seq_id: None}.
                    slot_mapping.extend(vec![PAD_SLOT_ID; sequence_length]);
                    continue;
                }

                // Compute the slot mapping.
                let block_table = sequence_group_metadata
                    .block_tables
                    .get(sequence_id)
                    .expect("Block table should exist for a sequence on decoding phase");

                // Mask the [0, start_idx) tokens of the prompt with
                // _PAD_SLOT_ID, where start_idx is max(0, seq_len -
                // sliding_window). For example, if the prompt len is 10,
                // sliding window is 8, and block size is 4, the first two
                // tokens are masked and the slot mapping will be
                // [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
                let start_index = self
                    .model
                    .sliding_window()
                    .map(|sw| query_length.checked_sub(sw).unwrap_or(0))
                    .unwrap_or(0);

                slot_mapping.extend((context_length..sequence_length).map(|i| {
                    if i < start_index {
                        PAD_SLOT_ID
                    } else {
                        let block_number = block_table[i / self.cache_engine.block_size];
                        let block_offset = i % self.cache_engine.block_size;
                        ((block_number as usize) * self.cache_engine.block_size + block_offset)
                            as i64
                    }
                }));
            }
        }

        let batch_size = input_tokens.len();
        let max_query_len = *query_lengths.iter().max().unwrap_or(&0);
        let max_prefill_seq_len = *prefill_sequence_lengths.iter().max().unwrap_or(&0);
        let max_decode_seq_len = *decode_sequence_lengths.iter().max().unwrap_or(&0);

        let max_block_table_len = block_tables.iter().map(|bt| bt.len()).max().unwrap();
        let block_tables_tensor =
            utils::make_tensor_with_pad(block_tables, max_block_table_len, 0, &self.device)?;

        let sequence_lengths: Vec<_> = sequence_lengths.into_iter().map(|s| s as i64).collect();
        let seq_lens_tensor = Tensor::new(sequence_lengths, &self.device)?;
        let mut seq_start_loc =
            Tensor::zeros(seq_lens_tensor.dims1()? + 1, DType::U32, &self.device)?;

        let cumsum_seq_lens = seq_lens_tensor.cumsum(0)?;
        seq_start_loc = seq_start_loc.slice_assign(&[1..], &cumsum_seq_lens)?;

        let input_tokens_tensor =
            Tensor::new(input_tokens, &self.device)?.reshape((input_tokens.len(),))?;
        let input_positions_tensor = Tensor::new(input_positions, &self.device)?;
        let slot_mapping_tensor = Tensor::new(slot_mapping, &self.device)?;

        let context_lens_tensor = Tensor::new(context_lengths, &self.device)?;
        let query_lens_tensor = Tensor::new(query_lengths, &self.device)?;
        let mut query_start_loc =
            Tensor::zeros(query_lens_tensor.dims1()? + 1, DType::U32, &self.device)?;

        let cumsum_query_lens = query_lens_tensor.cumsum(0)?;
        query_start_loc = query_start_loc.slice_assign(&[1..], &cumsum_query_lens)?;

        let attention_prefill_metadata = if num_prefills > 0 {
            Some(FlashAttentionPrefillMetadata {
                max_query_length: Some(max_query_len),
                max_prefill_sequence_length: max_prefill_seq_len,
                query_start_locations: Some(query_start_loc),
                sequence_start_locations: seq_start_loc,
                sequence_lengths: Some(seq_lens_tensor),
                block_tables: None, // TODO: this is valid only for non-cached KV cache for prefill, add logic for it
            })
        } else {
            None
        };
        let attention_decoding_metadata = if num_decode_tokens > 0 {
            Some(FlashAttentionDecodingMetadata {
                block_tables: Some(block_tables_tensor),
                max_decoding_sequence_length: max_decode_seq_len,
                sequence_lengths: Some(seq_lens_tensor),
            })
        } else {
            None
        };
        let attention_metadata = FlashAttentionMetadata {
            prefill_metadata: attention_prefill_metadata,
            decoding_metadata: attention_decoding_metadata,
            context_lengths: Some(context_lens_tensor),
            slot_mapping: slot_mapping_tensor,
            num_prefill_tokens,
            num_decoding_tokens: num_decode_tokens,
        };

        Ok(ModelInput {
            input_tokens_tensor,
            input_positions: input_positions_tensor,
            num_decode_tokens,
            num_prefills,
            attention_metadata,
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

/// `CacheEngine` - Manages the KV cache.
///
/// This class is responsible for initializing and managing the GPU and CPU KV
/// caches. It also provides methods for performing KV cache operations, such
/// as swapping and copying.
pub struct CacheEngine {
    /// Block size
    block_size: usize,
    /// GPU device
    device: Device,
    /// Model's Cache dtype
    dtype: DType,
    /// The LLM head size
    head_size: usize,
    /// Number of layers
    num_layers: usize,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Number of CPU blocks
    num_cpu_blocks: usize,
    /// Number of GPU blocks
    num_gpu_blocks: usize,
    /// Paged attention backend
    paged_attention: PagedAttention,
    /// The CPU cache
    cpu_cache: Vec<Tensor>,
    /// The GPU cache
    gpu_cache: Vec<Tensor>,
    /// Tracing span
    span: Span,
}

impl CacheEngine {
    /// Constructor
    pub fn new(
        cache_config: &CacheConfig,
        device: Device,
        dtype: DType,
        head_size: usize,
        num_attention_heads: usize,
        num_layers: usize,
        num_kv_heads: usize,
        sliding_window: Option<usize>,
    ) -> Result<Self, CacheEngineError> {
        let mut this = Self {
            block_size: cache_config.block_size(),
            device,
            dtype,
            head_size: head_size,
            num_layers,
            num_kv_heads,
            num_cpu_blocks: cache_config.num_cpu_blocks(),
            num_gpu_blocks: cache_config.num_gpu_blocks(),
            paged_attention: PagedAttention::new(
                num_attention_heads,
                head_size,
                num_kv_heads,
                sliding_window,
                dtype,
                cache_config.cache_dtype(),
                cache_config.block_size,
            ),
            cpu_cache: vec![],
            gpu_cache: vec![],
            span: info_span!("cache-engine"),
        };

        this.cpu_cache = this.allocate_blocks(this.num_cpu_blocks, &Device::Cpu)?;
        this.gpu_cache = this.allocate_blocks(this.num_gpu_blocks, &device)?;

        Ok(this)
    }

    /// Allocates KV cache blocks, on the specified blocks
    #[instrument(skip_all)]
    fn allocate_blocks(
        &mut self,
        num_blocks: usize,
        device: &Device,
    ) -> Result<Vec<Tensor>, CacheEngineError> {
        let _enter = self.span.enter();
        let kv_cache_shape = self.paged_attention.get_kv_cache_shape(
            num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_size,
        );
        let mut kv_caches = Vec::with_capacity(self.num_layers);
        for _ in 0..self.num_layers {
            kv_caches.push(Tensor::zeros(kv_cache_shape, self.dtype, &device)?);
        }

        Ok(kv_caches)
    }

    /// Swaps CPU blocks into GPU physical blocks
    #[instrument(skip_all)]
    pub fn swap_in(&mut self, blocks_to_swap_in: Tensor) -> Result<(), CacheEngineError> {
        let _enter = self.span.enter();
        for i in 0..self.num_layers {
            self.paged_attention.swap_blocks(
                self.cpu_cache[i],
                self.gpu_cache[i],
                blocks_to_swap_in,
            )?
        }
        Ok(())
    }

    /// Swaps GPU blocks out to CPU
    #[instrument(skip_all)]
    pub fn swap_out(&mut self, blocks_to_swap_out: Tensor) -> Result<(), CacheEngineError> {
        let _enter = self.span.enter();
        for i in 0..self.num_layers {
            self.paged_attention.swap_blocks(
                self.gpu_cache[i],
                self.cpu_cache[i],
                blocks_to_swap_out,
            )?
        }
        Ok(())
    }

    /// Copy blocks
    #[instrument(skip_all)]
    pub fn copy_blocks(&mut self, blocks_to_copy: Tensor) -> Result<(), CacheEngineError> {
        let _enter = self.span.enter();
        Ok(self
            .paged_attention
            .copy_blocks(&mut self.gpu_cache, blocks_to_copy)?)
    }
}

#[derive(Debug, Error)]
pub enum CacheEngineError {
    #[error("Candle error: `{0}`")]
    CandleError(#[from] CandleError),
    #[error("DType parse error: `{0}`")]
    DTypeParseError(#[from] DTypeParseError),
}

pub(crate) mod utils {
    use candle::{Device, Tensor, WithDType};

    use super::ModelWorkerError;

    pub(crate) fn make_tensor_with_pad<D: WithDType>(
        x: Vec<Vec<D>>,
        max_length: usize,
        pad: D,
        device: &Device,
    ) -> Result<Tensor, ModelWorkerError> {
        let mut padded_output = Vec::new();
        for mut x_i in x {
            x_i.extend([pad].repeat(max_length - x_i.len()));
            let shape = (x_i.len(),);
            padded_output.push(Tensor::from_vec(x_i, shape, device)?);
        }
        Ok(Tensor::cat(&padded_output[..], 0)?)
    }
}
