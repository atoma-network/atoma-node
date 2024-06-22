use std::str::FromStr;

use crate::{
    block,
    config::{CacheConfig, ModelConfig, SchedulerConfig},
    model_executor::{ModelExecutor, ModelLoaderError},
    sequence::ExecuteModelRequest,
};
use candle::{DType, DTypeParseError, Device, Error as CandleError, Tensor};
use thiserror::Error;

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
}

impl<M> ModelWorker<M>
where
    M: ModelExecutor,
{
    /// Constructor
    pub async fn new(
        api_key: String,
        cache_config: CacheConfig,
        device: Device,
        dtype: DType,
        model_name: String, 
        revision: String,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, ModelWorkerError> {
        let file_paths = M::fetch(api_key, model_name, revision).await?;
        let model = M::load(file_paths).await?;
        let cache_engine = CacheEngine::new(
            &cache_config,
            model.head_size(),
            model.num_attention_heads(),
            model.num_layers(),
            model.num_kv_heads(),
            model.sliding_window(),
            dtype,
            device.clone(),
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
        })
    }

    /// Determines the number of available GPU blocks
    pub fn num_available_gpu_blocks(&self) -> usize {
        todo!()
    }

    /// Executes model's forward pass
    pub fn execute_model(&self, request: ExecuteModelRequest) -> Result<(), ModelWorkerError> {
        let ExecuteModelRequest {
            sequence_groups_metadata,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            running_queue_size,
        } = request;

        let num_sequence_groups = sequence_groups_metadata.len();

        // `blocks_to_swap_in` and `blocks_to_swap_out` are CPU tensors
        let blocks_to_swap_in =
            Tensor::new(blocks_to_swap_in, &candle::Device::Cpu)?.reshape((-1, 2))?;
        let blocks_to_swap_out =
            Tensor::new(blocks_to_swap_out, &candle::Device::Cpu)?.reshape((-1, 2))?;

        // `blocks_to_copy` is a gpu tensor. The src and tgt of
        // blocks to copy are in the same device, and `blocks_to_copy`
        // can be used directly within cuda kernels.
        let blocks_to_copy = Tensor::new(blocks_to_copy, &self.device)?.reshape((-1, 2))?;

        // At this point we need to perform cache swap operations
        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)?;

        Ok(())
    }

    /// Swaps cached blocks
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
            self.cache_engine.copy(blocks_to_copy)?
        }
        Ok(())
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
        let dtype = DType::from_str(dtype)?;
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
        };

        this.cpu_cache = this.allocate_blocks(this.num_cpu_blocks, &Device::Cpu)?;
        this.gpu_cache = this.allocate_blocks(this.num_gpu_blocks, &device)?;

        Ok(this)
    }

    /// Allocates KV cache blocks, on the specified blocks
    fn allocate_blocks(
        &mut self,
        num_blocks: usize,
        device: &Device,
    ) -> Result<Vec<Tensor>, CacheEngineError> {
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
    pub fn swap_in(&mut self, blocks_to_swap_in: Tensor) -> Result<(), CacheEngineError> { 
        for i in 0..self.num_layers { 
            self.paged_attention.swap_blocks(self.cpu_cache[i], self.gpu_cache[i], blocks_to_swap_in)?
        }
        Ok(())
    } 

    /// Swaps GPU blocks out to CPU
    pub fn swap_out(&mut self, blocks_to_swap_out: Tensor) -> Result<(), CacheEngineError> { 
        for i in 0..self.num_layers { 
            self.paged_attention.swap_blocks(self.gpu_cache[i], self.cpu_cache[i], blocks_to_swap_out)?
        }
        Ok(())
    }

    /// Copy blocks
    pub fn copy_blocks(&mut self, blocks_to_copy: Tensor) -> Result<(), CacheEngineError> { 
        Ok(self.paged_attention.copy_blocks(self.gpu_cache, blocks_to_copy)?)
    }
}

#[derive(Debug, Error)]
pub enum CacheEngineError {
    #[error("Candle error: `{0}`")]
    CandleError(#[from] CandleError),
    #[error("DType parse error: `{0}`")]
    DTypeParseError(#[from] DTypeParseError),
}
