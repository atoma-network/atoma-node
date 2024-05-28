use std::time::Duration;

use thiserror::Error;

const GB: usize = 1 << 30;

/// Configuration for the KV cache.
///
/// Args:
///   block_size: Size of a cache block in number of tokens.
///   gpu_memory_utilization: Fraction of GPU memory to use for the
///       vLLM execution.
///   swap_space: Size of the CPU swap space per GPU (in GiB).
///   cache_dtype: Data type for kv cache storage.
///   num_gpu_blocks_override: Number of GPU blocks to use. This overrides the
///       profiled num_gpu_blocks if specified. Does nothing if None.
///   sliding_window: Optional sliding window size.
///   num_gpu_blocks: Number of GPU blocks
///   num_cpu_block: Number of CPU blocks
pub struct CacheConfig {
    /// Block size
    block_size: usize,
    /// GPU memory utilization
    gpu_memory_utilization: f32,
    /// Swap space bytes
    swap_space_bytes: usize,
    /// Cache dtype
    cache_dtype: String,
    /// Number of GPU blocks to override (optional)
    num_gpu_blocks_override: Option<usize>,
    /// Sliding window (optional)
    sliding_window: Option<usize>,
    /// Number of GPU blocks
    num_gpu_blocks: usize,
    /// Number of CPU blocks
    num_cpu_blocks: usize,
}

impl CacheConfig {
    /// Constructor
    pub fn new(
        block_size: usize,
        gpu_memory_utilization: f32,
        swap_space: usize,
        cache_dtype: String,
        num_gpu_blocks_override: Option<usize>,
        sliding_window: Option<usize>,
        num_cpu_blocks: usize,
        num_gpu_blocks: usize,
    ) -> Result<Self, CacheConfigError> {
        let this = Self {
            block_size,
            gpu_memory_utilization,
            swap_space_bytes: swap_space * GB,
            cache_dtype,
            num_gpu_blocks_override,
            sliding_window,
            num_gpu_blocks,
            num_cpu_blocks,
        };

        this.verify_args()?;
        // this.verify_cache_dtype()?;
        Ok(this)
    }

    /// Verify `CacheConfig` arguments
    fn verify_args(&self) -> Result<(), CacheConfigError> {
        if self.gpu_memory_utilization > 1.0 {
            return Err(CacheConfigError::InvalidGpuMemoryUtilization(
                self.gpu_memory_utilization,
            ));
        }
        Ok(())
    }

    /// Getter for `block_size`
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Getter for `gpu_memory_utilization`
    pub fn gpu_memory_utilization(&self) -> f32 {
        self.gpu_memory_utilization
    }

    /// Getter for `swap_space_bytes`
    pub fn swap_space_bytes(&self) -> usize {
        self.swap_space_bytes
    }

    /// Getter for `sliding_window`
    pub fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }

    /// Getter for `num_gpu_block_override`
    pub fn num_gpu_block_override(&self) -> Option<usize> {
        self.num_gpu_blocks_override
    }

    /// Getter for `num_cpu_blocks`
    pub fn num_cpu_blocks(&self) -> usize {
        self.num_cpu_blocks
    }

    /// Getter for `num_gpu_blocks`
    pub fn num_gpu_blocks(&self) -> usize {
        self.num_gpu_blocks
    }
}

#[derive(Debug, Error)]
pub enum CacheConfigError {
    #[error("Invalid GPU memory utilization: `{0}`")]
    InvalidGpuMemoryUtilization(f32),
}

/// Scheduler configuration.

/// Args:
///   max_num_batched_tokens: Maximum number of tokens to be processed in
///      a single iteration.
///   max_num_sequences: Maximum number of sequences to be processed in a single
///      iteration.
///   max_sequence_len: Maximum length of a sequence (including prompt
///      and generated text).
///   delay_factor: Apply a delay (of delay factor multiplied by previous
///      prompt latency) before scheduling next prompt.
///   enable_chunked_prefill: If true, prefill requests can be chunked based
///      on the remaining max_num_batched_tokens.
pub struct SchedulerConfig {
    /// Maximum number of batched tokens
    max_num_batched_tokens: usize,
    /// Maxinum number of sequences
    max_num_sequences: usize,
    /// Maximum sequence length
    max_sequence_length: usize,
    /// Delay factor
    delay_factor: Duration,
    /// Enable chunked prefill
    enable_chunked_prefill: bool,
    /// Device ordinal
    device: usize,
}

impl SchedulerConfig {
    /// Constructor
    pub fn new(
        max_num_batched_tokens: usize,
        max_num_sequences: usize,
        max_sequence_length: usize,
        delay_factor: Duration,
        enable_chunked_prefill: bool,
        device: usize,
    ) -> Result<Self, SchedulerConfigError> {
        let this = Self {
            max_num_batched_tokens,
            max_num_sequences,
            max_sequence_length,
            delay_factor,
            enable_chunked_prefill,
            device,
        };

        this.verify_args()?;
        Ok(this)
    }

    fn verify_args(&self) -> Result<(), SchedulerConfigError> {
        if self.max_num_batched_tokens < self.max_sequence_length && !self.enable_chunked_prefill {
            return Err(SchedulerConfigError::FailedVerifySchedulerConfig(format!(
                "max_num_batched_tokens ({}) is smaller than max_sequence_length ({}). This effectively limits the maximum sequence length to max_num_batched_tokens and makes the scheduler reject longer sequences. Please increase max_num_batched_tokens or decrease max_sequence_length.",
                self.max_num_batched_tokens, self.max_sequence_length
            )));
        }

        if self.max_num_batched_tokens < self.max_num_sequences {
            return Err(SchedulerConfigError::FailedVerifySchedulerConfig(format!(
                "max_num_batched_tokens ({}) must be greater than or equal to max_num_sequences ({}).",
                self.max_num_batched_tokens, self.max_num_sequences
            )));
        }

        Ok(())
    }

    /// Getter for `device` ordinal
    pub fn device(&self) -> usize {
        self.device
    }
}

#[derive(Debug, Error)]
pub enum SchedulerConfigError {
    #[error("Failed verify scheduler config: `{0}")]
    FailedVerifySchedulerConfig(String),
}
