use super::config::CacheConfig;
use candle::{DType, DTypeParseError, Device, Error as CandleError, Shape, Tensor};
use std::{collections::HashMap, str::FromStr};
use thiserror::Error;
use tracing::{error, info, info_span, instrument, warn, Span};

const AVAILABLE_GPU_MEMORY_RATIO: f32 = 0.8;

/// `SequenceInfo` - Metadata regarding
/// the current sequence in the batch
pub struct SequenceInfo {
    /// Current position in batch
    position_in_batch: usize,
    /// The current sequence length
    current_len: usize,
}

/// `CacheEngine` - Manages the KV cache.
///
/// This class is responsible for initializing and managing the GPU and CPU KV
/// caches. It also provides methods for performing KV cache operations, such
/// as swapping and copying.
pub struct CacheEngine {
    /// The current active sequences.
    /// It is a mapping from `sequence_id` to `SequenceInfo`
    active_sequences: HashMap<u64, SequenceInfo>,
    /// Cache engine config
    config: CacheConfig,
    /// Maximum batch size
    max_batch_size: usize,
    /// Number of layers
    num_layers: usize,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// The CPU cache
    cpu_cache: Vec<Tensor>,
    /// The GPU cache
    gpu_cache: Vec<Tensor>,
    /// Tracing span
    span: Span,
}

impl CacheEngine {
    /// Constructor
    #[instrument(skip_all)]
    pub fn new(
        config: CacheConfig,
        head_dim: usize,
        num_layers: usize,
        num_kv_heads: usize,
    ) -> Result<Self, CacheEngineError> {
        let span = info_span!("cache-engine");
        let span_clone = span.clone();
        let _enter = span_clone.enter();
        info!("Starting a new `CacheEngine` instance");

        let device = Device::new_cuda(config.device_id)?;
        let dtype = DType::from_str(&config.dtype)?;
        let max_batch_size = utils::compute_max_batch_size(
            dtype,
            config.max_seq_len,
            num_layers,
            num_kv_heads,
            head_dim,
        )?;

        let cpu_cache = Self::allocate_cache(
            &Device::Cpu,
            num_layers,
            max_batch_size,
            config.max_seq_len,
            num_kv_heads,
            head_dim,
            dtype,
        )?;
        let gpu_cache = Self::allocate_cache(
            &device,
            num_layers,
            max_batch_size,
            config.max_seq_len,
            num_kv_heads,
            head_dim,
            dtype,
        )?;

        Ok(Self {
            active_sequences: HashMap::new(),
            config,
            max_batch_size,
            num_layers,
            num_kv_heads,
            head_dim,
            cpu_cache,
            gpu_cache,
            span,
        })
    }

    /// Allocates KV cache, at initialization
    #[instrument(skip_all)]
    fn allocate_cache(
        device: &Device,
        num_layers: usize,
        max_batch_size: usize,
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Result<Vec<Tensor>, CacheEngineError> {
        let mut cache = Vec::with_capacity(num_layers);
        let shape = Shape::from((2, max_batch_size, max_seq_len, num_heads, head_dim));
        for _ in 0..num_layers {
            cache.push(Tensor::zeros(shape.clone(), dtype, device)?);
        }
        Ok(cache)
    }

    /// Adds a new sequence KV tensors to the cache
    #[instrument(skip_all)]
    pub fn add_sequence(
        &mut self,
        sequence_id: u64,
        sequence_len: usize,
    ) -> Result<(), CacheEngineError> {
        info!("Adding sequence with sequence id = `{sequence_id}`");
        if sequence_len > self.config.max_seq_len {
            return Err(CacheEngineError::MaxSequenceLengthExceeded);
        }
        let position_in_batch = self.find_free_batch_index()?;
        self.active_sequences.insert(
            sequence_id,
            SequenceInfo {
                position_in_batch,
                current_len: sequence_len,
            },
        );
        Ok(())
    }

    /// Removes an already allocated sequence from the cache
    #[instrument(skip_all)]
    pub fn remove_sequence(&mut self, sequence_id: u64) -> Result<(), CacheEngineError> {
        info!("Removing sequence with sequence id = `{sequence_id}`");
        self.active_sequences.remove(&sequence_id);
        Ok(())
    }

    /// Updates the KV cache for a give allocated sequence
    #[instrument(skip_all)]
    pub fn update_sequence(
        &mut self,
        sequence_id: u64,
        new_kv: &[Tensor],
        new_tokens: usize,
    ) -> Result<(), CacheEngineError> {
        let seq_info = self
            .active_sequences
            .get_mut(&sequence_id)
            .ok_or(CacheEngineError::SequenceNotFound)?;
        let new_len = seq_info.current_len + new_tokens;
        if new_len > self.config.max_seq_len {
            return Err(CacheEngineError::MaxSequenceLengthExceeded);
        }
        for kv in new_kv.iter() {
            let (t, n, h, d) = kv.dims4()?;
            if t != 2 || n != new_len || h != self.num_kv_heads || d != self.head_dim {
                return Err(CacheEngineError::InvalidSequenceKvDims);
            }
        }
        for (layer, new_kv_layer) in self.gpu_cache.iter_mut().zip(new_kv.iter()) {
            // Assigns the new KV tensors to the cache
            // on the right positions according to the sequence
            // length and its position index in the current batch
            *layer = layer.slice_scatter(
                new_kv_layer,
                2,
                seq_info.position_in_batch * self.config.max_seq_len + new_len,
            )?;
        }
        seq_info.current_len = new_len;
        Ok(())
    }

    /// Finds a free batch index
    /// Finds a free batch index
    #[instrument(skip_all)]
    fn find_free_batch_index(&self) -> Result<usize, CacheEngineError> {
        for i in 0..self.max_batch_size {
            if !self
                .active_sequences
                .values()
                .any(|info| info.position_in_batch == i)
            {
                return Ok(i);
            }
        }
        warn!("No free batch index found");
        Err(CacheEngineError::NoBatchIndexAvailable)
    }
}

pub(crate) mod utils {
    use super::*;
    use candle::backend::BackendDevice;
    use cuda_runtime_sys::*;

    /// Computes the available GPU memory, in bytes
    pub(crate) unsafe fn compute_available_gpu_memory() -> Result<(usize, usize), CacheEngineError>
    {
        let mut free = 0;
        let mut total = 0;
        unsafe {
            let result = cudaMemGetInfo(&mut free, &mut total);
            if result != cudaError::cudaSuccess {
                error!("CUDA error: {:?}", result);
                return Err(CacheEngineError::CudaError(format!(
                    "CUDA error: {:?}",
                    result
                )));
            }
        }
        Ok((free, total))
    }

    pub(crate) fn compute_max_batch_size(
        dtype: DType,
        max_seq_len: usize,
        num_layers: usize,
        num_kv_heads: usize,
        hidden_dim: usize,
    ) -> Result<usize, CacheEngineError> {
        let (available_memory_in_bytes, _total_memory_in_bytes) =
            unsafe { compute_available_gpu_memory()? };
        let memory_to_allocate =
            (available_memory_in_bytes as f32 * AVAILABLE_GPU_MEMORY_RATIO) as usize;
        let num_dtype_bytes = dtype.size_in_bytes();
        // We need to multiply by 2 for key and value
        let num_elements = 2 * max_seq_len * num_layers * num_kv_heads * hidden_dim;
        let max_batch_size = memory_to_allocate / (num_dtype_bytes * num_elements);
        Ok(max_batch_size)
    }
}

#[derive(Debug, Error)]
pub enum CacheEngineError {
    #[error("Candle error: `{0}`")]
    CandleError(#[from] CandleError),
    #[error("DType parse error: `{0}`")]
    DTypeParseError(#[from] DTypeParseError),
    #[error("CUDA error: `{0}`")]
    CudaError(String),
    #[error("No batch index available")]
    NoBatchIndexAvailable,
    #[error("Invalid sequence Kv dims")]
    InvalidSequenceKvDims,
    #[error("Max sequence length exceeded")]
    MaxSequenceLengthExceeded,
    #[error("Sequence not found")]
    SequenceNotFound,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::IndexOp;

    #[test]
    fn test_compute_max_batch_size() {
        const MAX_SEQ_LEN: usize = 1024;
        const HEAD_DIM: usize = 128;
        const NUM_KV_HEADS: usize = 32;
        const NUM_LAYERS: usize = 32;
        const DTYPE: &str = "bf16";

        let mut cache = CacheEngine::new(
            CacheConfig {
                device_id: 0,
                dtype: DTYPE.to_string(),
                max_seq_len: MAX_SEQ_LEN,
            },
            HEAD_DIM,
            NUM_KV_HEADS,
            NUM_LAYERS,
        )
        .unwrap();
        let max_batch_size = cache.max_batch_size;
        assert!(max_batch_size > 0);

        let device = Device::new_cuda(0).unwrap();
        let dtype = DType::from_str("bf16").unwrap();

        let new_max_batch_size =
            utils::compute_max_batch_size(dtype, MAX_SEQ_LEN, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)
                .unwrap();
        
        assert_eq!(((1. - AVAILABLE_GPU_MEMORY_RATIO) * (max_batch_size as f32)) as usize, new_max_batch_size);

        let sequence_token_len = 10;
        for i in 0..max_batch_size {
            cache.add_sequence(i as u64, 10).unwrap();
            assert_eq!(cache.active_sequences.len(), i + 1);
        }
        assert_eq!(cache.active_sequences.len(), max_batch_size);

        assert!(cache.add_sequence(max_batch_size as u64, sequence_token_len).is_err());

        cache.remove_sequence(max_batch_size as u64 - 1).unwrap();
        assert_eq!(cache.active_sequences.len(), max_batch_size - 1);

        let kvs = std::iter::repeat_with(|| {
            Tensor::rand(
                0f32,
                10f32,
                (2, max_batch_size, sequence_token_len, NUM_KV_HEADS, HEAD_DIM),
                &device,
            )?
            .to_dtype(dtype)
        })
        .take(NUM_LAYERS)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        cache.update_sequence(0, kvs.as_slice(), 1024).unwrap();

        for (cache_tensor, kv_tensor) in cache.gpu_cache.iter().zip(kvs.iter()) {
            assert_eq!(
                cache_tensor
                    .i((.., 0, .., .., ..))
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                kv_tensor
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            );
        }
    }
}
