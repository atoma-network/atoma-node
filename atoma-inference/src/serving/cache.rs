/// `CacheEngine` - Manages the KV cache.
///
/// This class is responsible for initializing and managing the GPU and CPU KV
/// caches. It also provides methods for performing KV cache operations, such
/// as swapping and copying.
pub struct CacheEngine {
    /// Block size
    block_size: usize,
    /// Model's Cache dtype
    dtype: DType,
    /// Number of layers
    num_layers: usize,
    /// Number of CPU blocks
    num_cpu_blocks: usize,
    /// Number of GPU blocks
    num_gpu_blocks: usize,
    /// Flash attention backend,
    /// compatible with paged attention
    attention: FlashAttention,
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
        device: Device,
        dtype: DType,
        alibi_slopes: Option<&Tensor>,
        head_dim: usize,
        num_attention_heads: usize,
        num_layers: usize,
        num_kv_heads: usize,
        softmax_scale: f32,
        sliding_window: Option<usize>,
    ) -> Result<Self, CacheEngineError> {
        info!("Starting a new `CacheEngine` instance");
        let mut this = Self {
            dtype,
            num_layers,
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
        let kv_cache_shape = FlashAttention::get_kv_cache_shape(
            num_blocks,
            self.block_size,
            self.attention.num_kv_heads,
            self.attention.head_dim,
        );
        let mut kv_caches = Vec::with_capacity(self.num_layers);
        for _ in 0..self.num_layers {
            kv_caches.push(Tensor::zeros(kv_cache_shape.clone(), self.dtype, &device)?);
        }

        Ok(kv_caches)
    }

    /// Swaps CPU blocks into GPU physical blocks
    #[instrument(skip_all)]
    pub fn swap_in(
        &mut self,
        blocks_to_swap_in: &HashMap<u32, u32>,
    ) -> Result<(), CacheEngineError> {
        let _enter = self.span.enter();
        for i in 0..self.num_layers {
            self.attention.swap_blocks(
                &self.cpu_cache[i],
                &mut self.gpu_cache[i],
                blocks_to_swap_in,
            )?
        }
        Ok(())
    }

    /// Swaps GPU blocks out to CPU
    #[instrument(skip_all)]
    pub fn swap_out(
        &mut self,
        blocks_to_swap_out: &HashMap<u32, u32>,
    ) -> Result<(), CacheEngineError> {
        let _enter = self.span.enter();
        for i in 0..self.num_layers {
            self.attention.swap_blocks(
                &self.gpu_cache[i],
                &mut self.cpu_cache[i],
                blocks_to_swap_out,
            )?
        }
        Ok(())
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
    use candle::{Device, IndexOp, Tensor, WithDType};

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
            let shape = (1, x_i.len());
            padded_output.push(Tensor::from_vec(x_i, shape, device)?);
        }
        Ok(Tensor::cat(&padded_output[..], 0)?)
    }

    /// Computes selected token indices, for each sequence in the batch.
    /// For a given sequence, the associated selected token index should
    /// correspond to the right end of the sequence, in the output tensor
    pub(crate) fn compute_selected_token_indices(
        cumulative_query_lengths: &Tensor,
    ) -> Result<Tensor, ModelWorkerError> {
        let length = cumulative_query_lengths.dims()[0] - 1;
        let ones = Tensor::ones(
            (length,),
            cumulative_query_lengths.dtype(),
            cumulative_query_lengths.device(),
        )?;
        Ok(cumulative_query_lengths.i(1..)?.sub(&ones)?)
    }
}
