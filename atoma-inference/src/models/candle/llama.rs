use std::{path::PathBuf, str::FromStr, time::Instant};

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::llama::{Config, LlamaConfig},
};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

use candle_transformers::models::llama as model;
use tokenizers::Tokenizer;
use tracing::info;

use crate::models::{
    config::ModelConfig,
    token_output_stream::TokenOutputStream,
    types::{LlmLoadData, ModelType, TextModelInput, TextModelOutput},
    ModelError, ModelTrait,
};

use super::{device, hub_load_safetensors};

const EOS_TOKEN: &str = "</s>";

#[allow(dead_code)]
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
enum Which {
    V1,
    V2,
    Solar10_7B,
    TinyLlama1_1BChat,
}

pub struct LlamaModel {
    device: Device,
    model: model::Llama,
    model_type: ModelType,
    tokenizer: Tokenizer,
    config: Config,
    dtype: DType,
}

impl ModelTrait for LlamaModel {
    type Input = TextModelInput;
    type Output = TextModelOutput;
    type LoadData = LlmLoadData;

    fn fetch(
        api_key: String,
        cache_dir: PathBuf,
        config: ModelConfig,
    ) -> Result<Self::LoadData, ModelError> {
        let device = device(config.device_id())?;
        let dtype = DType::from_str(&config.dtype())?;

        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .with_cache_dir(cache_dir)
            .build()?;

        let model_type = ModelType::from_str(&config.model_id())?;
        let repo_id = model_type.repo().to_string();
        let revision = model_type.default_revision().to_string();

        let repo = api.repo(Repo::with_revision(
            repo_id.clone(),
            RepoType::Model,
            revision,
        ));
        let config_file_path = repo.get("config.json")?;
        let tokenizer_file_path = repo.get("tokenizer.json")?;

        let model_weights_file_paths = if &repo_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0" {
            vec![repo.get("model.safetensors")?]
        } else {
            hub_load_safetensors(&repo, "model.safetensors.index.json")?
        };

        let mut file_paths = Vec::with_capacity(2 + model_weights_file_paths.len());
        file_paths.extend(vec![config_file_path, tokenizer_file_path]);
        file_paths.extend(model_weights_file_paths);

        Ok(Self::LoadData {
            device,
            dtype,
            file_paths,
            model_type: ModelType::from_str(&config.model_id())?,
            use_flash_attention: config.use_flash_attention(),
        })
    }

    fn model_type(&self) -> ModelType {
        self.model_type.clone()
    }

    fn load(load_data: Self::LoadData) -> Result<Self, ModelError> {
        info!("Loading Llama model ...");

        let start = Instant::now();

        let device = load_data.device;
        let dtype = load_data.dtype;
        let (model, tokenizer_filename, config) = {
            let config_filename = load_data.file_paths[0].clone();
            let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;

            let tokenizer_filename = load_data.file_paths[1].clone();
            let config = config.into_config(load_data.use_flash_attention);

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&load_data.file_paths[2..], dtype, &device)?
            };
            (model::Llama::load(vb, &config)?, tokenizer_filename, config)
        };
        let tokenizer = Tokenizer::from_file(tokenizer_filename)?;
        info!("Loaded Llama model in {:?}", start.elapsed());

        Ok(Self {
            device,
            model,
            model_type: load_data.model_type,
            tokenizer,
            config,
            dtype,
        })
    }

    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN);
        let mut tokens = self
            .tokenizer
            .encode(input.prompt.clone(), true)?
            .get_ids()
            .to_vec();

        let mut tokenizer = TokenOutputStream::new(self.tokenizer.clone());
        let mut logits_processor = LogitsProcessor::new(
            input.random_seed,
            Some(input.temperature),
            Some(input.top_p),
        );
        let mut index_pos = 0;
        let mut res = String::new();
        let mut generated_tokens = 0;

        let start_gen = Instant::now();
        let mut cache = model::Cache::new(true, self.dtype, &self.config, &self.device)?;
        for index in 0..input.max_tokens {
            let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input_tensor = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input_tensor, context_index, &mut cache)?;
            let logits = logits.squeeze(0)?;
            let logits = if input.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(input.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    input.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            index_pos += ctxt.len();
            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);

            if Some(next_token) == eos_token_id {
                break;
            }
            if let Some(t) = tokenizer.next_token(next_token)? {
                res += &t;
            }

            generated_tokens += 1;
        }
        if let Some(rest) = tokenizer.decode_rest()? {
            res += &rest;
        }

        let dt = start_gen.elapsed();
        info!(
            "{generated_tokens} tokens generated ({} token/s)\n",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        Ok(TextModelOutput {
            text: res,
            time: dt.as_secs_f64(),
            tokens_count: generated_tokens,
        })
    }
}

mod multi_processor {
    use candle::backend::BackendStorage;
    use candle::{CpuStorage, CustomOp1, DType, Device, IndexOp, Layout, Result, Shape, Tensor, D};
    use candle_nn::{Embedding, Linear, Module, RmsNorm};
    use cudarc::nccl::safe::{Comm, ReduceOp};
    use half::{bf16, f16};
    use serde::Deserialize;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;

    const MAX_SEQ_LEN: usize = 4096;
    struct TensorParallelColumnLinear {
        linear: Linear,
    }

    impl TensorParallelColumnLinear {
        fn new(linear: Linear) -> Self {
            Self { linear }
        }
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            self.linear.forward(x)
        }
    }

    struct TensorParallelRowLinear {
        linear: Linear,
        comm: Rc<Comm>,
    }

    struct AllReduce {
        comm: Rc<Comm>,
    }

    /// This is actually not safe: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
    /// But for this example purposes, this will work
    unsafe impl Sync for AllReduce {}
    /// This is actually not safe: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
    /// But for this example purposes, this will work
    unsafe impl Send for AllReduce {}

    impl CustomOp1 for AllReduce {
        fn name(&self) -> &'static str {
            "allreduce"
        }

        fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
            todo!("implement allreduce for cpu is not necessary for single node");
        }

        #[cfg(feature = "cuda")]
        fn cuda_fwd(
            &self,
            s: &candle::CudaStorage,
            l: &Layout,
        ) -> Result<(candle::CudaStorage, Shape)> {
            use candle::cuda_backend::WrapErr;
            let elem_count = l.shape().elem_count();
            let dev = s.device().clone();
            match s.dtype() {
                DType::BF16 => {
                    let s = s.as_cuda_slice::<bf16>()?;
                    // let s = match l.contiguous_offsets() {
                    //     None => Err(Error::Wrapped("input has to be contiguous".into()))?,
                    //     Some((o1, o2)) => s.slice(o1..o2),
                    // };
                    let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                    self.comm.all_reduce(s, &mut dst, &ReduceOp::Sum).unwrap();
                    let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev);
                    Ok((dst, l.shape().clone()))
                }
                DType::F16 => {
                    let s = s.as_cuda_slice::<f16>()?;
                    // let s = match l.contiguous_offsets() {
                    //     None => Err(Error::Wrapped("input has to be contiguous".into()))?,
                    //     Some((o1, o2)) => s.slice(o1..o2),
                    // };
                    let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
                    self.comm.all_reduce(s, &mut dst, &ReduceOp::Sum).unwrap();
                    let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev);
                    Ok((dst, l.shape().clone()))
                }
                dtype => candle::bail!("unsupported dtype {dtype:?}"),
            }
        }
    }

    fn all_reduce_sum(x: &Tensor, comm: &Rc<Comm>) -> Result<Tensor> {
        x.apply_op1(AllReduce { comm: comm.clone() })
    }

    impl TensorParallelRowLinear {
        fn new(linear: Linear, comm: Rc<Comm>) -> Self {
            Self { linear, comm }
        }
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let x = self.linear.forward(x)?;
            all_reduce_sum(&x, &self.comm)
        }
    }

    fn shard(dim: usize, rank: usize, world_size: usize) -> candle_nn::var_builder::Shard {
        candle_nn::var_builder::Shard {
            dim,
            rank,
            world_size,
        }
    }

    impl TensorParallelColumnLinear {
        fn load(vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
            let rank = comm.rank();
            let size = comm.world_size();
            let weight = vb.get_with_hints((), "weight", shard(0, rank, size))?;
            Ok(Self::new(Linear::new(weight, None)))
        }

        fn load_multi(vb: VarBuilder, prefixes: &[&str], comm: Rc<Comm>) -> Result<Self> {
            let rank = comm.rank();
            let size = comm.world_size();
            let weights: Vec<_> = prefixes
                .iter()
                .map(|p| vb.pp(p).get_with_hints((), "weight", shard(0, rank, size)))
                .collect::<Result<Vec<_>>>()?;
            let weight = Tensor::cat(&weights, 0)?;
            Ok(Self::new(Linear::new(weight, None)))
        }
    }

    impl TensorParallelRowLinear {
        fn load(vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
            let rank = comm.rank();
            let size = comm.world_size();
            let weight = vb.get_with_hints((), "weight", shard(1, rank, size))?;
            Ok(Self::new(Linear::new(weight, None), comm))
        }
    }

    #[derive(Deserialize)]
    pub struct Config {
        pub hidden_size: usize,
        pub intermediate_size: usize,
        pub vocab_size: usize,
        pub num_hidden_layers: usize,
        pub num_attention_heads: usize,
        pub num_key_value_heads: usize,
        pub rms_norm_eps: f64,
        #[serde(default = "default_rope")]
        pub rope_theta: f32,
    }

    fn default_rope() -> f32 {
        10_000.0
    }

    #[derive(Clone)]
    pub struct Cache {
        #[allow(clippy::type_complexity)]
        kvs: Arc<Mutex<Vec<Option<(Tensor, Tensor)>>>>,
        cos: Tensor,
        sin: Tensor,
    }

    impl Cache {
        pub fn new(dtype: DType, config: &Config, device: &Device) -> Result<Self> {
            // precompute freqs_cis
            let n_elem = config.hidden_size / config.num_attention_heads;
            let theta: Vec<_> = (0..n_elem)
                .step_by(2)
                .map(|i| 1f32 / config.rope_theta.powf(i as f32 / n_elem as f32))
                .collect();
            let theta = Tensor::new(theta.as_slice(), device)?;
            let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
                .to_dtype(DType::F32)?
                .reshape((MAX_SEQ_LEN, 1))?
                .matmul(&theta.reshape((1, theta.elem_count()))?)?;
            // This is different from the paper, see:
            // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
            let cos = idx_theta.cos()?.to_dtype(dtype)?;
            let sin = idx_theta.sin()?.to_dtype(dtype)?;
            Ok(Self {
                kvs: Arc::new(Mutex::new(vec![None; config.num_hidden_layers])),
                cos,
                sin,
            })
        }
    }

    fn silu(xs: &Tensor) -> Result<Tensor> {
        xs / (xs.neg()?.exp()? + 1.0)?
    }

    fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<Linear> {
        let weight = vb.get((size2, size1), "weight")?;
        Ok(Linear::new(weight, None))
    }

    fn embedding(cfg: &Config, vb: VarBuilder) -> Result<Embedding> {
        let embeddings = vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        Ok(Embedding::new(embeddings, cfg.hidden_size))
    }

    struct CausalSelfAttention {
        qkv_proj: TensorParallelColumnLinear,
        o_proj: TensorParallelRowLinear,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        cache: Cache,
    }

    impl CausalSelfAttention {
        fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
            let (_b_sz, _, seq_len, _hidden_size) = x.shape().dims4()?;
            let cos = self.cache.cos.narrow(0, index_pos, seq_len)?;
            let sin = self.cache.sin.narrow(0, index_pos, seq_len)?;
            candle_nn::rotary_emb::rope(x, &cos, &sin)
        }

        fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize) -> Result<Tensor> {
            let (b_sz, seq_len, _) = x.shape().dims3()?;

            let qkv = self.qkv_proj.forward(x)?;
            let hidden_size = self.num_attention_heads * self.head_dim;

            let q = qkv.i((.., .., ..self.num_attention_heads * self.head_dim))?;
            let k = qkv.i((
                ..,
                ..,
                self.num_attention_heads * self.head_dim
                    ..self.num_attention_heads * self.head_dim
                        + self.num_key_value_heads * self.head_dim,
            ))?;
            let v = qkv.i((
                ..,
                ..,
                self.num_attention_heads * self.head_dim
                    + self.num_key_value_heads * self.head_dim..,
            ))?;
            // todo!("Q {:?} K {:?} V {:?} - x {:?}", q.shape(), k.shape(), v.shape(), x.shape());

            let q = q
                .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = k
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let mut v = v
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;

            let q = self.apply_rotary_emb(&q, index_pos)?;
            let mut k = self.apply_rotary_emb(&k, index_pos)?;

            let mut cache = self.cache.kvs.lock().unwrap();
            if let Some((cache_k, cache_v)) = &cache[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
                let k_seq_len = k.dims()[1];
                if k_seq_len > MAX_SEQ_LEN {
                    k = k
                        .narrow(D::Minus1, k_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
                let v_seq_len = v.dims()[1];
                if v_seq_len > 2 * MAX_SEQ_LEN {
                    v = v
                        .narrow(D::Minus1, v_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
            }
            cache[block_idx] = Some((k.clone(), v.clone()));

            let k = self.repeat_kv(k)?;
            let v = self.repeat_kv(v)?;
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            let y = candle_flash_attn::flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?
                .transpose(1, 2)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
            let y = self.o_proj.forward(&y)?;
            Ok(y)
        }

        fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
            let n_rep = self.num_attention_heads / self.num_key_value_heads;
            candle_transformers::utils::repeat_kv(x, n_rep)
        }

        fn load(vb: VarBuilder, cache: &Cache, cfg: &Config, comm: Rc<Comm>) -> Result<Self> {
            let qkv_proj = TensorParallelColumnLinear::load_multi(
                vb.clone(),
                &["q_proj", "k_proj", "v_proj"],
                comm.clone(),
            )?;
            let o_proj = TensorParallelRowLinear::load(vb.pp("o_proj"), comm.clone())?;
            Ok(Self {
                qkv_proj,
                o_proj,
                num_attention_heads: cfg.num_attention_heads / comm.world_size(),
                num_key_value_heads: cfg.num_key_value_heads / comm.world_size(),
                head_dim: cfg.hidden_size / cfg.num_attention_heads,
                cache: cache.clone(),
            })
        }
    }

    struct Mlp {
        c_fc1: TensorParallelColumnLinear,
        c_fc2: TensorParallelColumnLinear,
        c_proj: TensorParallelRowLinear,
    }

    impl Mlp {
        fn new(
            c_fc1: TensorParallelColumnLinear,
            c_fc2: TensorParallelColumnLinear,
            c_proj: TensorParallelRowLinear,
        ) -> Self {
            Self {
                c_fc1,
                c_fc2,
                c_proj,
            }
        }

        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let x = (silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
            self.c_proj.forward(&x)
        }

        fn load(vb: VarBuilder, _cfg: &Config, comm: Rc<Comm>) -> Result<Self> {
            let c_fc1 = TensorParallelColumnLinear::load(vb.pp("gate_proj"), comm.clone())?;
            let c_fc2 = TensorParallelColumnLinear::load(vb.pp("up_proj"), comm.clone())?;
            let c_proj = TensorParallelRowLinear::load(vb.pp("down_proj"), comm)?;
            Ok(Self::new(c_fc1, c_fc2, c_proj))
        }
    }

    struct Block {
        rms_1: RmsNorm,
        attn: CausalSelfAttention,
        rms_2: RmsNorm,
        mlp: Mlp,
    }

    fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
        let weight = vb.get_with_hints(size, "weight", shard(0, 0, 1))?;
        Ok(RmsNorm::new(weight, eps))
    }

    impl Block {
        fn new(rms_1: RmsNorm, attn: CausalSelfAttention, rms_2: RmsNorm, mlp: Mlp) -> Self {
            Self {
                rms_1,
                attn,
                rms_2,
                mlp,
            }
        }

        fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize) -> Result<Tensor> {
            let residual = x;
            let x = self.rms_1.forward(x)?;
            let x = (self.attn.forward(&x, index_pos, block_idx)? + residual)?;
            let residual = &x;
            let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
            Ok(x)
        }

        fn load(vb: VarBuilder, cache: &Cache, cfg: &Config, comm: Rc<Comm>) -> Result<Self> {
            let attn = CausalSelfAttention::load(vb.pp("self_attn"), cache, cfg, comm.clone())?;
            let mlp = Mlp::load(vb.pp("mlp"), cfg, comm)?;
            let input_layernorm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("input_layernorm"))?;
            let post_attention_layernorm =
                rms_norm(cfg.hidden_size, 1e-5, vb.pp("post_attention_layernorm"))?;
            Ok(Self::new(
                input_layernorm,
                attn,
                post_attention_layernorm,
                mlp,
            ))
        }
    }

    pub struct Llama {
        wte: Embedding,
        blocks: Vec<Block>,
        ln_f: RmsNorm,
        lm_head: Linear,
    }

    impl Llama {
        fn new(wte: Embedding, blocks: Vec<Block>, ln_f: RmsNorm, lm_head: Linear) -> Self {
            Self {
                wte,
                blocks,
                ln_f,
                lm_head,
            }
        }

        pub fn forward(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
            let (_b_sz, seq_len) = x.shape().dims2()?;
            let mut x = self.wte.forward(x)?;
            for (block_idx, block) in self.blocks.iter().enumerate() {
                x = block.forward(&x, index_pos, block_idx)?;
            }
            let x = self.ln_f.forward(&x)?;
            let x = x.i((.., seq_len - 1, ..))?;
            let logits = self.lm_head.forward(&x)?;
            logits.to_dtype(DType::F32)
        }

        pub fn load(vb: VarBuilder, cache: &Cache, cfg: &Config, comm: Rc<Comm>) -> Result<Self> {
            let wte = embedding(cfg, vb.pp("model.embed_tokens"))?;
            let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
            let norm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("model.norm"))?;
            let blocks: Vec<_> = (0..cfg.num_hidden_layers)
                .map(|i| {
                    Block::load(
                        vb.pp(&format!("model.layers.{i}")),
                        cache,
                        cfg,
                        comm.clone(),
                    )
                    .unwrap()
                })
                .collect();

            Ok(Self::new(wte, blocks, norm, lm_head))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_model_interface() {
        let api_key = "".to_string();
        let cache_dir: PathBuf = "./test_llama_cache_dir/".into();
        let model_id = "llama_tiny_llama_1_1b_chat".to_string();
        let dtype = "f32".to_string();
        let revision = "main".to_string();
        let device_id = 0;
        let use_flash_attention = false;
        let config = ModelConfig::new(
            model_id,
            dtype.clone(),
            revision,
            device_id,
            use_flash_attention,
        );
        let load_data = LlamaModel::fetch(api_key, cache_dir.clone(), config)
            .expect("Failed to fetch llama model");

        println!("model device = {:?}", load_data.device);
        let should_be_device = device(device_id).unwrap();
        if should_be_device.is_cpu() {
            assert!(load_data.device.is_cpu());
        } else if should_be_device.is_cuda() {
            assert!(load_data.device.is_cuda());
        } else if should_be_device.is_metal() {
            assert!(load_data.device.is_metal());
        } else {
            panic!("Invalid device")
        }

        assert_eq!(load_data.file_paths.len(), 3);
        assert_eq!(load_data.use_flash_attention, use_flash_attention);
        assert_eq!(load_data.model_type, ModelType::LlamaTinyLlama1_1BChat);

        let should_be_dtype = DType::from_str(&dtype).unwrap();
        assert_eq!(load_data.dtype, should_be_dtype);
        let mut model = LlamaModel::load(load_data).expect("Failed to load model");

        if should_be_device.is_cpu() {
            assert!(model.device.is_cpu());
        } else if should_be_device.is_cuda() {
            assert!(model.device.is_cuda());
        } else if should_be_device.is_metal() {
            assert!(model.device.is_metal());
        } else {
            panic!("Invalid device")
        }

        assert_eq!(model.model_type, ModelType::LlamaTinyLlama1_1BChat);

        let prompt = "Write a hello world rust program: ".to_string();
        let temperature = 0.6;
        let random_seed = 42;
        let repeat_penalty = 1.0;
        let repeat_last_n = 20;
        let max_tokens = 128;
        let top_k = 10;
        let top_p = 0.6;

        let input = TextModelInput::new(
            prompt.clone(),
            temperature,
            random_seed,
            repeat_penalty,
            repeat_last_n,
            max_tokens,
            top_k,
            top_p,
        );
        let output = model.run(input).expect("Failed to run inference");
        println!("{output}");

        assert!(output.text.len() > 1);
        assert!(output.text.split(' ').collect::<Vec<_>>().len() <= max_tokens);

        std::fs::remove_dir_all(cache_dir).unwrap();
    }
}
