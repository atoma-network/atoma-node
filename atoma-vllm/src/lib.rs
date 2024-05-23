//! This crate contains logic for fast inference servince, based on PagedAttention and the vLLM implementation. We refer the reader to
//! https://arxiv.org/pdf/2309.06180 for the detailed architecture of the service. We were highly inspired by the complete, in production, Python implementation
//! of vLLM, in https://github.com/vllm-project/vllm.

pub mod block;
pub mod block_allocator;
pub mod evictor;