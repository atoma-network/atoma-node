#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle_transformers::models::{clip, flux, t5};
use candle::{IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use tracing::info;
use tokenizers::Tokenizer;
