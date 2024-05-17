mod model;

use std::{path::PathBuf, rc::Rc};

use atoma_types::{TextModelInput, TextModelOutput};
use candle::{DType, Device, Tensor};
use candle_transformers::{generation::LogitsProcessor, models::llama::LlamaConfig};
use cudarc::nccl::Comm;
use tokenizers::Tokenizer;

use crate::{token_output_stream::TokenOutputStream, ModelError};

const MAX_SEQ_LEN: usize = 4096;

pub struct Model {
    config: LlamaConfig,
    tokenizer: Tokenizer,
    model: model::Llama,
    device: Device,
}

impl Model {
    // fn load_model(dtype: DType, model_type: ModelType) -> Result<Vec<PathBuf>, ModelError> {
    //     let api = Api::new()?;
    //     let model_id = model_type.repo();
    //     println!("loading the model weights from {model_id}");
    //     let revision = model_type.default_revision();
    //     let api = api.repo(Repo::with_revision(
    //         model_id.to_string(),
    //         RepoType::Model,
    //         revision.to_string(),
    //     ));
    //     let config_filename = api.get("config.json")?;
    //     let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    //     let tokenizer_filename = api.get("tokenizer.json")?;
    //     hub_load_safetensors(&api, "model.safetensors.index.json")
    // }
    pub fn load(
        config_filename: PathBuf,
        device: Device,
        dtype: DType,
        filenames: Vec<PathBuf>,
        tokenizer_filename: PathBuf,
        comm: Rc<Comm>,
    ) -> Result<Self, ModelError> {
        let config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let cache = model::Cache::new(dtype, &config, &device)?;
        let vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(&filenames, dtype, &device)?
        };
        let model = model::Llama::load(vb, &cache, &config, comm).unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

        Ok(Self {
            config,
            tokenizer,
            model,
            device,
        })
    }

    pub fn inference(&self, input: TextModelInput) -> Result<TextModelOutput, ModelError> {
        let temperature = if input.temperature <= 0. {
            None
        } else {
            Some(input.temperature)
        };

        let prompt = input.prompt;
        let mut tokens = self.tokenizer.encode(prompt, true)?.get_ids().to_vec();

        let mut tokenizer: TokenOutputStream = TokenOutputStream::new(self.tokenizer.clone());
        let mut logits_processor =
            LogitsProcessor::new(input.random_seed, temperature, input.top_p);
        let mut new_tokens = vec![];
        let mut start_gen = std::time::Instant::now();
        let mut index_pos = 0;
        let mut res = String::new();

        for index in 0..input.max_tokens {
            // Only start timing at the second token as processing the first token waits for all the
            // weights to be loaded in an async way.
            if index == 1 {
                start_gen = std::time::Instant::now()
            };
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, index_pos)?;
            let logits = logits.squeeze(0)?;
            index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            new_tokens.push(next_token);
            if Some(next_token) == self.config.eos_token_id {
                break;
            }
            if let Some(t) = tokenizer.next_token(next_token)? {
                res += &t;
            }
        }
        let dt = start_gen.elapsed();
        Ok(TextModelOutput {
            text: res,
            time: dt.as_secs_f64(),
            tokens_count: tokenizer.get_num_generated_tokens(),
        })
    }
}
