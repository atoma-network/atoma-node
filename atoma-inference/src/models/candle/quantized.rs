use std::{path::PathBuf, str::FromStr, sync::mpsc};

use candle::{
    quantized::{ggml_file, gguf_file},
    DType, Device, Tensor,
};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_llama::ModelWeights,
};
use hf_hub::api::sync::ApiBuilder;
use tokenizers::Tokenizer;
use tracing::info;

use crate::models::{
    candle::device,
    config::ModelConfig,
    token_output_stream::TokenOutputStream,
    types::{LlmLoadData, ModelType, TextModelInput, TextModelOutput},
    ModelError, ModelTrait,
};
use candle_transformers::models::quantized_llama as model;

pub struct QuantizedModel {
    model: ModelWeights,
    model_type: ModelType,
    device: Device,
    tokenizer: TokenOutputStream,
}

impl QuantizedModel {
    pub fn new(
        model: ModelWeights,
        device: Device,
        model_type: ModelType,
        tokenizer: Tokenizer,
        stream_tx: mpsc::Sender<String>,
    ) -> Self {
        Self {
            model,
            model_type,
            device,
            tokenizer: TokenOutputStream::new(tokenizer, stream_tx),
        }
    }
}

impl ModelTrait for QuantizedModel {
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
        let repo = api.model(repo_id);
        let filename = match model_type {
            ModelType::QuantizedLlamaV2_7b => "llama-2-7b.ggmlv3.q4_0.bin",
            ModelType::QuantizedLlamaV2_13b => "llama-2-13b.ggmlv3.q4_0.bin",
            ModelType::QuantizedLlamaV2_70b => "llama-2-70b.ggmlv3.q4_0.bin",
            ModelType::QuantizedLlamaV2_7bChat => "llama-2-7b-chat.ggmlv3.q4_0.bin",
            ModelType::QuantizedLlamaV2_13bChat => "llama-2-13b-chat.ggmlv3.q4_0.bin",
            ModelType::QuantizedLlamaV2_70bChat => "llama-2-70b-chat.ggmlv3.q4_0.bin",
            ModelType::QuantizedLlama7b => "codellama-7b.Q8_0.gguf",
            ModelType::QuantizedLlama13b => "codellama-13b.Q8_0.gguf",
            ModelType::QuantizedLlama34b => "codellama-34b.Q8_0.gguf",
            ModelType::QuantizedLeo7b => "leo-hessianai-7b.Q4_K_M.gguf",
            ModelType::QuantizedLeo13b => "leo-hessianai-13b.Q4_K_M.gguf",
            ModelType::QuantizedMixtral => "mixtral-8x7b-v0.1.Q4_K_M.gguf",
            ModelType::QuantizedMixtralInstruct => "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
            ModelType::QuantizedMistral7b => "mistral-7b-v0.1.Q4_K_S.gguf",
            ModelType::QuantizedMistral7bInstruct => "mistral-7b-instruct-v0.1.Q4_K_S.gguf",
            ModelType::QuantizedMistral7bInstructV02 => "mistral-7b-instruct-v0.2.Q4_K_S.gguf",
            ModelType::QuantizedZephyr7bAlpha => "zephyr-7b-alpha.Q4_K_M.gguf",
            ModelType::QuantizedZephyr7bBeta => "zephyr-7b-beta.Q4_K_M.gguf",
            ModelType::QuantizedOpenChat35 => "openchat_3.5.Q4_K_M.gguf",
            ModelType::QuantizedStarling7bAlpha => "starling-lm-7b-alpha.Q4_K_M.gguf",
            // TODO: swap to TheBloke model when available
            ModelType::QuantizedL8b => "Meta-Llama-3-8B.Q4_K_S.gguf",
            _ => unreachable!("Model not supported"),
        };
        let model_path = repo.get(filename)?;

        let repo_id = match model_type {
            ModelType::QuantizedLlamaV2_7b
            | ModelType::QuantizedLlamaV2_13b
            | ModelType::QuantizedLlamaV2_70b
            | ModelType::QuantizedLlamaV2_7bChat
            | ModelType::QuantizedLlamaV2_13bChat
            | ModelType::QuantizedLlamaV2_70bChat
            | ModelType::QuantizedLlama7b
            | ModelType::QuantizedLlama13b
            | ModelType::QuantizedLlama34b => "hf-internal-testing/llama-tokenizer",
            ModelType::QuantizedLeo7b => "LeoLM/leo-hessianai-7b",
            ModelType::QuantizedLeo13b => "LeoLM/leo-hessianai-13b",
            ModelType::QuantizedMixtral => "mistralai/Mixtral-8x7B-v0.1",
            ModelType::QuantizedMixtralInstruct => "mistralai/Mixtral-8x7B-Instruct-v0.1",
            ModelType::QuantizedMistral7b
            | ModelType::QuantizedMistral7bInstruct
            | ModelType::QuantizedMistral7bInstructV02
            | ModelType::QuantizedZephyr7bAlpha
            | ModelType::QuantizedZephyr7bBeta => "mistralai/Mistral-7B-v0.1",
            ModelType::QuantizedOpenChat35 => "openchat/openchat_3.5",
            ModelType::QuantizedStarling7bAlpha => "berkeley-nest/Starling-LM-7B-alpha",
            ModelType::QuantizedL8b => "meta-llama/Meta-Llama-3-8B",
            _ => unreachable!("Model not supported"),
        };

        let repo = api.model(repo_id.to_string());
        let tokenizer_path = repo.get("tokenizer.json")?;
        Ok(Self::LoadData {
            device,
            dtype,
            file_paths: vec![model_path, tokenizer_path],
            model_type: ModelType::from_str(&config.model_id())?,
            use_flash_attention: config.use_flash_attention(),
        })
    }

    fn load(load_data: Self::LoadData, stream_tx: mpsc::Sender<String>) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        let model_path = load_data.file_paths[0].clone();
        let mut file = std::fs::File::open(&model_path)?;
        let model = match model_path.extension().and_then(|v| v.to_str()) {
            Some("gguf") => {
                let model =
                    gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
                ModelWeights::from_gguf(model, &mut file, &load_data.device)?
            }
            Some("ggml" | "bin") | Some(_) | None => {
                let model = ggml_file::Content::read(&mut file, &load_data.device)
                    .map_err(|e| e.with_path(model_path))?;
                let default_gqa = match load_data.model_type {
                    ModelType::QuantizedLlamaV2_7b
                    | ModelType::QuantizedLlamaV2_13b
                    | ModelType::QuantizedLlamaV2_7bChat
                    | ModelType::QuantizedLlamaV2_13bChat
                    | ModelType::QuantizedLlama7b
                    | ModelType::QuantizedLlama13b
                    | ModelType::QuantizedLlama34b
                    | ModelType::QuantizedLeo7b
                    | ModelType::QuantizedLeo13b
                    | ModelType::QuantizedL8b => 1,
                    ModelType::QuantizedMixtral
                    | ModelType::QuantizedMixtralInstruct
                    | ModelType::QuantizedMistral7b
                    | ModelType::QuantizedMistral7bInstruct
                    | ModelType::QuantizedMistral7bInstructV02
                    | ModelType::QuantizedZephyr7bAlpha
                    | ModelType::QuantizedZephyr7bBeta
                    | ModelType::QuantizedLlamaV2_70b
                    | ModelType::QuantizedLlamaV2_70bChat
                    | ModelType::QuantizedOpenChat35
                    | ModelType::QuantizedStarling7bAlpha => 8,
                    _ => unreachable!("Model not supported"),
                };
                ModelWeights::from_ggml(model, default_gqa)?
            }
        };
        let tokenizer = Tokenizer::from_file(&load_data.file_paths[1])?;
        Ok(Self::new(
            model,
            load_data.device,
            load_data.model_type,
            tokenizer,
            stream_tx,
        ))
    }

    fn model_type(&self) -> ModelType {
        self.model_type.clone()
    }

    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let prompt_str = input.prompt;
        let mut output = String::new();
        self.tokenizer.clear();
        let tokens = self.tokenizer.tokenizer().encode(prompt_str, true)?;
        // if args.verbose_prompt {
        //     for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
        //         let token = token.replace('▁', " ").replace("<0x0A>", "\n");
        //         println!("{id:7} -> '{token}'");
        //     }
        // }

        let prompt_tokens = tokens.get_ids().to_vec();
        let to_sample = input.max_tokens.saturating_sub(1);
        let prompt_tokens = if prompt_tokens.len() + to_sample > model::MAX_SEQ_LEN - 10 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - model::MAX_SEQ_LEN;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens
        };
        let mut all_tokens = vec![];
        let mut logits_processor = {
            let temperature = input.temperature;
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (input.top_k, input.top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(input.random_seed, sampling)
        };

        let input3 = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input3, 0)?;
        let logits = logits.squeeze(0)?;
        let mut next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(t) = self.tokenizer.next_token(next_token, input.stream)? {
            output.push_str(t.as_str());
        }

        let eos_token = match self.model_type {
            ModelType::QuantizedL8b => "<|end_of_text|>",
            _ => "</s>",
        };

        let eos_token = *self
            .tokenizer
            .tokenizer()
            .get_vocab(true)
            .get(eos_token)
            .unwrap();
        let start_post_prompt = std::time::Instant::now();
        for index in 0..to_sample {
            let input2 = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input2, prompt_tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            let logits = if input.repeat_penalty == 1. {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(input.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    input.repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if let Some(t) = self.tokenizer.next_token(next_token, input.stream)? {
                output.push_str(t.as_str());
            }
            if next_token == eos_token {
                break;
            };
        }
        if let Some(rest) = self
            .tokenizer
            .decode_rest(input.stream)
            .map_err(candle::Error::msg)?
        {
            output.push_str(rest.as_str());
        }
        let dt = start_post_prompt.elapsed();
        let generated_tokens = self.tokenizer.get_num_generated_tokens();
        info!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(TextModelOutput {
            text: output,
            time: dt.as_secs_f64(),
            tokens_count: generated_tokens,
        })
    }
}
