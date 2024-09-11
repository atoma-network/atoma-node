use thiserror::Error;
use tokenizers::Encoding;
use tokio::sync::{mpsc, oneshot};
use tracing::{info_span, instrument, Span};

use crate::{
    serving::tokenizer::{PreparedInput, TokenizerError, TokenizerRequest},
    serving::types::{GenerateParameters, GenerateRequest},
};

const DEFAULT_RANDOM_SEED: u64 = 1_283_768_955;

/// `Validator` - Responsible for `Request`/`Response` validation
#[derive(Clone, Debug)]
pub struct Validation {
    /// Validation of `best_of`
    best_of: usize,
    /// Validation of `max_stop_sequences`
    max_stop_sequences: usize,
    /// Validation of `max_top_n_tokens`
    max_top_n_tokens: u32,
    /// Validation of `max_input_length`
    max_input_length: usize,
    /// Validation of `max_total_tokens`
    max_total_tokens: u32,
    /// Tracing span
    span: Span,
    /// Channel to communicate with the background tokenizer task
    sender: mpsc::UnboundedSender<TokenizerRequest>,
}

impl Validation {
    /// Constructor
    pub fn new(
        best_of: usize,
        max_stop_sequences: usize,
        max_top_n_tokens: u32,
        max_input_length: usize,
        max_total_tokens: u32,
        sender: mpsc::UnboundedSender<TokenizerRequest>,
    ) -> Self {
        Self {
            best_of,
            max_stop_sequences,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            span: info_span!("validation"),
            sender,
        }
    }

    /// Tokenize inputs
    #[instrument(skip(input))]
    pub async fn tokenize(
        &self,
        input: String,
        truncate: Option<usize>,
    ) -> Result<PreparedInput, ValidationError> {
        // Response channel
        let (response_sender, response_receiver) = oneshot::channel();
        let request = TokenizerRequest {
            input,
            sender: response_sender,
            span: Span::current(),
        };
        // Send request to the background tokenization task
        self.sender.send(request).unwrap(); // DON'T PANIC: safe to unwrap here

        let response = response_receiver.await.unwrap();
        Ok(response?)
    }

    #[instrument(skip(self, input))]
    /// Validates the input of a received `Request`. Returns the input, the input token length and number
    /// of maximum new tokens to generate
    async fn validate_input(
        &self,
        input: String,
        truncate: Option<usize>,
        max_new_tokens: Option<u32>,
    ) -> Result<(String, Encoding, u32), ValidationError> {
        let PreparedInput { encoding, input } = self.tokenize(input.clone(), truncate).await?;
        let input_len = if let Some(truncate) = truncate {
            std::cmp::min(truncate, encoding.len())
        } else {
            encoding.len()
        };

        if input_len == 0 {
            // TODO: handle the case in which input length == 0
        }

        // Get total number of tokens
        // NOTE: we assume `input_len < 2^32`
        let max_new_tokens = if let Some(max_new_tokens) = max_new_tokens {
            max_new_tokens
        } else {
            self.max_total_tokens.saturating_sub(input_len as u32)
        };
        let total_tokens = input_len as u32 + max_new_tokens;

        // Validate `total_tokens`
        if total_tokens > self.max_total_tokens {
            return Err(ValidationError::MaxTotalTokens(
                self.max_total_tokens,
                input_len,
                max_new_tokens,
            ));
        }

        // Validate `input_len`
        if input_len > self.max_input_length {
            return Err(ValidationError::InputLength(
                self.max_input_length,
                input_len,
            ));
        }

        let histogram = metrics::histogram!("atoma-vllm_input_length");
        histogram.record(input_len as f64);

        Ok((input, encoding, max_new_tokens))
    }

    /// Validates a payload and gets the number of tokens in the input
    #[instrument(skip_all)]
    pub(crate) async fn validate(
        &self,
        request: GenerateRequest,
    ) -> Result<ValidGenerateRequest, ValidationError> {
        let GenerateParameters {
            best_of,
            temperature,
            repetition_penalty,
            frequency_penalty,
            top_k,
            top_p,
            typical_p,
            do_sample,
            max_new_tokens,
            stop: stop_sequences,
            truncate,
            random_seed,
            decoder_input_details,
            top_n_tokens,
            n,
            return_full_text,
            repeat_last_n,
        } = request.parameters;

        // sampling must be true when best_of > 1
        let best_of = best_of.unwrap_or(1);
        let sampling = do_sample
            || temperature.is_some()
            || top_k.is_some()
            || top_p.is_some()
            || typical_p.is_some();

        if best_of > 1 && !sampling {
            return Err(ValidationError::BestOfSampling);
        }

        let temperature = temperature.unwrap_or(1.0);
        if temperature <= 0.0 {
            return Err(ValidationError::Temperature);
        }

        let repetition_penalty = repetition_penalty.unwrap_or(1.0);
        if repetition_penalty <= 0.0 {
            return Err(ValidationError::RepetitionPenalty);
        }

        let frequency_penalty = frequency_penalty.unwrap_or(0.0);
        if !(-2.0..=2.0).contains(&frequency_penalty) {
            return Err(ValidationError::FrequencyPenalty);
        }

        let top_p = top_p
            .map(|value| {
                if value <= 0.0 || value >= 1.0 {
                    return Err(ValidationError::TopP);
                }
                Ok(value)
            })
            .unwrap_or(Ok(1.0))?;

        let typical_p = typical_p
            .map(|value| {
                if value <= 0.0 || value >= 1.0 {
                    return Err(ValidationError::TypicalP);
                }
                Ok(value)
            })
            .unwrap_or(Ok(1.0))?;

        let top_k = top_k
            .map(|value| {
                if value == 0 {
                    return Err(ValidationError::TopK);
                }
                Ok(value)
            })
            .unwrap_or(Ok(0))?;

        if max_new_tokens == Some(0) {
            return Err(ValidationError::NegativeMaxNewTokens);
        }

        if stop_sequences.len() > self.max_stop_sequences {
            return Err(ValidationError::StopSequence(
                self.max_stop_sequences,
                stop_sequences.len(),
            ));
        }

        // If seed is None, assign a default value
        // TODO: how secure is this for Atoma nodes ?
        let random_seed = match random_seed {
            // TODO: this approach might be unsecure for Atoma nodes
            None => DEFAULT_RANDOM_SEED,
            Some(seed) => {
                if best_of > 1 {
                    return Err(ValidationError::BestOfSampling);
                }
                seed
            }
        };

        let top_n_tokens = top_n_tokens
            .map(|value| {
                if value > self.max_top_n_tokens {
                    return Err(ValidationError::TopNTokens(self.max_top_n_tokens, value));
                }
                Ok(value)
            })
            .unwrap_or(Ok(0))?;

        let repeat_last_n = repeat_last_n.unwrap_or(0);

        // Check if inputs is empty
        if request.inputs.is_empty() {
            return Err(ValidationError::EmptyInput);
        }

        // Check if truncate is strictly positive and less than max_input_length
        let truncate = truncate
            .map(|value| {
                if value == 0 || value > self.max_input_length {
                    return Err(ValidationError::Truncate(self.max_input_length, value));
                }
                Ok(Some(value))
            })
            .unwrap_or(Ok(None))?;

        // Validate inputs
        let (inputs, encoding, max_new_tokens) = self
            .validate_input(request.inputs, truncate, max_new_tokens)
            .await?;

        let parameters = NextTokenChooserParameters {
            temperature,
            repetition_penalty,
            best_of,
            frequency_penalty,
            top_k,
            top_p,
            typical_p,
            do_sample,
            random_seed,
            repeat_last_n,
            n,
        };
        let stopping_parameters = StoppingCriteriaParameters {
            max_new_tokens,
            stop_sequences,
            ignore_eos_token: false,
        };

        let histogram = metrics::histogram!("tgi_request_max_new_tokens");
        histogram.record(max_new_tokens as f64);

        let input_token_len = encoding.len();
        Ok(ValidGenerateRequest {
            request_id: request.request_id,
            inputs,
            decoder_input_details,
            encoding,
            input_token_len,
            truncate: truncate.unwrap_or(self.max_input_length) as u32,
            parameters,
            stopping_parameters,
            top_n_tokens,
            return_full_text: return_full_text.unwrap_or(false),
        })
    }
}

#[derive(Clone, Debug)]
/// `ValidGenerateRequest` - Obtained from a
/// `GenerateRequest`, after input validation has
/// taken place
pub(crate) struct ValidGenerateRequest {
    /// The request id
    pub request_id: String,
    /// Inputs, in the form of a `String`
    pub inputs: String,
    /// Input tokenizer encoding
    pub encoding: Encoding,
    /// Inputs token length
    pub input_token_len: usize,
    /// The truncation window of the input
    pub truncate: u32,
    /// Whether to return decoder input token logprobs and ids.
    pub decoder_input_details: bool,
    /// Set of parameters necessary for choosing the next token, after a
    /// LLM forward pass
    pub parameters: NextTokenChooserParameters,
    /// Stopping criteria parameters
    pub stopping_parameters: StoppingCriteriaParameters,
    /// Top `n` tokens
    pub top_n_tokens: u32,
    /// Whether to prepend the prompt to the generated text
    pub return_full_text: bool,
}

/// `NextTokenChooseParameters` - Set of parameters which
/// are necessary for choosing the next token, after a single
/// forward pass of a LLM
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NextTokenChooserParameters {
    /// Top n sequences
    pub n: usize,
    /// best of sequences
    pub best_of: usize,
    /// exponential scaling output probability distribution
    pub temperature: f32,
    /// restricting to the k highest probability elements
    pub top_k: u32,
    /// restricting to top tokens summing to prob_cut_off <= prob_cut_off
    pub top_p: f32,
    /// restricting to top tokens summing to prob_cut_off <= prob_cut_off
    pub typical_p: f32,
    /// apply sampling on the logits
    pub do_sample: bool,
    /// random seed for sampling
    pub random_seed: u64,
    /// repetition penalty
    pub repetition_penalty: f32,
    /// repeat last n tokens
    pub repeat_last_n: u32,
    /// frequency penalty
    pub frequency_penalty: f32,
}

/// `StoppingCriteriaParameters` - Criteria for stopping
/// LLM next token generation
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StoppingCriteriaParameters {
    /// Maximum number of generated tokens
    pub max_new_tokens: u32,
    /// Optional stopping sequences
    pub stop_sequences: Vec<String>,
    /// Ignore end of sequence token
    /// used for benchmarking
    pub ignore_eos_token: bool,
}

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Tokenizer error: `{0}`")]
    TokenizerError(#[from] TokenizerError),
    #[error("Maximum total number error: max_total_tokens = `{0}`, input_len = `{1}`, max_new_tokens = `{2}`")]
    MaxTotalTokens(u32, usize, u32),
    #[error("Input length error: max_input_len = `{0}`, input_len = `{1}`")]
    InputLength(usize, usize),
    #[error("Invalid best of sampling parameter")]
    BestOfSampling,
    #[error("Invalid temperature parameter")]
    Temperature,
    #[error("Invalid repetition parameter")]
    RepetitionPenalty,
    #[error("Invalid frequency penalty parameter")]
    FrequencyPenalty,
    #[error("Invalid top p parameter")]
    TopP,
    #[error("Invalid typical p parameter")]
    TypicalP,
    #[error("Invalid top k parameter")]
    TopK,
    #[error("Negative max new tokens to generate")]
    NegativeMaxNewTokens,
    #[error("Stop sequences size exceeds maximum number of stop sequences allowed: `{0}` < `{1}`")]
    StopSequence(usize, usize),
    #[error("Empty random seed")]
    NullRandomSeed,
    #[error("Invalid top n tokens parameter: `{0}`, `{1}`")]
    TopNTokens(u32, u32),
    #[error("Empty input")]
    EmptyInput,
    #[error("Invalid truncate paremeter: `{0}` < `{1}`")]
    Truncate(usize, usize),
}
