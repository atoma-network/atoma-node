use std::{collections::HashMap, path::PathBuf, sync::Arc};

use async_trait::async_trait;
use candle::{IndexOp, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use futures::{io::repeat, stream::FuturesUnordered};
use thiserror::Error;
use tokenizers::decoders::sequence;
use tokio::{
    sync::{
        mpsc,
        oneshot::{self, error::RecvError},
    },
    task::JoinHandle,
};
use tracing::{error, info, instrument, trace};

use crate::{
    sequence::{
        ExecuteModelRequest, LogProb, SequenceGroupMetadata, SequenceGroupOutput, SequenceOutput,
    },
    validation::{NextTokenChooserParameters, StoppingCriteriaParameters},
    worker::{ModelWorker, ModelWorkerError},
};

/// `ModelLoader` trait - interface for fetching
/// and loading a LLM model weights. Also has a method
/// providing the `eos_token_id` for the current model's
/// tokenizer.
pub trait ModelLoader {
    type FilePaths;

    fn fetch(
        api_key: String,
        model_name: String,
        revision: String,
    ) -> Result<Self::FilePaths, ModelLoaderError>;
    fn load(file_paths: Self::FilePaths) -> Result<Self, ModelLoaderError>
    where
        Self: Sized;
    fn cache_dir(&self) -> PathBuf;
    fn eos_token_id(&self) -> Option<u32>;
    fn head_size(&self) -> usize;
    fn num_attention_heads(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn sliding_window(&self) -> Option<usize>;
}

/// `ModelExecutor` trait - interface for running AI inference
/// from a LLM
pub trait ModelExecutor: ModelLoader {
    type AttentionMetadata;
    type Output: Into<u32>;

    fn forward(
        &mut self,
        input_tensor: &Tensor,
        input_positions: &Tensor,
        selected_token_positions: &Tensor,
        kv_cache: Vec<&mut Tensor>,
        attention_metadata: Self::AttentionMetadata,
    ) -> Result<Tensor, ModelExecutorError>;
    fn sample(
        &mut self,
        logits: &Tensor,
        sequence_groups_metadata: &Vec<Arc<SequenceGroupMetadata>>,
    ) -> Result<Self::Output, ModelExecutorError> {
        let total_num_sequences = sequence_groups_metadata
            .iter()
            .map(|metadata| metadata.sequence_data.keys().len())
            .sum::<usize>();

        // 1. Check if the logits zeroth dimension matches the total number of sequences
        if logits.dims()[0] != total_num_sequences {
            return Err(ModelExecutorError::InvalidLogits(
                logits.dims()[0],
                total_num_sequences,
            ));
        }

        let mut sequence_group_outputs = Vec::with_capacity(sequence_groups_metadata.len());
        let mut logits_idx = 0;
        for sequence_group_metadata in sequence_groups_metadata.iter() {
            // 2. Retrieve the next token chooser and stopping criteria parameters,
            //    from the `SequenceGroupMetadata`, to be used for sampling
            let NextTokenChooserParameters {
                n,
                best_of,
                temperature,
                repetition_penalty,
                repeat_last_n,
                frequency_penalty,
                top_k,
                top_p,
                typical_p,
                do_sample,
                random_seed,
            } = sequence_group_metadata.next_token_chooser_params;
            let StoppingCriteriaParameters {
                max_new_tokens,
                stop_sequences,
                ignore_eos_token,
            } = sequence_group_metadata.stopping_criteria_params;

            let sampling = if !do_sample || temperature == 1.0 {
                Sampling::ArgMax
            } else if top_p == 1.0 && top_k == 0 {
                Sampling::All {
                    temperature: temperature as f64,
                }
            } else if top_k == 0 && top_p < 1.0 {
                Sampling::TopP {
                    p: top_p,
                    temperature: temperature as f64,
                }
            } else if top_k != 0 && top_p == 1.0 {
                Sampling::TopK {
                    k: top_k as usize,
                    temperature: temperature as f64,
                }
            } else {
                Sampling::TopKThenTopP {
                    k: top_k as usize,
                    p: top_p as f64,
                    temperature: temperature as f64,
                }
            };

            // 3. Create a `LogitsProcessor` instance, with the sampling strategy
            let mut logits_processor = LogitsProcessor::from_sampling(random_seed, sampling);

            // 4. Allocate a `HashMap` to store each of the sequence group's outputs
            let mut sequence_outputs =
                HashMap::with_capacity(sequence_group_metadata.sequence_data.len());

            // 4. Iterate over each `SequenceData` in the `SequenceGroupMetadata`,
            //    to sample next tokens for each sequence
            for (sequence_id, sequence_data) in sequence_group_metadata.sequence_data.iter() {
                // 5. Select the given sequence logits, and apply a
                // repetition penalty if necessary
                let sequence_logits = if repetition_penalty == 1. {
                    logits.i(logits_idx)?
                } else {
                    debug_assert!(repeat_last_n > 0, "repeat_last_n should be > 0");
                    let num_sequence_tokens = sequence_data.length();
                    let start_at = num_sequence_tokens
                        .checked_sub(repeat_last_n)
                        .unwrap_or_default();
                    let context = sequence_data.get_token_ids();
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits.i(logits_idx)?,
                        repetition_penalty,
                        &context[start_at..],
                    )?
                };

                // 6. Sample the next token
                // TODO: we should be able to sample `best_of` sequences
                //      simultaneously, so we can later generate multiple
                //      sequences at once, in parallel.
                let next_token = logits_processor.sample(&sequence_logits)?;

                // 7. Update the logits index
                logits_idx += 1;

                // 8. Update the `output`
                // TODO: we are not forking a parent sequence into a new
                //       sequence group, so we should not have to update
                sequence_outputs.insert(
                    *sequence_id,
                    SequenceOutput {
                        parent_sequence_id: *sequence_id,
                        output_token: next_token,
                        logprob: HashMap::from_iter([(
                            next_token,
                            LogProb::new(0.8, Some(1), Some(next_token)),
                        )]),
                    },
                );
            }
        }
    }
}

/// `ModelThreadCommand` - encapsulates a `ValidGenerateRequest`
/// to run AI inference on, together with a oneshot::Sender
/// channel to communicate the AI generated output with the
/// main task
pub struct ModelThreadCommand {
    request: ExecuteModelRequest,
    sender: oneshot::Sender<Vec<SequenceGroupOutput>>,
}

/// `ModelThread` - encapsulates the logic
/// to run a model thread/task in the background.
/// It receives new coming requests and start processing
/// AI inference on it.
pub struct ModelThread<M: ModelExecutor> {
    worker: ModelWorker<M>,
    receiver: mpsc::UnboundedReceiver<ModelThreadCommand>,
}

impl<M> ModelThread<M>
where
    M: ModelExecutor + Send + Sync,
{
    /// Main loop, it listenings to incoming requests, in the form `ModelThreadCommand`.
    /// When a new request is received, it starts a new inference loop for the encapsulated
    /// AI model `M`. Once the AI generated output is ready, it sends it back using the corresponding
    /// `oneshot` `Sender` encapsulated in the `ModelThreadCommand`.
    #[instrument(skip(self))]
    pub fn run(mut self) -> Result<(), ModelThreadError> {
        info!("Start Model thread");

        while let Some(command) = self.receiver.blocking_recv() {
            let ModelThreadCommand { request, sender } = command;

            let sequence_groups_metadata = request.sequence_groups_metadata.clone();
            let next_token_chooser_params: Vec<NextTokenChooserParameters> = request
                .sequence_groups_metadata
                .iter()
                .map(|s| s.next_token_chooser_params.clone())
                .collect();
            let stopping_params: Vec<StoppingCriteriaParameters> = request
                .sequence_groups_metadata
                .iter()
                .map(|s| s.stopping_criteria_params.clone())
                .collect();

            let output = match self.worker.execute_model(request) {
                Ok(output) => hidden_states,
                Err(e) => {
                    error!("Failed to run forward pass on model, with error: {e}");
                    return Err(ModelThreadError::ModelExecutorError(e));
                }
            };

            let mut responses = Vec::with_capacity(next_token_chooser_params.len());

            // // TODO: should we parallelize this loop, with rayon or within the async runtime ?
            // for (next_token_params, (stopping_params, metadata)) in next_token_chooser_params
            //     .iter()
            //     .zip(stopping_params.iter().zip(sequence_groups_metadata))
            // {
            //     let mut outputs = HashMap::with_capacity(metadata.sequence_data.len());

            //     for sequence_id in metadata.sequence_data.keys() {
            //         let decode_token = match self.model.sample(
            //             logits.clone(),
            //             next_token_params.clone(),
            //             stopping_params.clone(),
            //         ) {
            //             Ok(token) => token,
            //             Err(e) => {
            //                 error!("Failed to sample next decoding token, with error: {e}");
            //                 return Err(ModelThreadError::ModelExecutorError(e));
            //             }
            //         };

            //         let output_token = decode_token.into();

            //         outputs.insert(
            //             *sequence_id,
            //             SequenceOutput {
            //                 parent_sequence_id: *sequence_id,
            //                 output_token,
            //                 logprob: HashMap::from_iter([(
            //                     output_token,
            //                     LogProb::new(0.8, None, None),
            //                 )]), // TODO: replace hardcoded values with logic
            //             },
            //         );
            //     }

            //     // TODO: Check this is the correct logic, once we integrate
            //     // with model executor
            //     let response = SequenceGroupOutput {
            //         outputs,
            //         sampled_token_ids: None,
            //         sampled_token_probs: None,
            //         logprobs: None,
            //         spec_decode_worker_metrics: None,
            //     };
            //     responses.push(response);
            // }

            sender.send(responses).ok();
        }

        Ok(())
    }
}

/// `ModelThreadDispatcher` - Responsible for managing incoming requests to
/// different the background LLM inference task
pub struct ModelThreadDispatcher {
    /// Mapping from each model id to the remove `Sender`'s `ModelThreadCommand`
    pub sender: mpsc::UnboundedSender<ModelThreadCommand>,
    /// A `FuturesUnordered` containing each generated `Response`'s oneshot receiver.
    /// It should yield everytime a new AI inference output is generated.
    pub responses: FuturesUnordered<oneshot::Receiver<Vec<SequenceGroupOutput>>>,
    /// The model's thread join handle
    pub join_handle: JoinHandle<Result<(), ModelThreadError>>,
}

impl ModelThreadDispatcher {
    /// Starts a new instance of a `ModelThreadDispatcher`. It further spawns a new thread model
    /// that continuously listens to incoming AI inference requests, and processes these.
    #[instrument(skip_all)]
    pub(crate) fn start<M>(
        api_key: String,
        cache_config: CacheConfig,
        device: Device,
        dtype: DType,
        model_name: String,
        revision: String,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, ModelThreadError>
    where
        M: ModelExecutor + Send + Sync + 'static,
    {
        let (sender, receiver) = mpsc::unbounded_channel();

        let join_handle = tokio::task::spawn_blocking(|| {
            let model_worker = ModelWorker::new(
                api_key,
                cache_config,
                device,
                dtype,
                model_name,
                revision,
                scheduler_config,
            )?;
            let model_thread = ModelThread {
                worker: model_worker,
                receiver,
            };
            if let Err(e) = model_thread.run() {
                error!("Model thread error: {e}");
                if !matches!(e, ModelThreadError::Shutdown(_)) {
                    panic!("Fatal error occurred: {e}");
                }
            }

            Ok(())
        });

        let model_dispatcher = ModelThreadDispatcher {
            sender,
            responses: FuturesUnordered::new(),
            join_handle,
        };

        Ok(model_dispatcher)
    }

    /// Sends a `ModelThreadCommand` instance into the corresponding
    /// `Model`'s thread, to be processed by the `Model` itself.
    #[instrument(skip(self))]
    pub fn send(&self, request: ExecuteModelRequest) {
        trace!("Sending new `ExecuteModelRequest` to model executor task");

        let (sender, receiver) = oneshot::channel();
        let command = ModelThreadCommand { request, sender };

        if let Err(e) = self.sender.send(command) {
            error!("Could not send command to model core, it might be shutting down: {e}");
        }

        self.responses.push(receiver);
    }
}

#[derive(Debug, Error)]
pub enum ModelThreadError {
    #[error("Core thread shutdown: `{0}`")]
    Shutdown(RecvError),
    #[error("Send error")]
    SendError,
    #[error("Model loader error: `{0}`")]
    ModelLoaderError(#[from] ModelLoaderError),
    #[error("Model executor error: `{0}`")]
    ModelExecutorError(#[from] ModelExecutorError),
    #[error("Model worker error: `{0}`")]
    ModelWorkerError(#[from] ModelWorkerError),
}

#[derive(Debug, Error)]
pub enum ModelLoaderError {}

#[derive(Debug, Error)]
pub enum ModelExecutorError {
    #[error("Candle error: `{0}`")]
    CandleError(#[from] candle::Error),
    #[error(
        "Invalid logits or next token parameters (logits dims: {0}, next token params dims: {1})"
    )]
    InvalidLogits(usize, usize),
    #[error("Invalid next token parameters or stopping parameters (next token params dims: {0}, stopping params dims: {1})")]
    InvalidNextTokenParams(usize, usize),
}
