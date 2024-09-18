use thiserror::Error;
use tokenizers::{tokenizer::Tokenizer, Encoding, Error};
use tokio::sync::{
    mpsc::{self, error::SendError},
    oneshot,
};
use tracing::{error, info, info_span, instrument, Span};

/// `EncodeTokenizerRequest` - A request for encoding a string input
/// into a suite of tokens (expressed as a `u32` vector)
pub struct EncodeTokenizerRequest {
    /// Input string
    pub input: String,
    /// Truncate the input to the given length
    pub truncate: Option<usize>,
    /// `oneshot::Sender`` responsible to deliver the result of tokenization,
    /// which includes the actual `Encoding`, together with the original input
    /// in `String` format
    pub sender: oneshot::Sender<Result<(Encoding, String), TokenizerError>>,
    /// The current tracing span
    pub span: Span,
}

/// `DecodeTokenizerRequest` - A request to decode a given token id into an actual text
pub struct DecodeTokenizerRequest {
    /// The token id to be decoded
    pub token_id: u32,
    /// `oneshot::Sender` responsible to deliver the result of detokenization
    pub sender: oneshot::Sender<Result<String, TokenizerError>>,
    /// The current tracing span
    pub span: Span,
}

/// `Tokenizer` - a tokenizer worker
/// responsible for prepare input requests
pub struct TokenizerWorker;

impl TokenizerWorker {
    /// Starts the tokenizer workers
    pub async fn start(
        tokenizer: Tokenizer,
        receiver: mpsc::UnboundedReceiver<EncodeTokenizerRequest>,
        workers: usize,
    ) -> Result<(), TokenizerError> {

        let mut senders = Vec::with_capacity(workers);

        for i in 0..workers {
            let tokenizer_clone = tokenizer.clone();
            let (sender, worker_receiver) = mpsc::unbounded_channel();
            senders.push(sender);

            // Spawning the worker
            let span = info_span!("tokenizer-worker");
            tokio::task::spawn_blocking(move || {
                let _span = span.clone();
                let _enter = _span.enter();
                info!("Starting {i}-th tokenizer task");
                start_tokenizer_task(tokenizer_clone, worker_receiver, span)?;
                Ok::<_, TokenizerError>(())
            });
        }

        // Create tokenization round robin task
        tokio::spawn(round_robin_task(receiver, senders));

        Ok(())
    }
}

/// Starts a new tokenizer tokio task
#[instrument(skip_all)]
fn start_tokenizer_task(
    tokenizer: Tokenizer,
    mut receiver: mpsc::UnboundedReceiver<EncodeTokenizerRequest>,
    span: Span,
) -> Result<(), TokenizerError> {
    let _enter = span.enter();
    info!("Starting tokenizer task..");

    // Loops over requests
    while let Some(request) = receiver.blocking_recv() {
        info!("Received new `EncodeTokenizerRequest`");
        let EncodeTokenizerRequest {
            input,
            truncate,
            sender,
            span,
        } = request;
        span.in_scope(|| {
            let prepared_inputs = prepare_inputs(&tokenizer, input, truncate);
            sender.send(prepared_inputs).unwrap_or(())
        });
    }
    Ok(())
}

/// A round robin algo for tokenization tasks.
/// Check https://en.wikipedia.org/wiki/Round-robin_scheduling
/// for more details.
async fn round_robin_task(
    mut receiver: mpsc::UnboundedReceiver<EncodeTokenizerRequest>,
    senders: Vec<mpsc::UnboundedSender<EncodeTokenizerRequest>>,
) -> Result<(), TokenizerError> {
    loop {
        for sender in &senders {
            match receiver.recv().await {
                None => {
                    error!("Received None from the tokenizer receiver");
                    return Ok(());
                }
                Some(request) => {
                    info!("Received a new request from the tokenizer receiver");
                    sender.send(request)?;
                }
            }
        }
    }
}

fn prepare_inputs(
    tokenizer: &Tokenizer,
    input: String,
    truncate: Option<usize>,
) -> Result<(Encoding, String), TokenizerError> {
    let input = if let Some(truncate) = truncate {
        if truncate > input.chars().count() {
            input
        } else {
            let start = input.chars_indices().nth(input.chars().count() - truncate).map(|(idx, _)| idx).unwrap_or(0);
            input[start..].to_string()
        }
    } else { 
        input
    };
    let encoding = tokenizer.encode(input.clone(), true)?;
    Ok((encoding, input))
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("Oneshot sender error: `{0}`")]
    OneshotSenderError(String),
    #[error("Tokenizer error: `{0}`")]
    Tokenizer(#[from] Error),
    #[error("Send error: `{0}`")]
    SendError(#[from] SendError<EncodeTokenizerRequest>),
}
