use thiserror::Error;
use tokenizers::{tokenizer::Tokenizer, Encoding, Error};
use tokio::sync::{
    mpsc::{self, error::SendError},
    oneshot,
};
use tracing::{error, Span};

/// `TokenizerRequest` - A request for the tokenizer worker tokenize
/// the string input
pub struct TokenizerRequest {
    /// Input string
    pub input: String,
    /// `oneshot::Sender` responsible to deliver the result of tokenization,
    /// which includes the actual `Encoding`, together with the original input
    /// in `String` format
    pub sender: oneshot::Sender<Result<(Encoding, String), TokenizerError>>,
    /// The current tracing span
    pub span: Span,
}

/// `DetokenizerRequest` - A request to decode a given token id into an actual text
pub struct DetokenizerRequest {
    /// The token id to be decoded
    pub token_id: u32,
    /// `oneshot::Sender` responsible to deliver the result of detokenization
    pub sender: oneshot::Sender<Result<String, TokenizerError>>,
    /// The current tracing span
    pub span: Span,
}

/// `Tokenizer` - a tokenizer worker
/// responsible for prepare input requests
pub struct TokenizerWorker {}

impl TokenizerWorker {
    /// Starts the tokenizer workers
    pub async fn start(
        tokenizer: Tokenizer,
        receiver: mpsc::UnboundedReceiver<TokenizerRequest>,
        workers: usize,
    ) -> Result<(), TokenizerError> {
        let mut senders = Vec::with_capacity(workers);

        for _ in 0..workers {
            let tokenizer_clone = tokenizer.clone();
            let (sender, receiver) = mpsc::unbounded_channel();
            senders.push(sender);

            // Spawning the worker
            tokio::task::spawn_blocking(|| {
                start_tokenizer_task(tokenizer_clone, receiver)?;
                Ok::<_, TokenizerError>(())
            });
        }

        // Create tokenization round robin task
        tokio::spawn(round_robin_task(receiver, senders));

        Ok(())
    }
}

/// Starts a new tokenizer tokio task
fn start_tokenizer_task(
    tokenizer: Tokenizer,
    mut receiver: mpsc::UnboundedReceiver<TokenizerRequest>,
) -> Result<(), TokenizerError> {
    // Loops over requests
    while let Some(request) = receiver.blocking_recv() {
        let TokenizerRequest {
            input,
            sender,
            span,
        } = request;
        span.in_scope(|| {
            let prepared_inputs = prepare_inputs(&tokenizer, input);
            sender.send(prepared_inputs).unwrap_or(())
        });
    }
    Ok(())
}

/// A round robin algo for tokenization tasks.
/// Check https://en.wikipedia.org/wiki/Round-robin_scheduling
/// for more details.
async fn round_robin_task(
    mut receiver: mpsc::UnboundedReceiver<TokenizerRequest>,
    senders: Vec<mpsc::UnboundedSender<TokenizerRequest>>,
) -> Result<(), TokenizerError> {
    loop {
        for sender in &senders {
            match receiver.recv().await {
                None => return Ok(()),
                Some(request) => sender.send(request)?,
            }
        }
    }
}

fn prepare_inputs(
    tokenizer: &Tokenizer,
    input: String,
) -> Result<(Encoding, String), TokenizerError> {
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
    SendError(#[from] SendError<TokenizerRequest>),
}
