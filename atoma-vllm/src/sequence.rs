// use std::time::Instant;

// use tracing::{info_span, Span};

// /// `LogProb` - log probabilities and token ranks.
// pub struct LogProb {
//     logprob: f32,
//     rank: Option<u32>,
//     decoded_token: Option<String>,
// }

// impl LogProb {
//     /// Constructor
//     pub fn new(logprob: f32, rank: Option<u32>, decoded_tokens: Option<String>) -> Self {
//         Self {
//             logprob,
//             rank,
//             decoded_token,
//         }
//     }
// }

// /// Stores the data, status, and block information of a sequence.
// ///
// /// Args:
// /// seq_id: The ID of the sequence.
// /// prompt: The prompt of the sequence.
// /// prompt_token_ids: The token IDs of the prompt.
// /// block_size: The block size of the sequence. Should be the same as the
// ///    block size used by the block manager and cache engine.
// ///
// /// Warn: Contrary to vLLM, we are not dealing with LoRA requests
// pub struct Sequence {
//     /// Sequence Id,
//     sequence_id: u64,
//     /// End of sequence token
//     eos_token_id: Option<u32>,
//     /// Prompt
//     prompt: String,
//     /// Prompt associated token ids
//     prompt_token_ids: Vec<u32>,
//     /// Block size
//     block_size: usize,
//     /// Span
//     span: Span,
// }

// impl Sequence {
//     /// Constructor
//     pub fn new(
//         sequence_id: u64,
//         eos_token_id: Option<u32>,
//         prompt: String,
//         prompt_token_ids: Vec<u32>,
//         block_size: usize,
//     ) -> Self {
//         Self {
//             sequence_id,
//             eos_token_id,
//             prompt,
//             prompt_token_ids,
//             block_size,
//             span: info_span!("sequence"),
//         }
//     }
// }

// /// `Sequence` - A group of sequences that are generated from the same prompt.
// ///
// /// Args:
// /// request_id: The ID of the request.
// /// seqs: The list of sequences.
// /// sampling_params: The sampling parameters used to generate the outputs.
// /// arrival_time: The arrival time of the request.
// /// multi_modal_data: Multi modal data associated with the request.
// /// embeddings: The embeddings vectors of the prompt of the sequence group
// ///    for an embedding model.
// ///
// /// Warn: Our implementation does not consider LoRA and embeddings requests (contrary to vLLM).
// pub struct SequenceGroup {
//     /// Request Id
//     request_id: String,
//     /// Sequences
//     sequences: Vec<Sequence>,
//     /// Request's arrival time
//     arrival_time: Instant,
//     /// Multi modal data
//     multi_model_data: Vec<MultiModelData>,
// }
