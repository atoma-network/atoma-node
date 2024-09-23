use std::time::Instant;

use atoma_types::{GenerateParameters, OutputDestination};

/// A digest is a 32-byte buffer used for cryptographic hashing.
pub type Digest = Vec<u8>;

/// Metadata for a chat session, containing various parameters and identifiers
/// relevant to the session, including user information, session limits, and
/// timing details to facilitate effective chat management.
#[derive(Clone, Debug)]
pub struct ChatSessionMetadata {
    /// A unique identifier for the current chat session, which corresponds to
    /// the blockchain Ticket ID specified by the chat session event. This ID
    /// is essential for tracking and referencing the session in a distributed
    /// environment.
    pub chat_id: String,
    /// The public key of the user participating in the chat session, used for
    /// authentication and encryption purposes. This key ensures that the
    /// communication remains secure and verifies the identity of the user.
    pub user_pk: String,
    /// The timestamp when the current node received the chat session request,
    /// useful for tracking session timing and order.
    pub received_time: Instant,
    /// The chat's LLM model identifier
    pub model_id: String,
    /// The maximum context window supported by the chat model
    pub max_context_window_size: usize,
    /// The maximum number of input tokens allowed in the chat session, which
    /// limits the size of user input to prevent overflow and ensure efficient
    /// processing of requests.
    pub max_input_tokens: usize,
    /// The maximum number of output tokens allowed in the chat session, which
    /// controls the length of the generated response. This limit helps maintain
    /// concise and relevant outputs.
    pub max_output_tokens: usize,
    /// The maximum number of messages allowed within the current chat session,
    /// ensuring that the session does not exceed a predefined limit. This
    /// constraint helps manage resource usage and maintain performance.
    pub max_messages: usize,
    /// The output destination for the chat session
    pub output_destination: OutputDestination,
    /// The random seed used throughout the entire chat session for consistency,
    /// allowing for reproducible results across multiple runs. This seed is
    /// crucial for scenarios where deterministic behavior is required.
    pub random_seed: u64,
}

/// Represents the data associated with a chat session, including the context
/// of the conversation, input and output token hashes, and relevant metadata.
/// This struct is essential for managing the flow of information during a
/// chat session, ensuring that all necessary data is captured and organized
/// for serving the end-user.
#[derive(Clone, Debug)]
pub struct ChatSessionData {
    /// The last prompt submitted by the end-user. It is optional, as at the
    /// start of the chat, the node does not have access to the first prompt.
    pub last_user_prompt: Option<String>,
    /// The last time the chat session was updated
    pub last_updated_time: Instant,
    /// A collection of token IDs representing the entire chat session context.
    /// Each inner vector corresponds to a sequence of tokens for a specific
    /// message or input within the session, allowing for structured context
    /// management. It's size can be at most the maximum chat's model
    /// maximum context window size.
    pub context_token_ids: Vec<u32>,
    /// The chat inference parameters, necessary for the decoding phase
    pub params: Vec<GenerateParameters>,
    /// A collection of cryptographic hashes for each input prompt's token IDs.
    /// These hashes provide a secure way to verify the integrity of the input
    /// prompts used in the chat session, ensuring that the original prompts
    /// can be reconstructed or validated if necessary.
    pub input_prompts_hashes: Vec<Digest>,
    /// A collection of cryptographic hashes for each generated output's token IDs.
    /// Similar to input prompts, these hashes ensure the integrity and authenticity
    /// of the responses generated during the chat session, allowing for traceability
    /// and verification of the outputs.
    pub output_prompts_hashes: Vec<Digest>,
    /// The number of input tokens in the chat session
    pub num_input_tokens: usize,
    /// The number of output tokens in the chat session
    pub num_output_tokens: usize,
    /// Metadata associated with the chat session, encapsulating important
    /// information such as user details, session limits, and timing data.
    /// This metadata is crucial for managing the chat session effectively
    /// and ensuring compliance with defined parameters.
    pub chat_session_metadata: ChatSessionMetadata,
}

impl ChatSessionData {
    pub fn new(
        chat_id: String,
        user_pk: String,
        received_time: Instant,
        model_id: String,
        max_context_window_size: usize,
        max_input_tokens: usize,
        max_output_tokens: usize,
        max_messages: usize,
        output_destination: OutputDestination,
        random_seed: u64,
    ) -> Self {
        let chat_session_metadata = ChatSessionMetadata {
            chat_id: chat_id,
            user_pk: user_pk,
            received_time: received_time,
            model_id: model_id,
            max_context_window_size: max_context_window_size,
            max_input_tokens: max_input_tokens,
            max_output_tokens: max_output_tokens,
            max_messages: max_messages,
            output_destination: output_destination,
            random_seed: random_seed,
        };
        ChatSessionData {
            chat_session_metadata: chat_session_metadata,
            last_user_prompt: None,
            last_updated_time: received_time,
            context_token_ids: Vec::new(),
            params: Vec::new(),
            input_prompts_hashes: Vec::new(),
            output_prompts_hashes: Vec::new(),
            num_input_tokens: 0,
            num_output_tokens: 0,
        }
    }
}
