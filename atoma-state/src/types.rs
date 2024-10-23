use serde::{Deserialize, Serialize};
use sqlx::FromRow;

/// Represents a task in the system
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, FromRow)]
pub struct Task {
    /// Unique small integer identifier for the task
    pub task_small_id: i64,
    /// Unique string identifier for the task
    pub task_id: String,
    /// Role associated with the task (encoded as an integer)
    pub role: i64,
    /// Optional name of the model used for the task
    pub model_name: Option<String>,
    /// Indicates whether the task is deprecated
    pub is_deprecated: bool,
    /// Optional epoch timestamp until which the task is valid
    pub valid_until_epoch: Option<i64>,
    /// Optional epoch timestamp when the task was deprecated
    pub deprecated_at_epoch: Option<i64>,
    /// String representation of task optimizations
    pub optimizations: String,
    /// Security level of the task (encoded as an integer)
    pub security_level: i64,
    /// Compute units required for the task
    pub task_metrics_compute_unit: i64,
    /// Optional time units for task metrics
    pub task_metrics_time_unit: Option<i64>,
    /// Optional value for task metrics
    pub task_metrics_value: Option<i64>,
    /// Optional minimum reputation score required for the task
    pub minimum_reputation_score: Option<i64>,
}

/// Represents a stack of compute units for a specific task
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, FromRow)]
pub struct Stack {
    /// Address of the owner of the stack
    pub owner_address: String,
    /// Unique small integer identifier for the stack
    pub stack_small_id: i64,
    /// Unique string identifier for the stack
    pub stack_id: String,
    /// Small integer identifier of the associated task
    pub task_small_id: i64,
    /// Identifier of the selected node for computation
    pub selected_node_id: i64,
    /// Total number of compute units in this stack
    pub num_compute_units: i64,
    /// Price of the stack (likely in smallest currency unit)
    pub price: i64,
    /// Number of compute units already processed
    pub already_computed_units: i64,
    /// Indicates whether the stack is currently in the settle period
    pub in_settle_period: bool,
    /// Joint concatenation of SHA256 hashes of each payload and response pairs that was already processed
    /// by the node for this stack.
    pub total_hash: Vec<u8>,
    /// Number of payload requests that were received by the node for this stack.
    pub num_total_messages: i64,
}

/// Represents a settlement ticket for a compute stack
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, FromRow)]
pub struct StackSettlementTicket {
    /// Unique small integer identifier for the stack
    pub stack_small_id: i64,
    /// Identifier of the node selected for computation
    pub selected_node_id: i64,
    /// Number of compute units claimed to be processed
    pub num_claimed_compute_units: i64,
    /// Comma-separated list of node IDs requested for attestation
    pub requested_attestation_nodes: String,
    /// Cryptographic proof of the committed stack state
    pub committed_stack_proofs: Vec<u8>,
    /// Merkle leaf representing the stack in a larger tree structure
    pub stack_merkle_leaves: Vec<u8>,
    /// Optional epoch timestamp when a dispute was settled
    pub dispute_settled_at_epoch: Option<i64>,
    /// Comma-separated list of node IDs that have already attested
    pub already_attested_nodes: String,
    /// Indicates whether the stack is currently in a dispute
    pub is_in_dispute: bool,
    /// Amount to be refunded to the user (likely in smallest currency unit)
    pub user_refund_amount: i64,
    /// Indicates whether the settlement ticket has been claimed
    pub is_claimed: bool,
}

/// Represents a dispute in the stack attestation process
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, FromRow)]
pub struct StackAttestationDispute {
    /// Unique small integer identifier for the stack involved in the dispute
    pub stack_small_id: i64,
    /// Cryptographic commitment provided by the attesting node
    pub attestation_commitment: Vec<u8>,
    /// Identifier of the node that provided the attestation
    pub attestation_node_id: i64,
    /// Identifier of the original node that performed the computation
    pub original_node_id: i64,
    /// Original cryptographic commitment provided by the computing node
    pub original_commitment: Vec<u8>,
}

/// Represents a node subscription to a task
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, FromRow)]
pub struct NodeSubscription {
    /// Unique small integer identifier for the node subscription
    pub node_small_id: i64,
    /// Unique small integer identifier for the task
    pub task_small_id: i64,
    /// Price per compute unit for the subscription
    pub price_per_compute_unit: i64,
    /// Maximum number of compute units for the subscription
    pub max_num_compute_units: i64,
    /// Indicates whether the subscription is valid
    pub valid: bool,
}
