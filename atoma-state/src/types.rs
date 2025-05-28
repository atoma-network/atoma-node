use crate::AtomaStateManagerError;
use atoma_sui::events::{
    StackAttestationDisputeEvent, StackCreateAndUpdateEvent, StackCreatedEvent,
    StackTrySettleEvent, TaskRegisteredEvent,
};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use std::result::Result;
use tokio::sync::oneshot;
use utoipa::ToSchema;

/// Represents a task in the system
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, FromRow, ToSchema)]
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
    /// Security level of the task (encoded as an integer)
    pub security_level: i64,
    /// Optional minimum reputation score required for the task
    pub minimum_reputation_score: Option<i64>,
}

impl From<TaskRegisteredEvent> for Task {
    fn from(event: TaskRegisteredEvent) -> Self {
        Self {
            task_id: event.task_id,
            task_small_id: event.task_small_id.inner as i64,
            role: i64::from(event.role.inner),
            model_name: event.model_name,
            is_deprecated: false,
            valid_until_epoch: None,
            deprecated_at_epoch: None,
            security_level: i64::from(event.security_level.inner),
            minimum_reputation_score: event.minimum_reputation_score.map(i64::from),
        }
    }
}

/// Represents a stack of compute units for a specific task
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, FromRow, ToSchema)]
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
    pub price_per_one_million_compute_units: i64,
    /// Number of compute units already processed
    pub already_computed_units: i64,
    /// Indicates whether the stack is currently in the settle period
    pub in_settle_period: bool,
    /// Number of payload requests that were received by the node for this stack.
    pub num_total_messages: i64,
    /// Indicates whether the stack is claimed (for testing purposes)
    pub is_claimed: bool,
    /// Indicates whether the stack is locked for claim (for testing purposes)
    pub is_locked_for_claim: bool,
}

impl From<StackCreatedEvent> for Stack {
    fn from(event: StackCreatedEvent) -> Self {
        Self {
            owner_address: event.owner,
            stack_id: event.stack_id,
            stack_small_id: event.stack_small_id.inner as i64,
            task_small_id: event.task_small_id.inner as i64,
            selected_node_id: event.selected_node_id.inner as i64,
            num_compute_units: event.num_compute_units as i64,
            price_per_one_million_compute_units: event.price_per_one_million_compute_units as i64,
            already_computed_units: 0,
            in_settle_period: false,
            num_total_messages: 0,
            is_claimed: false,
            is_locked_for_claim: false,
        }
    }
}

impl From<StackCreateAndUpdateEvent> for Stack {
    fn from(event: StackCreateAndUpdateEvent) -> Self {
        Self {
            owner_address: event.owner,
            stack_small_id: event.stack_small_id.inner as i64,
            stack_id: event.stack_id,
            task_small_id: event.task_small_id.inner as i64,
            selected_node_id: event.selected_node_id.inner as i64,
            num_compute_units: event.num_compute_units as i64,
            price_per_one_million_compute_units: event.price_per_one_million_compute_units as i64,
            already_computed_units: event.already_computed_units,
            in_settle_period: false,
            num_total_messages: 1,
            is_claimed: false,
            is_locked_for_claim: false,
        }
    }
}

/// Represents the availability of a stack
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StackAvailability {
    /// The stack is available
    Available,
    /// The stack does not exist
    DoesNotExist,
    /// The stack is locked
    Locked,
    /// The stack is unavailable at the moment
    Unavailable,
}

/// Represents a settlement ticket for a compute stack
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, FromRow, ToSchema)]
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

impl TryFrom<StackTrySettleEvent> for StackSettlementTicket {
    type Error = AtomaStateManagerError;

    fn try_from(event: StackTrySettleEvent) -> Result<Self, Self::Error> {
        let num_attestation_nodes = event.requested_attestation_nodes.len();
        let expanded_size = 32 * num_attestation_nodes;

        let mut expanded_proofs = event.committed_stack_proof;
        expanded_proofs.resize(expanded_size, 0);

        let mut expanded_leaves = event.stack_merkle_leaf;
        expanded_leaves.resize(expanded_size, 0);

        Ok(Self {
            stack_small_id: event.stack_small_id.inner as i64,
            selected_node_id: event.selected_node_id.inner as i64,
            num_claimed_compute_units: event.num_claimed_compute_units as i64,
            requested_attestation_nodes: serde_json::to_string(
                &event
                    .requested_attestation_nodes
                    .into_iter()
                    .map(|id| id.inner)
                    .collect::<Vec<_>>(),
            )
            .map_err(AtomaStateManagerError::JsonParseError)?,
            committed_stack_proofs: expanded_proofs,
            stack_merkle_leaves: expanded_leaves,
            dispute_settled_at_epoch: None,
            already_attested_nodes: serde_json::to_string(&Vec::<i64>::new())
                .map_err(AtomaStateManagerError::JsonParseError)?,
            is_in_dispute: false,
            user_refund_amount: 0,
            is_claimed: false,
        })
    }
}

/// Represents a dispute in the stack attestation process
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, FromRow, ToSchema)]
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

impl From<StackAttestationDisputeEvent> for StackAttestationDispute {
    fn from(event: StackAttestationDisputeEvent) -> Self {
        Self {
            stack_small_id: event.stack_small_id.inner as i64,
            attestation_commitment: event.attestation_commitment,
            attestation_node_id: event.attestation_node_id.inner as i64,
            original_node_id: event.original_node_id.inner as i64,
            original_commitment: event.original_commitment,
        }
    }
}

/// Represents a node subscription to a task
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, FromRow, ToSchema)]
pub struct NodeSubscription {
    /// Unique small integer identifier for the node subscription
    pub node_small_id: i64,
    /// Unique small integer identifier for the task
    pub task_small_id: i64,
    /// Price per compute unit for the subscription
    pub price_per_one_million_compute_units: i64,
    /// Maximum number of compute units for the subscription
    pub max_num_compute_units: i64,
    /// Indicates whether the subscription is valid
    pub valid: bool,
}

/// Represents a node in the system
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, FromRow, ToSchema)]
pub struct Node {
    /// Unique small integer identifier for the node
    pub node_small_id: i64,
    /// Unique string identifier for the node
    pub node_id: String,
    /// Sui address of the owner of the node
    pub node_sui_address: String,
}

pub enum AtomaAtomaStateManagerEvent {
    /// Represents an update to the number of compute units in a stack
    UpdateStackNumTokens {
        /// Unique small integer identifier for the stack
        stack_small_id: i64,
        /// Estimated total number of tokens in the stack
        estimated_total_tokens: i64,
        /// Total number of tokens in the stack
        total_tokens: i64,
        /// Number of concurrent requests for the stack
        concurrent_requests: u64,
    },
    /// Represents an update to the amount of fiat currency for a user to be locked
    LockFiatAmount {
        /// User id in the proxy db
        user_id: i64,
        /// User address
        user_address: String,
        /// Amount to be locked in fiat currency for input
        estimated_input_amount: i64,
        /// Amount to be locked in fiat currency for output
        estimated_output_amount: i64,
    },
    /// Represents an update to the amount of fiat currency for a user, after the inference is done
    UpdateFiatAmount {
        /// User id in the proxy db
        user_id: i64,
        /// User address
        user_address: String,
        /// Model name used for the inference
        model_name: String,
        /// Estimated total amount for input in fiat currency
        estimated_input_amount: i64,
        /// Total amount for input in fiat currency
        input_amount: i64,
        /// Number of input tokens,
        input_tokens: i64,
        /// Estimated total amount for output in fiat currency
        estimated_output_amount: i64,
        /// Number of output tokens
        output_tokens: i64,
        /// Total amount for output in fiat currency
        output_amount: i64,
    },
    /// Gets an available stack with enough compute units for a given stack and public key
    GetAvailableStackWithComputeUnits {
        /// Unique small integer identifier for the stack
        stack_small_id: i64,
        /// Sui address of the user
        sui_address: String,
        /// Total number of compute units
        total_num_tokens: i64,
        /// Oneshot channel to send the result back to the sender channel
        result_sender: oneshot::Sender<Result<StackAvailability, AtomaStateManagerError>>,
    },
    /// Gets the pricing for a model
    GetModelPricing {
        /// Model name
        model: String,
        /// Oneshot channel to send the result back to the sender channel
        result_sender: oneshot::Sender<Result<Option<i64>, AtomaStateManagerError>>,
    },
}

/// Represents the result of updating the number of compute units in a stack and claiming funds
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, FromRow)]
pub struct UpdateStackNumComputeUnitsAndClaimFunds {
    /// Ratio of compute units that have been computed
    pub ratio: f64,

    /// Number of compute units that have been computed already for the current stack
    pub stack_computed_units: i64,

    /// Indicates whether the stack is confidential
    pub is_confidential: bool,

    /// Indicates whether the stack is locked for claim
    pub is_locked_for_claim: bool,
}
