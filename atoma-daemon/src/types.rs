use serde::{Deserialize, Serialize};
use sui_sdk::types::base_types::ObjectID;

/// Represents a request to register a node.
///
/// This struct is used to encapsulate the necessary parameters
/// for registering a node, including optional gas-related fields.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeRegistrationRequest {
    /// Optional gas object ID.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas: Option<ObjectID>,

    /// Optional gas budget.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_budget: Option<u64>,

    /// Optional gas price.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_price: Option<u64>,
}

/// Represents a response to a node registration request.
///
/// This struct contains the transaction digest, which is a unique
/// identifier for the transaction associated with the node registration.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeRegistrationResponse {
    /// The transaction digest.
    /// This is a unique identifier for the transaction.
    pub tx_digest: String,
}

/// Represents a request to subscribe to a node model.
///
/// This struct encapsulates the necessary parameters for subscribing
/// to a node model, including optional gas-related fields.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeModelSubscriptionRequest {
    /// The name of the model to subscribe to.
    pub model_name: String,

    /// The echelon ID associated with the subscription.
    pub echelon_id: u64,

    /// Optional node badge ID.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_badge_id: Option<ObjectID>,

    /// Optional gas object ID.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas: Option<ObjectID>,

    /// Optional gas budget.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_budget: Option<u64>,

    /// Optional gas price.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_price: Option<u64>,
}

/// Represents a response to a node model subscription request.
///
/// This struct contains the transaction digest, which is a unique
/// identifier for the transaction associated with the node model subscription.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeModelSubscriptionResponse {
    /// The transaction digest.
    /// This is a unique identifier for the transaction.
    pub tx_digest: String,
}

/// Represents a request to subscribe to a node task.
///
/// This struct encapsulates the necessary parameters for subscribing
/// to a node task, including optional gas-related fields.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeTaskSubscriptionRequest {
    /// The small ID of the task to subscribe to.
    pub task_small_id: i64,

    /// Optional small ID of the node.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_small_id: Option<i64>,

    /// The price per compute unit.
    pub price_per_compute_unit: u64,

    /// The maximum number of compute units.
    pub max_num_compute_units: u64,

    /// Optional gas object ID.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas: Option<ObjectID>,

    /// Optional gas budget.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_budget: Option<u64>,

    /// Optional gas price.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_price: Option<u64>,
}

/// Represents a response to a node task subscription request.
///
/// This struct contains the transaction digest, which is a unique
/// identifier for the transaction associated with the node task subscription.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeTaskSubscriptionResponse {
    /// The transaction digest.
    /// This is a unique identifier for the transaction.
    pub tx_digest: String,
}

/// Represents a request to unsubscribe from a node task.
///
/// This struct encapsulates the necessary parameters for unsubscribing
/// from a node task, including optional gas-related fields.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeTaskUnsubscriptionRequest {
    /// The small ID of the task to unsubscribe from.
    pub task_small_id: i64,

    /// Optional small ID of the node.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_small_id: Option<i64>,

    /// Optional gas object ID.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas: Option<ObjectID>,

    /// Optional gas budget.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_budget: Option<u64>,

    /// Optional gas price.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_price: Option<u64>,
}

/// Represents a response to a node task unsubscription request.
///
/// This struct contains the transaction digest, which is a unique
/// identifier for the transaction associated with the node task unsubscription.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeTaskUnsubscriptionResponse {
    /// The transaction digest.
    /// This is a unique identifier for the transaction.
    pub tx_digest: String,
}

/// Represents a request to try settling a stack.
///
/// This struct encapsulates the necessary parameters for attempting
/// to settle a stack, including optional gas-related fields.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeTrySettleStacksRequest {
    /// The small IDs of the stacks to settle.
    pub stack_small_ids: Vec<i64>,

    /// The number of compute units claimed.
    pub num_claimed_compute_units: u64,

    /// Optional small ID of the node.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_small_id: Option<i64>,

    /// Optional gas object ID.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas: Option<ObjectID>,

    /// Optional gas budget.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_budget: Option<u64>,

    /// Optional gas price.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_price: Option<u64>,
}

/// Represents a response to a node try settle stack request.
///
/// This struct contains the transaction digest, which is a unique
/// identifier for the transaction associated with the attempt to settle a stack.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeTrySettleStacksResponse {
    /// The transaction digests.
    /// This is a unique identifier for the transaction.
    pub tx_digests: Vec<String>,
}

/// Represents a request to submit a node attestation proof.
///
/// This struct encapsulates the necessary parameters for submitting
/// a node attestation proof.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeAttestationProofRequest {
    /// The small IDs of the stacks to attest to.
    pub stack_small_ids: Vec<i64>,

    /// Optional small ID of the node.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_small_id: Option<i64>,

    /// Optional gas object ID.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas: Option<ObjectID>,

    /// Optional gas budget.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_budget: Option<u64>,

    /// Optional gas price.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_price: Option<u64>,
}

/// Represents a response to a node attestation proof request.
///
/// This struct contains the transaction digests, which are unique
/// identifiers for the transactions associated with the attestation proof.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeAttestationProofResponse {
    /// The transaction digests.
    pub tx_digests: Vec<String>,
}

/// Represents a request to claim funds from a stack.
///
/// This struct encapsulates the necessary parameters for claiming funds
/// from a stack, including optional gas-related fields.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeClaimFundsRequest {
    /// The small IDs of the stacks to claim funds from.
    pub stack_small_ids: Vec<i64>,

    /// Optional small ID of the node.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_small_id: Option<i64>,

    /// Optional gas object ID.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas: Option<ObjectID>,

    /// Optional gas budget.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_budget: Option<u64>,

    /// Optional gas price.
    /// If not provided, the default is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_price: Option<u64>,
}

/// Represents a response to a node claim funds request.
///
/// This struct contains the transaction digest, which is a unique
/// identifier for the transaction associated with the claim funds.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeClaimFundsResponse {
    /// The associated transaction digest.
    pub tx_digest: String,
}
