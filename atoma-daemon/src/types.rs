use serde::{Deserialize, Serialize};
use sui_sdk::types::base_types::ObjectID;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeRegistrationRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas: Option<ObjectID>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_budget: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_price: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeRegistrationResponse {
    pub tx_digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeModelSubscriptionRequest {
    pub model_name: String,
    pub echelon_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_badge_id: Option<ObjectID>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas: Option<ObjectID>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_budget: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_price: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeModelSubscriptionResponse {
    pub tx_digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeTaskSubscriptionRequest {
    pub task_small_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_small_id: Option<u64>,
    pub price_per_compute_unit: u64,
    pub max_num_compute_units: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas: Option<ObjectID>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_budget: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_price: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeTaskSubscriptionResponse {
    pub tx_digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeTaskUnsubscriptionRequest {
    pub task_small_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_small_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas: Option<ObjectID>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_budget: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_price: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeTaskUnsubscriptionResponse {
    pub tx_digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeTrySettleStackRequest {
    pub stack_small_id: u64,
    pub num_claimed_compute_units: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_small_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas: Option<ObjectID>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_budget: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_price: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeTrySettleStackResponse {
    pub tx_digest: String,
}
