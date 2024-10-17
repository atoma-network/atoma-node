use serde::{Deserialize, Serialize};
use sqlx::FromRow;

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct Task {
    pub task_small_id: i64,
    pub task_id: String,
    pub role: i64,
    pub model_name: Option<String>,
    pub is_deprecated: bool,
    pub valid_until_epoch: Option<i64>,
    pub deprecated_at_epoch: Option<i64>,
    pub optimizations: String,
    pub security_level: i64,
    pub task_metrics_compute_unit: i64,
    pub task_metrics_time_unit: Option<i64>,
    pub task_metrics_value: Option<i64>,
    pub minimum_reputation_score: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct Stack {
    pub stack_small_id: i64,
    pub stack_id: String,
    pub task_small_id: i64,
    pub selected_node_id: i64,
    pub num_compute_units: i64,
    pub price: i64,
    pub already_computed_units: i64,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct StackSettlementTicket {
    pub stack_small_id: i64,
    pub selected_node_id: i64,
    pub num_claimed_compute_units: i64,
    pub requested_attestation_nodes: String,
    pub committed_stack_proof: Vec<u8>,
    pub stack_merkle_leaf: Vec<u8>,
    pub dispute_settled_at_epoch: Option<i64>,
    pub already_attested_nodes: String,
    pub is_in_dispute: bool,
    pub user_refund_amount: i64,
    pub is_claimed: bool,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct StackAttestationDispute {
    pub stack_small_id: i64,
    pub attestation_commitment: Vec<u8>,
    pub attestation_node_id: i64,
    pub original_node_id: i64,
    pub original_commitment: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct StackAttestation {
    pub stack_small_id: i64,
    pub attestation_commitment: Vec<u8>,
    pub attestation_node_id: i64,
    pub original_node_id: i64,
    pub original_commitment: Vec<u8>,
}
