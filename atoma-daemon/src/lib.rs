#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_sign_loss)]

pub(crate) mod components;
pub mod config;
pub(crate) mod handlers;
pub mod server;
pub mod telemetry;
pub mod types;

pub use crate::{config::AtomaDaemonConfig, server::DaemonState};

use axum::http::StatusCode;
use blake2::{Blake2b, Digest};
use rs_merkle::Hasher;
use tracing::{error, instrument};

/// A hasher implementation using the Blake2b algorithm.
///
/// This struct implements the `Hasher` trait, allowing it to be used
/// for creating Merkle trees with the `rs_merkle` crate. The Blake2b
/// algorithm is a cryptographic hash function that provides a high
/// level of security and is suitable for use in various applications
/// requiring data integrity and authenticity.
#[derive(Clone)]
pub struct Blake2bHasher;

impl Hasher for Blake2bHasher {
    type Hash = [u8; 32];

    fn hash(data: &[u8]) -> Self::Hash {
        let mut hasher = Blake2b::new();
        hasher.update(data);
        hasher.finalize().into()
    }
}

/// A proof of commitment for a stack in the Atoma Network protocol.
///
/// This struct contains the root and leaf of a Merkle tree, which are used
/// to verify the integrity and authenticity of a stack's data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedStackProof {
    /// A vector of bytes representing the final commitment proof. This
    /// is computed by iteratively hashing the Merkle root with a prefix index.
    pub root: Vec<u8>,
    /// A vector of bytes representing the Merkle leaf. This is computed
    /// by hashing the Merkle root with a prefix key value of 0.
    pub leaf: Vec<u8>,
}

/// A struct representing the indices of a node in the attestation nodes list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttestationNodeIndices {
    /// The index of the attestation node in the attestation nodes list.
    pub attestation_node_index: usize,
    /// The index of the node small ID in the node small IDs list.
    pub node_small_id_index: usize,
}

/// Calculates the indices of nodes in the attestation nodes list that match the provided small IDs.
///
/// This function searches through the attestation nodes list to find matches with the provided
/// node small IDs and returns their respective positions in both lists.
///
/// # Arguments
///
/// * `node_small_ids` - A slice of node small IDs to search for in the attestation nodes list
/// * `attestation_nodes` - A slice of attestation nodes to search through
///
/// # Returns
///
/// Returns a `Result` containing a vector of `AttestationNodeIndices` on success, or a `StatusCode`
/// on failure. Each `AttestationNodeIndices` contains:
/// - `attestation_node_index`: The position of the matching node in the attestation nodes list
/// - `node_small_id_index`: The position of the matching node in the node small IDs list
///
/// # Errors
///
/// Returns `StatusCode::BAD_REQUEST` if no matches are found between the node small IDs and
/// attestation nodes lists.
///
/// # Example
///
/// ```rust,ignore
/// let node_small_ids = vec![1, 2, 3];
/// let attestation_nodes = vec![2, 4, 1];
///
/// match calculate_node_index(&node_small_ids, &attestation_nodes) {
///     Ok(indices) => {
///         for index in indices {
///             println!(
///                 "Match found at attestation_node_index: {}, node_small_id_index: {}",
///                 index.attestation_node_index,
///                 index.node_small_id_index
///             );
///         }
///     }
///     Err(status) => {
///         eprintln!("Error: {:?}", status);
///     }
/// }
/// ```
#[instrument(level = "trace", skip(node_small_ids, attestation_nodes))]
pub(crate) fn calculate_node_index(
    node_small_ids: &[i64],
    attestation_nodes: &[i64],
) -> Result<Vec<AttestationNodeIndices>, StatusCode> {
    let mut node_indices = Vec::new();
    for (i, attestation_node_id) in attestation_nodes.iter().enumerate() {
        for (j, node_small_id) in node_small_ids.iter().enumerate() {
            if attestation_node_id == node_small_id {
                node_indices.push(AttestationNodeIndices {
                    attestation_node_index: i,
                    node_small_id_index: j,
                });
                break;
            }
        }
    }
    if node_indices.is_empty() {
        error!("Node index not found");
        return Err(StatusCode::BAD_REQUEST);
    }
    Ok(node_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_inputs() {
        let result = calculate_node_index(&[], &[]);
        assert_eq!(result, Err(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn test_no_matches() {
        let node_small_ids = vec![1, 2, 3];
        let attestation_nodes = vec![4, 5, 6];
        let result = calculate_node_index(&node_small_ids, &attestation_nodes);
        assert_eq!(result, Err(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn test_single_match() {
        let node_small_ids = vec![1, 2, 3];
        let attestation_nodes = vec![4, 2, 5];
        let result = calculate_node_index(&node_small_ids, &attestation_nodes);

        assert!(result.is_ok());
        let indices = result.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].attestation_node_index, 1);
        assert_eq!(indices[0].node_small_id_index, 1);
    }

    #[test]
    fn test_multiple_matches() {
        let node_small_ids = vec![1, 2, 3];
        let attestation_nodes = vec![2, 3, 1];
        let result = calculate_node_index(&node_small_ids, &attestation_nodes);

        assert!(result.is_ok());
        let indices = result.unwrap();
        assert_eq!(indices.len(), 3);

        // Check first match
        assert_eq!(indices[0].attestation_node_index, 0);
        assert_eq!(indices[0].node_small_id_index, 1);

        // Check second match
        assert_eq!(indices[1].attestation_node_index, 1);
        assert_eq!(indices[1].node_small_id_index, 2);

        // Check third match
        assert_eq!(indices[2].attestation_node_index, 2);
        assert_eq!(indices[2].node_small_id_index, 0);
    }

    #[test]
    fn test_duplicate_values() {
        let node_small_ids = vec![1, 1, 2];
        let attestation_nodes = vec![1, 2, 1];
        let result = calculate_node_index(&node_small_ids, &attestation_nodes);

        assert!(result.is_ok());
        let indices = result.unwrap();
        assert_eq!(indices.len(), 3);

        // First '1' matches with first attestation node
        assert_eq!(indices[0].attestation_node_index, 0);
        assert_eq!(indices[0].node_small_id_index, 0);

        // First '1' matches with third attestation node
        assert_eq!(indices[1].attestation_node_index, 1);
        assert_eq!(indices[1].node_small_id_index, 2);

        // Second '1' matches with first attestation node
        assert_eq!(indices[2].attestation_node_index, 2);
        assert_eq!(indices[2].node_small_id_index, 0);
    }

    #[test]
    fn test_different_length_inputs() {
        let node_small_ids = vec![1, 2, 3, 4];
        let attestation_nodes = vec![2, 3];
        let result = calculate_node_index(&node_small_ids, &attestation_nodes);

        assert!(result.is_ok());
        let indices = result.unwrap();
        assert_eq!(indices.len(), 2);

        assert_eq!(indices[0].attestation_node_index, 0);
        assert_eq!(indices[0].node_small_id_index, 1);

        assert_eq!(indices[1].attestation_node_index, 1);
        assert_eq!(indices[1].node_small_id_index, 2);
    }

    #[test]
    fn test_negative_values() {
        let node_small_ids = vec![-1, -2, 3];
        let attestation_nodes = vec![-2, 3, -1];
        let result = calculate_node_index(&node_small_ids, &attestation_nodes);

        assert!(result.is_ok());
        let indices = result.unwrap();
        assert_eq!(indices.len(), 3);

        assert_eq!(indices[0].attestation_node_index, 0);
        assert_eq!(indices[0].node_small_id_index, 1);

        assert_eq!(indices[1].attestation_node_index, 1);
        assert_eq!(indices[1].node_small_id_index, 2);

        assert_eq!(indices[2].attestation_node_index, 2);
        assert_eq!(indices[2].node_small_id_index, 0);
    }

    #[test]
    fn test_large_numbers() {
        let node_small_ids = vec![i64::MAX, i64::MIN, 0];
        let attestation_nodes = vec![0, i64::MAX];
        let result = calculate_node_index(&node_small_ids, &attestation_nodes);

        assert!(result.is_ok());
        let indices = result.unwrap();
        assert_eq!(indices.len(), 2);

        assert_eq!(indices[0].attestation_node_index, 0);
        assert_eq!(indices[0].node_small_id_index, 2);

        assert_eq!(indices[1].attestation_node_index, 1);
        assert_eq!(indices[1].node_small_id_index, 0);
    }

    #[test]
    fn test_single_element_inputs() {
        let node_small_ids = vec![42];
        let attestation_nodes = vec![42];
        let result = calculate_node_index(&node_small_ids, &attestation_nodes);

        assert!(result.is_ok());
        let indices = result.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].attestation_node_index, 0);
        assert_eq!(indices[0].node_small_id_index, 0);
    }
}
