pub mod config;
pub mod daemon;
pub mod types;

use axum::http::StatusCode;
use blake2::{
    digest::generic_array::{typenum::U32, GenericArray},
    Blake2b, Digest,
};
use rs_merkle::{Hasher, MerkleTree};
use tracing::{error, instrument};

/// Number of attestation nodes in the Atoma Network protocol
const PROTOCOL_NUMBER_OF_ATTESTATION_NODES: u64 = 1;

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

/// Computes a committed stack proof for a given total hash of a stack.
///
/// # Arguments
///
/// * `total_hash` - A vector of bytes representing the total hash of the stack.
///   The length of this vector must be a multiple of 32, as each 32-byte chunk
///   represents a leaf in the Merkle tree.
///
/// # Returns
///
/// Returns a `Result` containing a `CommittedStackProof` on success, or a
/// `StatusCode` on failure. The `CommittedStackProof` includes:
/// - `root`: The final commitment proof, computed by iteratively hashing the
///   Merkle root with a prefix index.
/// - `leaf`: The Merkle leaf, computed by hashing the Merkle root with a
///   prefix key value of 0.
///
/// # Errors
///
/// Returns `StatusCode::INTERNAL_SERVER_ERROR` if:
/// - The `total_hash` is empty.
/// - The length of `total_hash` is not a multiple of 32.
///
/// # Process
///
/// 1. **Validation**: The function first checks if the `total_hash` is empty or
///    if its length is not a multiple of 32. If either condition is true, an
///    error is returned.
///
/// 2. **Merkle Tree Construction**: The `total_hash` is divided into 32-byte
///    chunks, each representing a leaf of the Merkle tree. These leaves are
///    used to construct a Merkle tree using the `Blake2bHasher`.
///
/// 3. **Merkle Root Calculation**: The root of the Merkle tree is computed.
///    This root represents the hash of the entire stack's lifetime.
///
/// 4. **Merkle Leaf Calculation**: A new Blake2b hasher is initialized, and the
///    Merkle root is hashed with a prefix key value of 0. The result is the
///    `stack_merkle_leaf`.
///
/// 5. **Commitment Proof Calculation**: Another Blake2b hasher is initialized.
///    The Merkle root is iteratively hashed with prefix indices ranging from 0
///    to `PROTOCOL_NUMBER_OF_ATTESTATION_NODES - 1`. The result is the
///    `committed_stack_proof`.
///
/// # Example
///
/// ```rust,ignore
/// let total_hash = vec![/* 32-byte aligned data */];
/// match compute_committed_stack_proof(total_hash) {
///     Ok(proof) => {
///         println!("Root: {:?}", proof.root);
///         println!("Leaf: {:?}", proof.leaf);
///     }
///     Err(status) => {
///         eprintln!("Error: {:?}", status);
///     }
/// }
/// ```
#[instrument(
    level = "trace",
    skip(total_hash),
    fields(total_hash_len = total_hash.len())
)]
pub(crate) fn compute_committed_stack_proof(
    total_hash: &[u8],
    node_index: u64,
) -> Result<CommittedStackProof, StatusCode> {
    if total_hash.is_empty() {
        error!("Stack total hash is empty");
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }
    if total_hash.len() % 32 != 0 {
        error!("Stack total hash length is not a multiple of 32");
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    let stack_merkle_leaves: Vec<[u8; 32]> = total_hash
        .chunks(32)
        .map(|chunk| chunk.try_into().unwrap())
        .collect();

    // Compute the merkle tree root hash for the entire stack lifetime
    let stack_merkle_tree = MerkleTree::<Blake2bHasher>::from_leaves(&stack_merkle_leaves);
    let stack_merkle_root = stack_merkle_tree.root().unwrap();

    // Since this logic runs on the host node (that settles the stack),
    // it needs to be committed with appended key value 0
    let mut blake2b = Blake2b::new();
    blake2b.update([node_index as u8]); // NOTE: attestation nodes size is to be less than 256
    blake2b.update(stack_merkle_root);
    let stack_merkle_leaf_ga: GenericArray<u8, U32> = blake2b.finalize();
    let stack_merkle_leaf = stack_merkle_leaf_ga.as_slice().to_vec();

    let mut blake2b = Blake2b::new();
    for i in 0..PROTOCOL_NUMBER_OF_ATTESTATION_NODES {
        blake2b.update([i as u8]);
        blake2b.update(stack_merkle_root);
    }
    let committed_stack_proof_ga: GenericArray<u8, U32> = blake2b.finalize();
    let committed_stack_proof = committed_stack_proof_ga.as_slice().to_vec();

    Ok(CommittedStackProof {
        root: committed_stack_proof,
        leaf: stack_merkle_leaf,
    })
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
    fn test_empty_total_hash() {
        let total_hash = vec![];
        let result = compute_committed_stack_proof(&total_hash, 0);
        assert_eq!(result, Err(StatusCode::INTERNAL_SERVER_ERROR));
    }

    #[test]
    fn test_non_multiple_of_32_total_hash() {
        let total_hash = vec![0u8; 31]; // Length is not a multiple of 32
        let result = compute_committed_stack_proof(&total_hash, 0);
        assert_eq!(result, Err(StatusCode::INTERNAL_SERVER_ERROR));
    }

    #[test]
    fn test_valid_total_hash_single_chunk() {
        let total_hash = vec![0u8; 32]; // Single 32-byte chunk
        let result = compute_committed_stack_proof(&total_hash, 0);
        assert!(result.is_ok());

        let proof = result.unwrap();
        assert_eq!(proof.root.len(), 32);
        assert_eq!(proof.leaf.len(), 32);

        // Verify the Merkle root
        let expected_tree =
            MerkleTree::<Blake2bHasher>::from_leaves(&[total_hash.try_into().unwrap()]);
        let expected_root = expected_tree.root().unwrap();
        let mut blake2b = Blake2b::new();
        blake2b.update([0]);
        blake2b.update(expected_root);
        let expected_leaf: GenericArray<u8, U32> = blake2b.finalize();
        assert_eq!(proof.leaf, expected_leaf.as_slice().to_vec());

        // Verify the commitment proof
        let mut blake2b = Blake2b::new();
        for i in 0..PROTOCOL_NUMBER_OF_ATTESTATION_NODES {
            blake2b.update([i as u8]);
            blake2b.update(expected_root);
        }
        let expected_commitment: GenericArray<u8, U32> = blake2b.finalize();
        assert_eq!(proof.root, expected_commitment.as_slice().to_vec());
    }

    #[test]
    fn test_valid_total_hash_multiple_chunks() {
        let total_hash = vec![0u8; 64]; // Two 32-byte chunks
        let result = compute_committed_stack_proof(&total_hash, 0);
        assert!(result.is_ok());

        let proof = result.unwrap();
        assert_eq!(proof.root.len(), 32);
        assert_eq!(proof.leaf.len(), 32);

        // Verify the Merkle root
        let leaves: Vec<[u8; 32]> = total_hash
            .chunks(32)
            .map(|chunk| chunk.try_into().unwrap())
            .collect();
        let expected_tree = MerkleTree::<Blake2bHasher>::from_leaves(&leaves);
        let expected_root = expected_tree.root().unwrap();
        let mut blake2b = Blake2b::new();
        blake2b.update([0]);
        blake2b.update(expected_root);
        let expected_leaf: GenericArray<u8, U32> = blake2b.finalize();
        assert_eq!(proof.leaf, expected_leaf.as_slice().to_vec());

        // Verify the commitment proof
        let mut blake2b = Blake2b::new();
        for i in 0..PROTOCOL_NUMBER_OF_ATTESTATION_NODES {
            blake2b.update([i as u8]);
            blake2b.update(expected_root);
        }
        let expected_commitment: GenericArray<u8, U32> = blake2b.finalize();
        assert_eq!(proof.root, expected_commitment.as_slice().to_vec());
    }

    #[test]
    fn test_different_data_same_length() {
        let total_hash1 = vec![0u8; 64];
        let total_hash2 = vec![1u8; 64];

        let result1 = compute_committed_stack_proof(&total_hash1, 0);
        let result2 = compute_committed_stack_proof(&total_hash2, 0);

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        let proof1 = result1.unwrap();
        let proof2 = result2.unwrap();

        assert_ne!(proof1.root, proof2.root);
        assert_ne!(proof1.leaf, proof2.leaf);

        // Verify the Merkle roots
        let leaves1: Vec<[u8; 32]> = total_hash1
            .chunks(32)
            .map(|chunk| chunk.try_into().unwrap())
            .collect();
        let expected_tree1 = MerkleTree::<Blake2bHasher>::from_leaves(&leaves1);
        let expected_root1 = expected_tree1.root().unwrap();
        let mut blake2b = Blake2b::new();
        blake2b.update([0]);
        blake2b.update(expected_root1);
        let expected_leaf1: GenericArray<u8, U32> = blake2b.finalize();
        assert_eq!(proof1.leaf, expected_leaf1.as_slice().to_vec());

        let leaves2: Vec<[u8; 32]> = total_hash2
            .chunks(32)
            .map(|chunk| chunk.try_into().unwrap())
            .collect();
        let expected_tree2 = MerkleTree::<Blake2bHasher>::from_leaves(&leaves2);
        let expected_root2 = expected_tree2.root().unwrap();
        let mut blake2b = Blake2b::new();
        blake2b.update([0]);
        blake2b.update(expected_root2);
        let expected_leaf2: GenericArray<u8, U32> = blake2b.finalize();
        assert_eq!(proof2.leaf, expected_leaf2.as_slice().to_vec());
    }

    #[test]
    fn test_large_total_hash() {
        let total_hash = vec![0u8; 3200]; // 100 chunks of 32 bytes
        let result = compute_committed_stack_proof(&total_hash, 0);
        assert!(result.is_ok());

        let proof = result.unwrap();
        assert_eq!(proof.root.len(), 32);
        assert_eq!(proof.leaf.len(), 32);

        // Verify the Merkle root
        let leaves: Vec<[u8; 32]> = total_hash
            .chunks(32)
            .map(|chunk| chunk.try_into().unwrap())
            .collect();
        let expected_tree = MerkleTree::<Blake2bHasher>::from_leaves(&leaves);
        let expected_root = expected_tree.root().unwrap();
        let mut blake2b = Blake2b::new();
        blake2b.update([0]);
        blake2b.update(expected_root);
        let expected_leaf: GenericArray<u8, U32> = blake2b.finalize();
        assert_eq!(proof.leaf, expected_leaf.as_slice().to_vec());

        // Verify the commitment proof
        let mut blake2b = Blake2b::new();
        for i in 0..PROTOCOL_NUMBER_OF_ATTESTATION_NODES {
            blake2b.update([i as u8]);
            blake2b.update(expected_root);
        }
        let expected_commitment: GenericArray<u8, U32> = blake2b.finalize();
        assert_eq!(proof.root, expected_commitment.as_slice().to_vec());
    }

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
