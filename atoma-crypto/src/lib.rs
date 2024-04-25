use rs_merkle::{Hasher, MerkleProof, MerkleTree};

pub use rs_merkle::algorithms::Sha256;

/// Calculates a cryptographic commitment and Merkle proof for a given data chunk.
///
/// This function takes input data, an index indicating the position of the data chunk
/// in a larger dataset, and the total number of leaves in the Merkle tree.
/// It computes a cryptographic commitment and Merkle proof for the data chunk.
///
/// # Parameters
///
/// - `data`: The input data chunk represented as a byte slice (`&[u8]`).
/// - `index`: The index of the data chunk within the larger dataset.
/// - `num_leaves`: The total number of leaves (data chunks) in the Merkle tree.
///
/// # Returns
///
/// A tuple containing:
/// - The cryptographic hash of the Merkle tree root (`H::Hash`).
/// - The Merkle proof (`MerkleProof<H>`), which contains the cryptographic path from
///   the leaf node to the root of the Merkle tree.
///
/// # Panics
///
/// This function will panic if the input data slice is empty or if the computed chunks
/// for the Merkle tree are empty.
///
/// # Notes
///
/// - This function requires the `Hasher` trait to be implemented for the chosen hash algorithm (`H`).
/// - The data chunk size is determined based on the total number of leaves in the Merkle tree.
/// - The function performs assertions to ensure that the input data slice and computed chunks are not empty.
pub fn calculate_commitment<H: Hasher, T: AsRef<[u8]>>(
    data: T,
    index: usize,
    num_leaves: usize,
) -> (H::Hash, MerkleProof<H>) {
    let data = data.as_ref();
    assert!(!data.is_empty());
    let chunk_size = data.len() / num_leaves;

    let chunks = data
        .chunks(chunk_size)
        .map(|buf| H::hash(buf))
        .collect::<Vec<_>>();

    assert!(!chunks.is_empty());

    let merkle_tree = MerkleTree::<H>::from_leaves(&chunks);
    let merkle_proof = merkle_tree.proof(&[index]);

    (merkle_tree.root().unwrap(), merkle_proof)
}
