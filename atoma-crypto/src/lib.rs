use rs_merkle::{Hasher, MerkleProof, MerkleTree};

pub use rs_merkle::algorithms::Sha256;

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
