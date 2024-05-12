pub use blake2::{digest::consts::U32, Blake2b, Digest};

const HASH_SIZE: usize = 32;
pub type Hash = [u8; HASH_SIZE];

pub trait Hasher {
    fn init() -> Self;
    fn hash<T: AsRef<[u8]>>(self, data: T) -> Hash;
}

impl Hasher for Blake2b<U32> {
    fn init() -> Self {
        Blake2b::new()
    }

    fn hash<T: AsRef<[u8]>>(mut self, data: T) -> Hash {
        self.update(data);
        let output = self.finalize();
        let mut hash = Hash::default();
        hash.copy_from_slice(output.as_slice());
        hash
    }
}

/// Calculates cryptographic commitments for a given data buffer.
///
/// This function takes input data, an index indicating the leaf position, 
/// and the total number of leaves in the Merkle tree.
///
/// # Parameters
///
/// - `data`: The input data buffer represented as a byte slice (`&[u8]`).
/// - `index`: The index of the data chunk within the larger dataset.
/// - `num_leaves`: The total number of leaves (data chunks) in the Merkle tree.
///
/// # Returns
///
/// A tuple containing:
/// - The cryptographic hash of the Merkle tree root (`Hash`).
/// - The cryptographic hash of the data itself and its index (`H(data | index)`).
///
/// # Panics
///
/// This function will panic if the input data slice is empty or if the computed chunks
/// for the Merkle tree are empty.
///
/// # Notes
///
/// - This function requires the `Hasher` trait to be implemented for the chosen hash algorithm (`H`).
pub fn calculate_commitment<H: Hasher, T: AsRef<[u8]>>(
    data: T,
    index: usize,
    num_leaves: usize,
) -> (Hash, Hash) {
    let data = data.as_ref();
    assert!(!data.is_empty());

    let leaves = (0..num_leaves)
        .map(|i| {
            let hasher = H::init();
            hasher.hash([data, &i.to_le_bytes()].concat())
        })
        .collect::<Vec<_>>();

    let hasher = H::init();
    let root = hasher.hash(leaves.concat().as_slice());

    (root, leaves[index])
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    #[test]
    fn test_calculate_commitment() {
        const SIZE: usize = 128;
        const INDEX: usize = 16;
        const NUM_CHUNKS: usize = 32;
        const CHUNKS_SIZE: usize = 4;

        let mut rng = rand::thread_rng();
        let data = (0..SIZE).map(|_| rng.gen::<u8>()).collect::<Vec<_>>();

        // assert root size is correct
        let (root, leaf) = calculate_commitment::<Blake2b<U32>, _>(data.clone(), INDEX, NUM_CHUNKS);
        assert_eq!(root.len(), 32);
        assert_eq!(leaf.len(), 32);

        // assert that leaves are constructed correctly
        let mut leaves = Vec::with_capacity(NUM_CHUNKS);
        for i in 0..NUM_CHUNKS {
            let mut hasher = Blake2b::<U32>::new();
            hasher.update(&data[(CHUNKS_SIZE * i)..(CHUNKS_SIZE * (i + 1))]);
            let leaf: [u8; 32] = hasher.finalize().into();
            leaves.push(leaf);
        }
        assert_eq!(leaf, leaves[INDEX]);

        // assert that root is properly constructed
        let mut hasher = Blake2b::<U32>::new();
        hasher.update(leaves.concat());
        let should_be_root: [u8; 32] = hasher.finalize().into();
        assert_eq!(root, should_be_root);
    }
}
