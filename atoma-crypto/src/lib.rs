pub use blake2::{digest::consts::U32, Blake2b, Digest};

const HASH_SIZE: usize = 32;
pub type Hash = [u8; HASH_SIZE];

/// `Hasher` - A basic hasher trait
///
/// This trait defines the interface for a cryptographic hasher, which is responsible for
/// generating a fixed-size hash from an arbitrary input data buffer. Implementations of
/// this trait should provide methods to initialize the hasher and to compute the hash
/// of a given data buffer.
///
/// # Example
///
/// ```
/// use atoma_crypto::Hasher;
///
/// struct MyHasher;
///
/// impl Hasher for MyHasher {
///     fn init() -> Self {
///         MyHasher
///     }
///
///     fn hash<T: AsRef<[u8]>>(self, data: T) -> Hash {
///         // Implementation of the hash function
///     }
/// }
///
/// let hasher = MyHasher::init();
/// let data = b"example data";
/// let hash = hasher.hash(data);
/// println!("{:x?}", hash);
/// ```
pub trait Hasher {
    /// Initializes a new instance of the hasher.
    ///
    /// This method should return a new instance of the hasher, ready to process data.
    ///
    /// # Returns
    ///
    /// A new instance of the hasher.
    fn init() -> Self;

    /// Hashes the provided data buffer and returns the resulting hash.
    ///
    /// This method takes an input data buffer, processes it using the hasher's algorithm,
    /// and returns the resulting fixed-size hash.
    ///
    /// # Parameters
    ///
    /// - `data`: The input data buffer to be hashed, represented as a type that can be
    ///   referenced as a byte slice (`T: AsRef<[u8]>`).
    ///
    /// # Returns
    ///
    /// A fixed-size array (`Hash`) representing the cryptographic hash of the input data.
    fn hash<T: AsRef<[u8]>>(self, data: T) -> Hash;
}

/// Implementation of the `Hasher` trait to the `Blake2b` type
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

/// Computes the Blake2b hash of the given data.
///
/// This function takes an input data buffer and returns its cryptographic hash
/// using the Blake2b hashing algorithm with a 32-byte output.
///
/// # Parameters
///
/// - `data`: The input data buffer represented as a type that can be referenced as a byte slice (`T: AsRef<[u8]>`).
///
/// # Returns
///
/// A 32-byte array (`Hash`) representing the cryptographic hash of the input data.
///
/// # Examples
///
/// ```
/// use atoma_crypto::blake2b_hash;
///
/// let data = b"hello world";
/// let hash = blake2b_hash(data);
/// println!("{:x?}", hash);
/// ```
///
/// # Notes
///
/// - This function uses the `Blake2b` hashing algorithm with a 32-byte output size.
pub fn blake2b_hash<T: AsRef<[u8]>>(data: T) -> Hash {
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Calculates cryptographic commitments for a given data buffer.
///
/// This function computes the cryptographic commitments for a given data buffer
/// by generating a Merkle tree and returning the root hash and the hash of the
/// specified data chunk combined with its index.
///
/// # Parameters
///
/// - `data`: The input data buffer represented as a type that can be referenced as a byte slice (`T: AsRef<[u8]>`).
/// - `index`: The index of the data chunk within the larger dataset. This value must be within the range of `0..num_leaves`.
/// - `num_leaves`: The total number of leaves (data chunks) in the Merkle tree. This value must be greater than zero.
///
/// # Returns
///
/// A tuple containing:
/// - The cryptographic hash of the Merkle tree root (`Hash`).
/// - The cryptographic hash of the data chunk combined with its index (`H(data | index)`).
///
/// # Panics
///
/// This function will panic if:
/// - The input data slice is empty.
/// - The `num_leaves` is zero.
/// - The `index` is out of bounds (i.e., greater than or equal to `num_leaves`).
///
/// # Notes
///
/// - This function requires the `Hasher` trait to be implemented for the chosen hash algorithm (`H`).
/// - The Merkle tree is constructed by hashing each data chunk combined with its index, and then concatenating
///   these hashes to form the tree structure.
pub fn calculate_event_commitment<H: Hasher, T: AsRef<[u8]>>(
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

/// Calculates the cryptographic commitment for a chat session.
///
/// This function computes the cryptographic commitment for a chat session by generating
/// a Merkle tree from the concatenated hashes of input and output prompts. The root hash
/// of the Merkle tree is returned as the commitment.
///
/// # Parameters
///
/// - `input_prompt_hashes`: A vector of hashes representing the input prompts.
/// - `output_prompt_hashes`: A vector of hashes representing the output prompts.
///
/// # Returns
///
/// A `Hash` representing the root of the Merkle tree, which serves as the cryptographic
/// commitment for the chat session.
///
/// # Panics
///
/// This function will panic if:
/// - The lengths of `input_prompt_hashes` and `output_prompt_hashes` are not equal.
///
/// # Notes
///
/// - The function pads the leaves to the next power of two with zero hashes to ensure
///   the Merkle tree is balanced.
/// - The `Hasher` trait must be implemented for the chosen hash algorithm (`H`).
///
/// # Examples
///
/// ```
/// use atoma_crypto::{calculate_chat_session_commitment, Blake2b, Hash};
///
/// let input_hashes: Vec<Hash> = vec![/* input hashes */];
/// let output_hashes: Vec<Hash> = vec![/* output hashes */];
///
/// let commitment = calculate_chat_session_commitment::<Blake2b<U32>>(input_hashes, output_hashes);
/// println!("{:x?}", commitment);
/// ```
pub fn calculate_chat_session_commitment<H: Hasher>(
    input_prompt_hashes: &[Hash],
    output_prompt_hashes: &[Hash],
) -> Hash {
    assert_eq!(input_prompt_hashes.len(), output_prompt_hashes.len());

    // Concatenate each input prompt hash with the corresponding output prompt hash
    let mut leaves: Vec<Hash> = input_prompt_hashes
        .iter()
        .zip(output_prompt_hashes.iter())
        .map(|(input_hash, output_hash)| {
            let concatenated = [input_hash.as_slice(), output_hash.as_slice()].concat();
            let hasher = H::init();
            hasher.hash(concatenated)
        })
        .collect();

    // Pad the leaves to the next power of two with zero hashes
    let next_power_of_two = leaves.len().next_power_of_two();
    while leaves.len() < next_power_of_two {
        leaves.push([0u8; HASH_SIZE]);
    }

    debug_assert!(leaves.len() == next_power_of_two);

    // Compute the Merkle tree root hash
    let root = if leaves.len() == 1 {
        leaves[0]
    } else {
        let mut current_level = leaves;
        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                // We have padded the leaves to the next power of two, so we can
                // safely assume that the length of the chunk is 2
                let concatenated = [chunk[0].as_slice(), chunk[1].as_slice()].concat();
                let hasher = H::init();
                let hashed = hasher.hash(concatenated);
                next_level.push(hashed);
            }
            current_level = next_level;
        }
        current_level[0]
    };

    root
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

        let mut rng = rand::thread_rng();
        let data = (0..SIZE).map(|_| rng.gen::<u8>()).collect::<Vec<_>>();

        // assert root size is correct
        let (root, leaf) =
            calculate_event_commitment::<Blake2b<U32>, _>(data.clone(), INDEX, NUM_CHUNKS);
        assert_eq!(root.len(), 32);
        assert_eq!(leaf.len(), 32);

        // assert that leaves are constructed correctly
        let mut leaves = Vec::with_capacity(NUM_CHUNKS);
        for i in 0..NUM_CHUNKS {
            let mut hasher = Blake2b::<U32>::new();
            hasher.update([data.as_slice(), i.to_le_bytes().as_slice()].concat());
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

    #[test]
    fn test_calculate_chat_session_commitment() {}
}
