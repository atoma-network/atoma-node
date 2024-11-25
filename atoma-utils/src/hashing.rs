use blake2::{
    digest::generic_array::{typenum::U32, GenericArray},
    Blake2b, Digest,
};

/// Computes a Blake2b hash of the input data
///
/// # Arguments
/// * `slice` - A byte slice containing the data to be hashed
///
/// # Returns
/// A 32-byte [`GenericArray`] containing the computed hash
///
/// # Example
/// ```rust,ignore
/// use atoma_utils::hashing::blake2b_hash;
///
/// let data = b"Hello, world!";
/// let hash = blake2b_hash(data);
/// ```
pub fn blake2b_hash(slice: &[u8]) -> GenericArray<u8, U32> {
    let mut hasher = Blake2b::new();
    hasher.update(slice);
    hasher.finalize()
}
