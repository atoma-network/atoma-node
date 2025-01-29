use blake2::{
    digest::generic_array::{typenum::U32, GenericArray},
    Blake2b, Digest,
};

/// Computes the `BLAKE2b` hash of the provided data
///
/// # Arguments
/// * `slice` - A byte slice containing the data to be hashed
/// # Returns
/// The 32-byte `BLAKE2b` hash
#[must_use]
pub fn blake2b_hash(slice: &[u8]) -> GenericArray<u8, U32> {
    let mut hasher = Blake2b::new();
    hasher.update(slice);
    hasher.finalize()
}
