use keccak_hash::keccak;

use super::AtomaHasher;

pub struct Keccak256 {}

impl AtomaHasher for Keccak256 {
    fn hash_data(data: &[u8]) -> [u8; super::HASH_SIZE] {
        keccak(data).to_fixed_bytes()
    }
}
