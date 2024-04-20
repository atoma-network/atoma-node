use keccak_hash::keccak;

use super::Hasher;

pub struct Keccak256 {}

impl Hasher for Keccak256 { 
    fn hash(data: &[u8]) -> [u8; super::HASH_SIZE] {
        keccak(data).to_fixed_bytes()
    }
}