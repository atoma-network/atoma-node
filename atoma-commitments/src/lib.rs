use ed25519_consensus::{Signature, SigningKey as PrivateKey, VerificationKey as PublicKey};
use keccak_hash::keccak;
use thiserror::Error;

use crate::merkle_tree::compute_merkle_root;

mod merkle_tree;

pub struct AtomaCommitment {
    private_key: PrivateKey,
    public_key: PublicKey,
}

impl AtomaCommitment {
    pub fn new(private_key: PrivateKey) -> Self {
        Self {
            public_key: private_key.verification_key(),
            private_key,
        }
    }

    pub fn calculate_commitment<T: AsRef<[u8]>>(
        &self,
        mut data: T,
        index: usize,
        num_leaves: usize,
    ) -> Signature {
        let data = data.as_ref();
        let chunk_size = data.len() / num_leaves;
        let padded_num_leaves = 2_usize.pow(num_leaves.ilog2() + 1);
        let pad = vec![&[0_u8; 1]; padded_num_leaves - num_leaves];
        let mut chunks = data.chunks(chunk_size);
        let leaves = chunks
            .map(|chunk| {
                let hash = keccak(chunk);
                hash.as_bytes().to_vec()
            })
            .chain((0..(padded_num_leaves - num_leaves)).map(|i| {
                let hash = keccak(&i.to_be_bytes());
                hash.as_bytes().to_vec()
            })).collect();
        let merkle_root = compute_merkle_root(leaves);
        self.private_key.sign(&merkle_root)
    }

    pub fn run(self) -> Result<(), AtomaCommitmentError> {
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum AtomaCommitmentError {}
