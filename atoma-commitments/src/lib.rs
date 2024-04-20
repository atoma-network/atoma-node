use crypto::Commitment;
use ed25519_consensus::{SigningKey as PrivateKey, VerificationKey as PublicKey};
use thiserror::Error;

use crate::{crypto::Hasher, merkle_tree::MerkleTree};

mod crypto;
mod merkle_tree;

pub struct AtomaCommitment {
    private_key: PrivateKey,
}

impl AtomaCommitment {
    pub fn new(private_key: PrivateKey) -> Self {
        Self {
            private_key,
        }
    }

    pub fn calculate_commitment<H: Hasher, T: AsRef<[u8]>>(
        &self,
        data: T,
        index: usize,
        num_leaves: usize,
    ) -> Commitment {
        let data = data.as_ref();
        let chunk_size = data.len() / num_leaves;
        let padded_num_leaves = 2_usize.pow(num_leaves.ilog2() + 1);

        let pad = vec![&[0_u8; 1]; padded_num_leaves - num_leaves];
        let chunks = data.chunks(chunk_size);

        let merkle_tree = MerkleTree::<H>::create(chunks);
        Commitment::new(self.private_key.sign(&merkle_tree.root()).to_bytes())
    }
}

#[derive(Debug, Error)]
pub enum AtomaCommitmentError {}
