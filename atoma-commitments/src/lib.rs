use crypto::Commitment;
use ed25519_consensus::SigningKey as PrivateKey;
use merkle_tree::MerklePath;
use thiserror::Error;

use crate::{crypto::AtomaHasher, merkle_tree::MerkleTree};

mod crypto;
mod merkle_tree;

pub struct AtomaCommitment {
    private_key: PrivateKey,
}

impl AtomaCommitment {
    pub fn new(private_key: PrivateKey) -> Self {
        Self { private_key }
    }

    pub fn calculate_commitment<H: AtomaHasher, T: AsRef<[u8]>>(
        &self,
        data: T,
        index: usize,
        num_leaves: usize,
    ) -> (Commitment, MerklePath) {
        let data = data.as_ref();
        let chunk_size = data.len() / num_leaves;

        let chunks = data.chunks(chunk_size);

        let merkle_tree = MerkleTree::<H>::create(chunks);
        let merkle_path = merkle_tree.path(index);
        let commitment = Commitment::new(self.private_key.sign(&merkle_tree.root()).to_bytes());

        (commitment, merkle_path)
    }
}

#[derive(Debug, Error)]
pub enum AtomaCommitmentError {}
