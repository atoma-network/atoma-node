use crypto::Commitment;
use ed25519_consensus::SigningKey as PrivateKey;
use rs_merkle::{Hasher, MerkleProof, MerkleTree};

mod crypto;

pub struct AtomaCommitment {
    private_key: PrivateKey,
}

impl AtomaCommitment {
    pub fn new(private_key: PrivateKey) -> Self {
        Self { private_key }
    }

    pub fn calculate_commitment<H: Hasher, T: AsRef<[u8]>>(
        &self,
        data: T,
        index: usize,
        num_leaves: usize,
    ) -> (Commitment, MerkleProof<H>) {
        let data = data.as_ref();
        let chunk_size = data.len() / num_leaves;

        let chunks = data
            .chunks(chunk_size)
            .map(|buf| H::hash(buf))
            .collect::<Vec<_>>();

        assert!(chunks.len() > 0);

        let merkle_tree = MerkleTree::<H>::from_leaves(&chunks);
        let merkle_proof = merkle_tree.proof(&[index]);
        let commitment = Commitment::new(
            self.private_key
                .sign(&merkle_tree.root().unwrap().into())
                .to_bytes(),
        );

        (commitment, merkle_proof)
    }
}
