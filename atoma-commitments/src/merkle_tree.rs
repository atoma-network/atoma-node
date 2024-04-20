use std::marker::PhantomData;

use crate::crypto::{AtomaHasher, Hash};
use rayon::prelude::*;
use thiserror::Error;

/// Our implementation is inspired by the one of Plonky2:
/// see https://github.com/mir-protocol/plonky2/blob/main/plonky2/src/hash/merkle_tree.rs#L39.
pub struct MerkleTree<H: AtomaHasher> {
    pub(crate) digests: Vec<Hash>,
    pub(crate) root: Hash,
    _phantom_data: PhantomData<H>,
}

pub struct MerklePath {
    path: Vec<Hash>,
}

impl<H: AtomaHasher> MerkleTree<H> {
    /// Method `create`:
    ///
    ///     Creates a new instance of the MerkleTree struct.
    ///
    /// Arguments:
    ///
    ///     data: A vector containing vectors of elements of `Goldilocks` field type. Each inner vector represents the data for a leaf node of the Merkle tree.
    ///
    /// Returns:
    ///
    ///     Returns a MerkleTree instance representing the constructed Merkle tree.
    ///
    /// Panics:
    ///
    ///     Panics if the length of data is not a power of two or is less than 2.
    ///
    /// Description:
    ///
    ///     This method constructs a Merkle tree based on the input data. The number of leaves in the tree should be a power of two.
    ///     It iterates over the provided data to compute the hashes of the leaf nodes using the PoseidonHash::hash_or_noop function.
    ///     Then, it iterates over the heights of the tree, combining pairs of digests to compute intermediate hash nodes until the root hash is computed.
    pub fn create<'a>(data_chunks: impl Iterator<Item = &'a [u8]>) -> Self {
        let leaves = data_chunks
            .map(|chunk| H::hash_data(chunk))
            .collect::<Vec<_>>();

        let num_leaves = leaves.len();
        let pad_num_leaves = if num_leaves.is_power_of_two() {
            num_leaves
        } else {
            2_usize.pow(num_leaves.ilog2() + 1)
        };
        let merkle_tree_height = pad_num_leaves.ilog2();

        let leaves = leaves
            .into_iter()
            .chain((num_leaves..pad_num_leaves).map(|i| H::hash_data(i.to_be_bytes().as_slice())));

        let mut digests = Vec::with_capacity(2 * pad_num_leaves - 1);
        digests.extend(leaves);

        let mut current_tree_height_index = 0;
        let mut i = 0;
        for height in 0..merkle_tree_height {
            while i < current_tree_height_index + (1 << (merkle_tree_height - height)) {
                let hash = H::hash_data(&[digests[i as usize], digests[i as usize + 1]].concat());
                println!("{:?}", hash);
                digests.push(hash);
                i += 2;
            }
            current_tree_height_index += 1 << (merkle_tree_height - height);
        }

        // we assume that the number of leaves is > 1, so we should have a proper root
        let root = *digests.last().unwrap();

        Self {
            digests,
            root,
            _phantom_data: PhantomData,
        }
    }

    pub fn root(&self) -> Hash {
        self.root
    }

    pub fn height(&self) -> usize {
        (self.digests.len() + 1).ilog2() as usize - 1
    }

    pub fn num_leaves(&self) -> usize {
        (self.digests.len() + 1) / 2
    }

    pub fn path(&self, mut index: usize) -> MerklePath {
        assert!(index < self.num_leaves());
        let mut path = Vec::with_capacity(self.height());

        let mut height = 0;
        let mut height_offset = 0;
        let merkle_height = self.height();

        while height < merkle_height {
            if index % 2 == 0 {
                path.push(self.digests[index + 1]);
            } else {
                path.push(self.digests[index - 1]);
            }

            height_offset += 1 << (merkle_height - height);
            height += 1;
            index = height_offset + index / 2;
        }

        MerklePath { path }
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::OsRng, RngCore};
    use rs_merkle::{
        algorithms::{self, Sha256},
        Hasher,
    };

    use crate::crypto::{AtomaHasher, Keccak256, HASH_SIZE};

    use super::*;

    impl AtomaHasher for algorithms::Sha256 {
        fn hash_data(data: &[u8]) -> [u8; HASH_SIZE] {
            Sha256::hash(data)
        }
    }

    #[test]
    fn test_merkle_tree_create() {
        let leaf_1 = [0_u8, 1];
        let leaf_2 = [2_u8, 3];

        let data = [leaf_1.as_slice(), leaf_2.as_slice()];

        let merkle_tree = MerkleTree::<Keccak256>::create(data.iter().map(|x| *x));
        let root = merkle_tree.root();

        assert_eq!(
            root,
            Keccak256::hash_data(
                &[Keccak256::hash_data(&leaf_1), Keccak256::hash_data(&leaf_2)].concat()
            )
        );

        let mut rng = OsRng;
        let leaves = (0..256).map(|_| {
            let mut dest = [0u8; 32];
            rng.fill_bytes(&mut dest);
            dest
        });
        let merkle_tree = MerkleTree::<Sha256>::create(leaves.map(|x| x.as_slice()));
        let should_be_merkle_tree =
            rs_merkle::MerkleTree::<Sha256>::from_leaves(&leaves.collect::<Vec<_>>());
        assert_eq!(merkle_tree.root(), should_be_merkle_tree.root().unwrap())
    }
}
