use std::marker::PhantomData;

use crate::crypto::{Hash, Hasher};
use rayon::prelude::*;
use thiserror::Error;

/// Our implementation is inspired by the one of Plonky2:
/// see https://github.com/mir-protocol/plonky2/blob/main/plonky2/src/hash/merkle_tree.rs#L39.
pub struct MerkleTree<H: Hasher> {
    pub(crate) digests: Vec<Hash>,
    pub(crate) root: Hash,
    _phantom_data: PhantomData<H>,
}

impl<H: Hasher> MerkleTree<H> {
    /// Method `new`:
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
        let leaves = data_chunks.map(|chunk| H::hash(chunk)).collect::<Vec<_>>();

        let num_leaves = leaves.len();
        let pad_num_leaves = 2_usize.pow(num_leaves.ilog2() + 1);
        let merkle_tree_height = pad_num_leaves.ilog2();

        let leaves = leaves
            .into_iter()
            .chain((num_leaves..pad_num_leaves).map(|i| H::hash(i.to_be_bytes().as_slice())));

        let mut digests = Vec::with_capacity(2 * pad_num_leaves - 1);
        digests.extend(leaves);

        let mut current_tree_height_index = 0;
        let mut i = 0;
        for height in 0..merkle_tree_height {
            while i < current_tree_height_index + (1 << (merkle_tree_height - height)) {
                let hash = H::hash(&[digests[i as usize], digests[i as usize + 1]].concat());
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

    pub fn path(&self, index: usize) -> Vec<Hash> { 
        vec![]
    }
}
