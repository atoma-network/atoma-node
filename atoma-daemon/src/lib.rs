#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_sign_loss)]

pub(crate) mod components;
pub mod config;
pub(crate) mod handlers;
pub mod server;
pub mod telemetry;
pub mod types;

pub use crate::{config::AtomaDaemonConfig, server::DaemonState};

use blake2::{Blake2b, Digest};
use rs_merkle::Hasher;

/// A hasher implementation using the Blake2b algorithm.
///
/// This struct implements the `Hasher` trait, allowing it to be used
/// for creating Merkle trees with the `rs_merkle` crate. The Blake2b
/// algorithm is a cryptographic hash function that provides a high
/// level of security and is suitable for use in various applications
/// requiring data integrity and authenticity.
#[derive(Clone)]
pub struct Blake2bHasher;

impl Hasher for Blake2bHasher {
    type Hash = [u8; 32];

    fn hash(data: &[u8]) -> Self::Hash {
        let mut hasher = Blake2b::new();
        hasher.update(data);
        hasher.finalize().into()
    }
}
