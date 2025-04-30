//! Authentication for Atoma nodes. It provides cryptographic verification of messages,
//! and supports multiple signature schemes (including ed25519, secp256k1, and secp256r1,
//! matching SUI's supported cryptography primitives).

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::uninlined_format_args)]

pub(crate) mod components;
pub mod config;
pub mod error;
pub mod handlers;
pub mod middleware;
pub mod server;
pub mod streamer;
#[cfg(test)]
mod tests;
pub mod types;
