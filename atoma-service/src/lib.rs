//! Authentication for Atoma nodes. It provides cryptographic verification of messages,
//! and supports multiple signature schemes (including ed25519, secp256k1, and secp256r1,
//! matching SUI's supported cryptography primitives).
pub(crate) mod components;
pub mod config;
pub(crate) mod handlers;
pub mod middleware;
pub mod proxy;
pub mod server;
pub mod streamer;
#[cfg(test)]
mod tests;
