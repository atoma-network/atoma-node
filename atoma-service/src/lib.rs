//! Authentication for Atoma nodes. It provides cryptographic verification of messages,
//! and supports multiple signature schemes (including ed25519, secp256k1, and secp256r1,
//! matching SUI's supported cryptography primitives).
pub mod authentication;
pub mod middleware;
pub mod server;
#[cfg(test)]
mod tests;
pub mod types;
