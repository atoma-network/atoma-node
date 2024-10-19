use std::{fmt::{self, Display}, str::FromStr};

/// Represents the supported signature schemes for authentication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureScheme {
    /// ECDSA using the secp256k1 curve
    EcdsaSecp256k1,
    /// ECDSA using the secp256r1 curve (also known as P-256)
    EcdsaSecp256r1,
    /// EdDSA using Curve25519
    Ed25519,
}


impl Display for SignatureScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignatureScheme::EcdsaSecp256k1 => write!(f, "ecdsa_secp256k1"),
            SignatureScheme::EcdsaSecp256r1 => write!(f, "ecdsa_secp256r1"),
            SignatureScheme::Ed25519 => write!(f, "ed25519"),
        }
    }
}

impl FromStr for SignatureScheme {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ecdsa_secp256k1" => Ok(SignatureScheme::EcdsaSecp256k1),
            "ecdsa_secp256r1" => Ok(SignatureScheme::EcdsaSecp256r1),
            "ed25519" => Ok(SignatureScheme::Ed25519),
            _ => Err("Invalid signature scheme"),
        }
    }
}
