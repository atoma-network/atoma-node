use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use p256::ecdsa::{Signature as P256Signature, VerifyingKey as P256PublicKey};
use secp256k1::{
    ecdsa::Signature as Secp256k1Signature, Message, PublicKey as Secp256k1PublicKey, Secp256k1,
};
use std::{
    fmt::{self, Display},
    str::FromStr,
};
use thiserror::Error;

/// Lengths of the public keys and signatures for the supported signature schemes.
const ED25519_PUBLIC_KEY_LENGTH: usize = 32;
const ED25519_SIGNATURE_LENGTH: usize = 64;
const SECP256K1_PUBLIC_KEY_LENGTH: usize = 33;
const SECP256K1_SIGNATURE_LENGTH: usize = 64;
const SHA256_HASH_LENGTH: usize = 32;

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

impl SignatureScheme {
    /// Verifies a signature using the specified signature scheme.
    ///
    /// # Arguments
    ///
    /// * `message` - The message to verify. For all schemes, this MUST be a 32-byte SHA256 hash of the original message.
    /// * `signature` - The signature bytes to verify.
    /// * `public_key` - The public key bytes used for verification.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the signature is valid, or an `AuthenticationError` if verification fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The message, signature, or public key have incorrect lengths for the chosen scheme.
    /// - The signature or public key are malformed or invalid.
    /// - The signature verification fails.
    ///
    /// # Warning
    ///
    /// The `message` parameter MUST be a 32-byte SHA256 hash of the original message for all signature schemes.
    /// Passing the raw message will result in incorrect verification or errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use sha2::{Sha256, Digest};
    /// # use your_crate::{SignatureScheme, AuthenticationError};
    ///
    /// # fn main() -> Result<(), AuthenticationError> {
    /// let scheme = SignatureScheme::Ed25519;
    /// let original_message = b"Hello, world!";
    /// let message = Sha256::digest(original_message);
    /// let signature = // ... obtain signature bytes ...
    /// let public_key = // ... obtain public key bytes ...
    ///
    /// scheme.verify(&message, &signature, &public_key)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn verify(
        &self,
        message: &[u8],
        signature: &[u8],
        public_key: &[u8],
    ) -> Result<(), AuthenticationError> {
        // Common check for message length
        Self::check_message_length(message)?;

        match self {
            SignatureScheme::EcdsaSecp256k1 => {
                Self::verify_secp256k1(message, signature, public_key)
            }
            SignatureScheme::EcdsaSecp256r1 => {
                Self::verify_secp256r1(message, signature, public_key)
            }
            SignatureScheme::Ed25519 => Self::verify_ed25519(message, signature, public_key),
        }
    }

    /// Checks if the provided message has the correct length for a SHA256 hash.
    ///
    /// # Arguments
    ///
    /// * `message` - The message to check, expected to be a SHA256 hash.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the message length is correct, or an `AuthenticationError` if not.
    ///
    /// # Errors
    ///
    /// This function will return an `AuthenticationError::MalformedMessage` if:
    /// - The message length is not equal to `SHA256_HASH_LENGTH` (32 bytes).
    fn check_message_length(message: &[u8]) -> Result<(), AuthenticationError> {
        if message.len() != SHA256_HASH_LENGTH {
            return Err(AuthenticationError::MalformedMessage(format!(
                "Message must be {} bytes long, as it is the sha256 digest of the message, instead got {} bytes",
                SHA256_HASH_LENGTH,
                message.len(),
            )));
        }
        Ok(())
    }

    /// Checks if the provided public key has the correct length for the given signature scheme.
    ///
    /// # Arguments
    ///
    /// * `public_key` - The public key bytes to check.
    /// * `expected_length` - The expected length of the public key in bytes.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the public key length is correct, or an `AuthenticationError` if not.
    ///
    /// # Errors
    ///
    /// This function will return an `AuthenticationError::MalformedPublicKey` if:
    /// - The public key length does not match the expected length for the signature scheme.
    fn check_key_length(
        public_key: &[u8],
        expected_length: usize,
    ) -> Result<(), AuthenticationError> {
        if public_key.len() != expected_length {
            return Err(AuthenticationError::MalformedPublicKey(format!(
                "Public key must be {} bytes long, instead got {} bytes",
                expected_length,
                public_key.len(),
            )));
        }
        Ok(())
    }

    /// Checks if the provided signature has the correct length for the given signature scheme.
    ///
    /// # Arguments
    ///
    /// * `signature` - The signature bytes to check.
    /// * `expected_length` - The expected length of the signature in bytes.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the signature length is correct, or an `AuthenticationError` if not.
    ///
    /// # Errors
    ///
    /// This function will return an `AuthenticationError::MalformedSignature` if:
    /// - The signature length does not match the expected length for the signature scheme.
    fn check_signature_length(
        signature: &[u8],
        expected_length: usize,
    ) -> Result<(), AuthenticationError> {
        if signature.len() != expected_length {
            return Err(AuthenticationError::MalformedSignature(format!(
                "Signature must be {} bytes long, instead got {} bytes",
                expected_length,
                signature.len(),
            )));
        }
        Ok(())
    }

    /// Verifies a secp256k1 ECDSA signature.
    ///
    /// # Arguments
    ///
    /// * `message` - The message to verify. This must be a 32-byte SHA256 hash of the original message.
    /// * `signature` - The signature bytes to verify.
    /// * `public_key` - The public key bytes used for verification.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the signature is valid, or an `AuthenticationError` if verification fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The public key length is incorrect (should be 33 bytes).
    /// - The public key, signature, or message is malformed or invalid.
    /// - The signature verification fails.
    ///
    /// # Panics
    ///
    /// This function will panic if the `public_key` or `signature` slices cannot be converted to fixed-size arrays.
    /// This should never happen due to the length checks performed at the beginning of the function.
    fn verify_secp256k1(
        message: &[u8],
        signature: &[u8],
        public_key: &[u8],
    ) -> Result<(), AuthenticationError> {
        Self::check_key_length(public_key, SECP256K1_PUBLIC_KEY_LENGTH)?;
        Self::check_signature_length(signature, SECP256K1_SIGNATURE_LENGTH)?;
        let public_key = Secp256k1PublicKey::from_slice(public_key)?;
        let message_bytes: [u8; 32] = message.try_into().unwrap();
        let message = Message::from_digest(message_bytes);
        let signature = Secp256k1Signature::from_compact(signature)?;
        let secp = Secp256k1::new();
        Ok(secp.verify_ecdsa(&message, &signature, &public_key)?)
    }

    /// Verifies a secp256r1 (P-256) ECDSA signature.
    ///
    /// # Arguments
    ///
    /// * `message` - The message to verify. This must be a 32-byte SHA256 hash of the original message.
    /// * `signature` - The signature bytes to verify.
    /// * `public_key` - The public key bytes used for verification.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the signature is valid, or an `AuthenticationError` if verification fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The public key length is incorrect (should be 32 bytes).
    /// - The signature length is incorrect (should be 64 bytes).
    /// - The public key, signature, or message is malformed or invalid.
    /// - The signature verification fails.
    fn verify_secp256r1(
        message: &[u8],
        signature: &[u8],
        public_key: &[u8],
    ) -> Result<(), AuthenticationError> {
        let public_key = P256PublicKey::from_sec1_bytes(public_key)?;
        let signature = P256Signature::from_der(signature)?;
        Ok(public_key.verify(message, &signature)?)
    }

    /// Verifies an Ed25519 signature.
    ///
    /// # Arguments
    ///
    /// * `message` - The message to verify. This must be a 32-byte SHA256 hash of the original message.
    /// * `signature` - The signature bytes to verify.
    /// * `public_key` - The public key bytes used for verification.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the signature is valid, or an `AuthenticationError` if verification fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The public key length is incorrect (should be 32 bytes).
    /// - The signature length is incorrect (should be 64 bytes).
    /// - The public key is invalid or cannot be parsed.
    /// - The signature verification fails.
    ///
    /// # Panics
    ///
    /// This function will panic if the `public_key` or `signature` slices cannot be converted to fixed-size arrays.
    /// This should never happen due to the length checks performed at the beginning of the function.
    fn verify_ed25519(
        message: &[u8],
        signature: &[u8],
        public_key: &[u8],
    ) -> Result<(), AuthenticationError> {
        Self::check_key_length(public_key, ED25519_PUBLIC_KEY_LENGTH)?;
        Self::check_signature_length(signature, ED25519_SIGNATURE_LENGTH)?;
        let public_key_bytes: [u8; 32] = public_key.try_into().unwrap();
        let public_key = VerifyingKey::from_bytes(&public_key_bytes)?;
        let signature_bytes: [u8; 64] = signature.try_into().unwrap();
        let signature = Signature::from_bytes(&signature_bytes);
        Ok(public_key.verify(message, &signature)?)
    }
}

#[derive(Debug, Error)]
pub enum AuthenticationError {
    #[error("Failed to verify Ed25519 signature: `{0}`")]
    FailedToVerifyEd25519Signature(#[from] ed25519_dalek::ed25519::Error),
    #[error("Failed to verify ECDSA signature: `{0}`")]
    FailedToVerifySecp256k1Signature(#[from] secp256k1::Error),
    #[error("Malformed public key: `{0}`")]
    MalformedPublicKey(String),
    #[error("Malformed signature: `{0}`")]
    MalformedSignature(String),
    #[error("Malformed message: `{0}`")]
    MalformedMessage(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use p256::ecdsa::{signature::Signer, SigningKey as P256SigningKey};
    use rand::{rngs::OsRng, RngCore};
    use secp256k1::{Message, SecretKey};
    use sha2::{Digest, Sha256};

    fn generate_test_message() -> [u8; 32] {
        let message = b"Hello, world!";
        Sha256::digest(message).into()
    }

    #[test]
    fn test_ed25519_verification() {
        let message = generate_test_message();

        let mut random_bytes = [0u8; 32];
        OsRng.fill_bytes(&mut random_bytes);
        let signing_key = SigningKey::from_bytes(&random_bytes);
        let verifying_key = signing_key.verifying_key();

        let signature = signing_key.sign(&message);

        assert!(SignatureScheme::Ed25519
            .verify(
                &message,
                signature.to_vec().as_ref(),
                verifying_key.as_bytes()
            )
            .is_ok());

        // Test with incorrect message
        let wrong_message = [0u8; 32];
        assert!(SignatureScheme::Ed25519
            .verify(
                &wrong_message,
                signature.to_vec().as_ref(),
                verifying_key.as_bytes()
            )
            .is_err());
    }

    #[test]
    fn test_secp256k1_verification() {
        let message = generate_test_message();
        let secp = Secp256k1::new();
        let mut random_bytes = [0u8; 32];
        OsRng.fill_bytes(&mut random_bytes);
        let secret_key =
            SecretKey::from_byte_array(&random_bytes).expect("32 bytes, within curve order");
        let public_key = Secp256k1PublicKey::from_secret_key(&secp, &secret_key);
        let message = Message::from_digest(message);
        let signature = secp.sign_ecdsa(&message, &secret_key);

        assert!(SignatureScheme::EcdsaSecp256k1
            .verify(
                message.as_ref(),
                &signature.serialize_compact(),
                &public_key.serialize()
            )
            .is_ok());

        // Test with incorrect message
        let wrong_message = Message::from_digest([0u8; 32]);
        assert!(SignatureScheme::EcdsaSecp256k1
            .verify(
                wrong_message.as_ref(),
                &signature.serialize_compact(),
                &public_key.serialize()
            )
            .is_err());
    }

    #[test]
    fn test_secp256r1_verification() {
        let message = generate_test_message();
        let secret_key = P256SigningKey::random(&mut OsRng);
        let public_key = secret_key.verifying_key();
        let signature: P256Signature = secret_key.sign(&message);

        assert!(SignatureScheme::EcdsaSecp256r1
            .verify(
                &message,
                signature.to_der().as_ref(),
                &public_key.to_sec1_bytes()
            )
            .is_ok());

        // Test with incorrect message
        let wrong_message = [0u8; 32];
        assert!(SignatureScheme::EcdsaSecp256r1
            .verify(
                &wrong_message,
                signature.to_der().as_ref(),
                &public_key.to_sec1_bytes()
            )
            .is_err());
    }

    #[test]
    fn test_invalid_message_length() {
        let invalid_message = [0u8; 16]; // Too short
        let dummy_signature = [0u8; 64];
        let dummy_public_key = [0u8; 32];

        for scheme in [
            SignatureScheme::Ed25519,
            SignatureScheme::EcdsaSecp256k1,
            SignatureScheme::EcdsaSecp256r1,
        ] {
            assert!(matches!(
                scheme.verify(&invalid_message, &dummy_signature, &dummy_public_key),
                Err(AuthenticationError::MalformedMessage(_))
            ));
        }
    }

    #[test]
    fn test_invalid_signature_length() {
        let message = [0u8; 32];
        let invalid_signature = [0u8; 32]; // Too short

        for scheme in [SignatureScheme::Ed25519, SignatureScheme::EcdsaSecp256k1] {
            if scheme == SignatureScheme::Ed25519 {
                let dummy_public_key = [0u8; ED25519_PUBLIC_KEY_LENGTH];
                assert!(matches!(
                    scheme.verify(&message, &invalid_signature, &dummy_public_key),
                    Err(AuthenticationError::MalformedSignature(_))
                ));
            } else {
                let dummy_public_key = [0u8; SECP256K1_PUBLIC_KEY_LENGTH];
                assert!(matches!(
                    scheme.verify(&message, &invalid_signature, &dummy_public_key),
                    Err(AuthenticationError::MalformedSignature(_))
                ));
            }
        }

        const SECP256R1_PUBLIC_KEY_LENGTH: usize = 64;
        let scheme = SignatureScheme::EcdsaSecp256r1;
        let dummy_public_key = [0u8; SECP256R1_PUBLIC_KEY_LENGTH];
        assert!(matches!(
            scheme.verify(&message, &invalid_signature, &dummy_public_key),
            Err(AuthenticationError::FailedToVerifyEd25519Signature(_))
        ));
    }

    #[test]
    fn test_invalid_public_key_length() {
        let message = [0u8; 32];
        let dummy_signature = [0u8; 64];
        let invalid_public_key = [0u8; 16]; // Too short

        for scheme in [SignatureScheme::Ed25519, SignatureScheme::EcdsaSecp256k1] {
            assert!(matches!(
                scheme.verify(&message, &dummy_signature, &invalid_public_key),
                Err(AuthenticationError::MalformedPublicKey(_))
            ));
        }

        let scheme = SignatureScheme::EcdsaSecp256r1;
        assert!(matches!(
            scheme.verify(&message, &dummy_signature, &invalid_public_key),
            Err(AuthenticationError::FailedToVerifyEd25519Signature(_))
        ));
    }
}
