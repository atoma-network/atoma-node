use fastcrypto::{
    ed25519::{Ed25519PublicKey, Ed25519Signature},
    secp256k1::{Secp256k1PublicKey, Secp256k1Signature},
    secp256r1::{Secp256r1PublicKey, Secp256r1Signature},
    traits::{ToFromBytes as FastCryptoToFromBytes, VerifyingKey},
};
use flume::Sender;
use sui_sdk::types::{
    base_types::SuiAddress,
    crypto::{PublicKey, Signature, SignatureScheme, SuiSignature, ToFromBytes},
};
use tokio::sync::oneshot;
use tracing::{error, instrument};
use url::Url;

use crate::{
    errors::AtomaP2pNodeError, service::StateManagerEvent, types::NodeMessage, AtomaP2pEvent,
};

/// The threshold for considering a timestamp as expired
const EXPIRED_TIMESTAMP_THRESHOLD: u64 = 10 * 60; // 10 minutes

/// Validates a UsageMetrics message by checking the URL format and timestamp freshness.
///
/// This function performs two key validations:
/// 1. Ensures the node_public_url is a valid URL format
/// 2. Verifies the timestamp is within an acceptable time window relative to the current time
///
/// # Arguments
///
/// * `response` - The `UsageMetrics` containing the node_public_url and timestamp to validate
///
/// # Returns
///
/// * `Ok(())` - If all validation checks pass
/// * `Err(AtomaP2pNodeError)` - If any validation fails
///
/// # Errors
///
/// Returns `AtomaP2pNodeError::InvalidPublicAddressError` when:
/// * The URL is invalid or malformed
/// * The timestamp is older than `EXPIRED_TIMESTAMP_THRESHOLD` (10 minutes)
/// * The timestamp is in the future
///
/// # Example
///
/// ```rust,ignore
/// let metrics = UsageMetrics {
///     node_public_url: "https://example.com".to_string(),
///     timestamp: std::time::Instant::now().elapsed().as_secs(),
///     signature: vec![],
///     node_small_id: 1,
/// };
///
/// match validate_public_address_message(&metrics) {
///     Ok(()) => println!("UsageMetrics is valid"),
///     Err(e) => println!("Invalid UsageMetrics: {}", e),
/// }
/// ```
///
/// # Security Considerations
///
/// This validation helps prevent:
/// * Invalid or malicious URLs from being propagated
/// * Replay attacks by enforcing timestamp freshness
/// * Future timestamp manipulation attempts
#[instrument(level = "debug", skip_all)]
pub fn validate_node_message_country_url_timestamp(
    node_message: &NodeMessage,
) -> Result<(), AtomaP2pNodeError> {
    let now = std::time::Instant::now().elapsed().as_secs();

    let country = node_message.node_metadata.country.as_str();
    validate_country_code(country)?;

    // Check if the URL is valid
    let _usage_metrics_url =
        Url::parse(&node_message.node_metadata.node_public_url).map_err(|e| {
            error!(
                target = "atoma-p2p",
                event = "invalid_url_format",
                error = %e,
                "Invalid URL format, received address: {}",
                node_message.node_metadata.node_public_url
            );
            AtomaP2pNodeError::UrlParseError(e)
        })?;

    // Check if the timestamp is within a reasonable time frame
    if now < node_message.node_metadata.timestamp
        || now > node_message.node_metadata.timestamp + EXPIRED_TIMESTAMP_THRESHOLD
    {
        error!(
            target = "atoma-p2p",
            event = "invalid_timestamp",
            "Timestamp is invalid, timestamp: {}, now: {}",
            node_message.node_metadata.timestamp,
            now
        );
        return Err(AtomaP2pNodeError::InvalidPublicAddressError(
            "Timestamp is too far in the past".to_string(),
        ));
    }

    Ok(())
}

/// Custom validation function for ISO 3166-1 alpha-2 country codes
fn validate_country_code(code: &str) -> Result<(), AtomaP2pNodeError> {
    isocountry::CountryCode::for_alpha2(code).map_err(|_| {
        AtomaP2pNodeError::InvalidCountryCodeError("Country code is invalid.".to_string())
    })?;
    Ok(())
}

/// Verifies a cryptographic signature against a message hash using the signature scheme embedded in the signature.
///
/// This function supports multiple signature schemes:
/// - ED25519
/// - Secp256k1
/// - Secp256r1
///
/// # Arguments
///
/// * `signature_bytes` - Raw bytes of the signature. This should include both the signature data and metadata
///   that allows extracting the public key and signature scheme.
/// * `body_hash` - A 32-byte array containing the hash of the message that was signed. This is typically
///   produced using blake3 or another cryptographic hash function.
///
/// # Returns
///
/// * `Ok(())` - If the signature is valid for the given message hash
/// * `Err(AtomaP2pNodeError)` - If any step of the verification process fails
///
/// # Errors
///
/// This function will return an error in the following situations:
/// * `SignatureParseError` - If the signature bytes cannot be parsed into a valid signature structure
///   or if the public key cannot be extracted from the signature
/// * `SignatureVerificationError` - If the signature verification fails (indicating the signature is invalid
///   for the given message)
/// * `SignatureParseError` - If the signature uses an unsupported signature scheme
///
/// # Example
///
/// ```rust,ignore
/// use your_crate::utils::verify_signature;
///
/// let message = b"Hello, world!";
/// let message_hash = blake3::hash(message).as_bytes();
/// let signature_bytes = // ... obtained from somewhere ...
///
/// match verify_signature(&signature_bytes, &message_hash) {
///     Ok(()) => println!("Signature is valid!"),
///     Err(e) => println!("Signature verification failed: {}", e),
/// }
/// ```
#[instrument(level = "trace", skip_all)]
pub fn verify_signature(
    signature_bytes: &[u8],
    body_hash: &[u8; 32],
) -> Result<(), AtomaP2pNodeError> {
    let signature = Signature::from_bytes(signature_bytes).map_err(|e| {
        error!("Failed to parse signature");
        AtomaP2pNodeError::SignatureParseError(e.to_string())
    })?;
    let public_key_bytes = signature.public_key_bytes();
    let signature_scheme = signature.scheme();
    let public_key =
        PublicKey::try_from_bytes(signature_scheme, public_key_bytes).map_err(|e| {
            error!("Failed to extract public key from bytes, with error: {e}");
            AtomaP2pNodeError::SignatureParseError(e.to_string())
        })?;
    let signature_bytes = signature.signature_bytes();

    match signature_scheme {
        SignatureScheme::ED25519 => {
            let public_key = Ed25519PublicKey::from_bytes(public_key.as_ref()).unwrap();
            let signature = Ed25519Signature::from_bytes(signature_bytes).unwrap();
            public_key.verify(body_hash, &signature).map_err(|e| {
                error!("Failed to verify signature");
                AtomaP2pNodeError::SignatureVerificationError(e.to_string())
            })?;
        }
        SignatureScheme::Secp256k1 => {
            let public_key = Secp256k1PublicKey::from_bytes(public_key.as_ref()).unwrap();
            let signature = Secp256k1Signature::from_bytes(signature_bytes).unwrap();
            public_key.verify(body_hash, &signature).map_err(|e| {
                error!("Failed to verify signature");
                AtomaP2pNodeError::SignatureVerificationError(e.to_string())
            })?;
        }
        SignatureScheme::Secp256r1 => {
            let public_key = Secp256r1PublicKey::from_bytes(public_key.as_ref()).unwrap();
            let signature = Secp256r1Signature::from_bytes(signature_bytes).unwrap();
            public_key.verify(body_hash, &signature).map_err(|e| {
                error!("Failed to verify signature");
                AtomaP2pNodeError::SignatureVerificationError(e.to_string())
            })?;
        }
        e => {
            error!("Currently unsupported signature scheme, error: {e}");
            return Err(AtomaP2pNodeError::SignatureParseError(e.to_string()));
        }
    }
    Ok(())
}

/// Verifies that a node owns its claimed small ID by checking the signature and querying the state manager.
///
/// This function performs the following steps:
/// 1. Parses the signature to extract the public key
/// 2. Derives the Sui address from the public key
/// 3. Sends a verification request to the state manager
/// 4. Waits for the state manager's response
///
/// # Arguments
///
/// * `node_small_id` - The small ID claimed by the node
/// * `signature` - The signature to verify, containing the public key
/// * `state_manager_sender` - Channel to send verification requests to the state manager
///
/// # Returns
///
/// * `Ok(())` - If the node owns the small ID
/// * `Err(AtomaP2pNodeError)` - If verification fails at any step
///
/// # Errors
///
/// This function can return the following errors:
/// * `SignatureParseError` - If the signature cannot be parsed
/// * `StateManagerError` - If the verification request cannot be sent to the state manager
/// * `NodeSmallIdOwnershipVerificationError` - If the state manager reports the node does not own the ID
///
/// # Example
///
/// ```rust,ignore
/// let signature = // ... obtained from node ...;
/// let result = verify_node_small_id_ownership(
///     42,
///     &signature,
///     state_manager_sender
/// ).await;
///
/// match result {
///     Ok(()) => println!("Verification succeeded"),
///     Err(e) => println!("Verification failed: {}", e),
/// }
/// ```
///
/// # Security Considerations
///
/// - Uses cryptographic signatures to prevent impersonation
/// - Relies on the state manager's authoritative record of node ownership
/// - Protects against node ID spoofing attacks
#[instrument(level = "debug", skip_all)]
pub async fn verify_node_small_id_ownership(
    node_small_id: u64,
    signature: &[u8],
    state_manager_sender: &Sender<StateManagerEvent>,
) -> Result<(), AtomaP2pNodeError> {
    let signature = Signature::from_bytes(signature).map_err(|e| {
        error!("Failed to parse signature");
        AtomaP2pNodeError::SignatureParseError(e.to_string())
    })?;
    let public_key_bytes = signature.public_key_bytes();
    let public_key =
        PublicKey::try_from_bytes(signature.scheme(), public_key_bytes).map_err(|e| {
            error!("Failed to extract public key from bytes, with error: {e}");
            AtomaP2pNodeError::SignatureParseError(e.to_string())
        })?;
    let sui_address = SuiAddress::from(&public_key);
    let (sender, receiver) = oneshot::channel();
    if let Err(e) = state_manager_sender.send((
        AtomaP2pEvent::VerifyNodeSmallIdOwnership {
            node_small_id,
            sui_address: sui_address.to_string(),
        },
        Some(sender),
    )) {
        error!(
            target = "atoma-p2p",
            event = "failed_to_send_event_to_state_manager",
            error = %e,
            "Failed to send event to state manager"
        );
        return Err(AtomaP2pNodeError::StateManagerError(e));
    }
    match receiver.await {
        Ok(result) => {
            if result {
                Ok(())
            } else {
                Err(AtomaP2pNodeError::NodeSmallIdOwnershipVerificationError(
                    "Node small ID ownership verification failed".to_string(),
                ))
            }
        }
        Err(e) => {
            error!("Failed to receive result from state manager, with error: {e}");
            Err(AtomaP2pNodeError::NodeSmallIdOwnershipVerificationError(
                e.to_string(),
            ))
        }
    }
}

/// Validates the messages received from the P2P network
///
/// This function validates the sharing node public addresses messages received from the P2P network by checking the signature and the node small ID ownership.
///
/// # Arguments
///
/// * `response` - The message received from the P2P network
/// * `state_manager_sender` - The sender of the state manager
///
/// # Returns
///
/// Returns `Ok(())` if the message is valid, or a `AtomaP2pNodeError` if any step fails.
#[instrument(level = "debug", skip_all)]
pub async fn validate_signed_node_message(
    node_message: &NodeMessage,
    node_message_hash: &[u8; 32],
    signature: &[u8],
    state_manager_sender: &Sender<StateManagerEvent>,
) -> Result<(), AtomaP2pNodeError> {
    // Validate the message's node public URL and timestamp
    validate_node_message_country_url_timestamp(node_message)?;
    // Verify the signature of the message
    verify_signature(signature, node_message_hash)?;
    // Verify the node small ID ownership
    verify_node_small_id_ownership(
        node_message.node_metadata.node_small_id,
        signature,
        state_manager_sender,
    )
    .await?;
    Ok(())
}
