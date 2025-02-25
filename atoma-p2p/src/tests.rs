use crate::{
    broadcast_metrics::NodeMetrics,
    errors::AtomaP2pNodeError,
    types::{NodeMessage, NodeP2pMetadata, SignedNodeMessage},
    utils::validate_signed_node_message,
    AtomaP2pEvent,
};

use bytes::Bytes;
use flume::unbounded;
use sui_keys::keystore::{AccountKeystore, InMemKeystore};
use tokio::sync::oneshot;
use tracing::error;

/// Creates a test keystore
///
/// # Returns
///
/// Returns an `InMemKeystore` struct
#[must_use]
pub fn create_test_keystore() -> InMemKeystore {
    // Create a new in-memory keystore
    InMemKeystore::new_insecure_for_tests(1)
}

/// Creates a test usage metrics message
///
/// # Arguments
///
/// * `keystore` - The keystore to use for signing the message
/// * `timestamp_offset` - The offset to add to the current timestamp
/// * `node_small_id` - The small ID of the node
///
/// # Returns
///
/// Returns a `SignedNodeMessage` struct
///
/// # Panics
///
/// Panics if the usage metrics cannot be serialized
#[must_use]
pub fn create_test_signed_node_message(
    keystore: &InMemKeystore,
    timestamp_offset: i64,
    node_small_id: u64,
) -> SignedNodeMessage {
    let now = std::time::Instant::now().elapsed().as_secs();
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_wrap)]
    let timestamp = (now as i64 + timestamp_offset) as u64;

    let node_public_url = "https://test.example.com".to_string();
    let country = "US".to_string();

    let node_message = NodeMessage {
        node_metadata: NodeP2pMetadata {
            node_public_url,
            node_small_id,
            country,
            timestamp,
        },
        node_metrics: NodeMetrics::default(),
    };

    let mut node_message_bytes = Vec::new();
    ciborium::into_writer(&node_message, &mut node_message_bytes)
        .map_err(|e| {
            error!(
                target = "atoma-p2p",
                event = "serialize_usage_metrics_error",
                error = %e,
                "Failed to serialize usage metrics"
            );
            AtomaP2pNodeError::UsageMetricsSerializeError(e)
        })
        .expect("Failed to serialize usage metrics");
    let message_hash = blake3::hash(&node_message_bytes);

    let active_address = keystore.addresses()[0];
    let signature = keystore
        .sign_hashed(&active_address, message_hash.as_bytes())
        .expect("Failed to sign message");

    SignedNodeMessage {
        node_message,
        signature: Bytes::copy_from_slice(signature.as_ref()),
    }
}

#[tokio::test]
async fn test_validate_usage_metrics_message_success() {
    let keystore = create_test_keystore();
    let (tx, rx) = unbounded();

    // Create valid usage metrics
    let signed_node_message = create_test_signed_node_message(&keystore, 0, 1);

    // Mock successful node small ID ownership verification
    tokio::spawn(async move {
        let (event, optional_response_sender): (AtomaP2pEvent, Option<oneshot::Sender<bool>>) =
            rx.recv_async().await.unwrap();
        if let AtomaP2pEvent::VerifyNodeSmallIdOwnership {
            node_small_id,
            sui_address: _,
        } = event
        {
            assert_eq!(node_small_id, 1);
            let response_sender = optional_response_sender.unwrap();
            response_sender.send(true).unwrap();
        }
    });

    let mut node_meessage_bytes = Vec::new();
    ciborium::into_writer(&signed_node_message.node_message, &mut node_meessage_bytes).unwrap();
    let node_message_hash = blake3::hash(&node_meessage_bytes);
    // Validation should succeed
    let result = validate_signed_node_message(
        &signed_node_message.node_message,
        node_message_hash.as_bytes(),
        &signed_node_message.signature,
        &tx,
    )
    .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_validate_usage_metrics_message_invalid_url() {
    let keystore = create_test_keystore();
    let (tx, _rx) = unbounded();

    // Create base metrics then modify before serialization
    let node_message = NodeMessage {
        node_metadata: NodeP2pMetadata {
            node_public_url: "invalid_url".to_string(), // Direct invalid URL
            node_small_id: 1,
            country: "US".to_string(),
            timestamp: std::time::Instant::now().elapsed().as_secs(),
        },
        node_metrics: NodeMetrics::default(),
    };

    let mut bytes = Vec::new();
    ciborium::into_writer(&node_message, &mut bytes).unwrap();
    let hash = blake3::hash(&bytes);

    // Serialize modified message
    let signature = keystore
        .sign_hashed(&keystore.addresses()[0], hash.as_bytes())
        .unwrap();

    let bad_metrics = SignedNodeMessage {
        node_message,
        signature: Bytes::copy_from_slice(signature.as_ref()),
    };

    let result = validate_signed_node_message(
        &bad_metrics.node_message,
        hash.as_bytes(),
        &bad_metrics.signature,
        &tx,
    )
    .await;
    assert!(matches!(result, Err(AtomaP2pNodeError::UrlParseError(_))));
}

#[tokio::test]
async fn test_validate_usage_metrics_message_expired_timestamp() {
    let keystore = create_test_keystore();
    let (tx, _rx) = unbounded();

    // Create metrics with expired timestamp (11 minutes ago)
    let signed_node_message = create_test_signed_node_message(&keystore, -(11 * 60), 1);
    let node_message = &signed_node_message.node_message;
    let mut node_message_bytes = Vec::new();
    ciborium::into_writer(node_message, &mut node_message_bytes).unwrap();
    let node_message_hash = blake3::hash(&node_message_bytes);

    let result = validate_signed_node_message(
        node_message,
        node_message_hash.as_bytes(),
        &signed_node_message.signature,
        &tx,
    )
    .await;
    assert!(matches!(
        result,
        Err(AtomaP2pNodeError::InvalidPublicAddressError(_))
    ));
}

#[tokio::test]
async fn test_validate_usage_metrics_message_future_timestamp() {
    let keystore = create_test_keystore();
    let (tx, _rx) = unbounded();

    // Create metrics with future timestamp
    let signed_node_message = create_test_signed_node_message(&keystore, 60, 1);
    let node_message = &signed_node_message.node_message;
    let mut node_message_bytes = Vec::new();
    ciborium::into_writer(node_message, &mut node_message_bytes).unwrap();
    let node_message_hash = blake3::hash(&node_message_bytes);

    let result = validate_signed_node_message(
        node_message,
        node_message_hash.as_bytes(),
        &signed_node_message.signature,
        &tx,
    )
    .await;
    assert!(matches!(
        result,
        Err(AtomaP2pNodeError::InvalidPublicAddressError(_))
    ));
}

#[tokio::test]
async fn test_validate_usage_metrics_message_invalid_signature() {
    let keystore = create_test_keystore();
    let (tx, _rx) = unbounded();

    let signed_node_message = create_test_signed_node_message(&keystore, 0, 1);
    let node_message = &signed_node_message.node_message;

    // Corrupt the signature part after scheme byte
    let scheme_length = 1; // Ed25519 scheme byte
    let sig_start = scheme_length;
    let mut signature = signed_node_message.signature.as_ref().to_vec();
    for byte in &mut signature[sig_start..sig_start + 64] {
        *byte = 0xff;
    }
    let signature = Bytes::copy_from_slice(&signature);

    let mut node_message_bytes = Vec::new();
    ciborium::into_writer(&node_message, &mut node_message_bytes).unwrap();
    let node_message_hash = blake3::hash(&node_message_bytes);

    let result = validate_signed_node_message(
        &signed_node_message.node_message,
        node_message_hash.as_bytes(),
        &signature,
        &tx,
    )
    .await;
    assert!(matches!(
        result,
        Err(AtomaP2pNodeError::SignatureVerificationError(_))
    ));
}

#[tokio::test]
async fn test_validate_usage_metrics_message_invalid_node_ownership() {
    let keystore = create_test_keystore();
    let (tx, rx) = unbounded();

    // Create valid metrics
    let signed_node_message = create_test_signed_node_message(&keystore, 0, 1);
    let node_message = &signed_node_message.node_message;

    // Mock failed node small ID ownership verification
    tokio::spawn(async move {
        let (event, _response_sender) = rx.recv_async().await.unwrap();
        if let AtomaP2pEvent::VerifyNodeSmallIdOwnership {
            node_small_id,
            sui_address: _,
        } = event
        {
            assert_eq!(node_small_id, 1);
        }
    });

    let mut node_message_bytes = Vec::new();
    ciborium::into_writer(node_message, &mut node_message_bytes).unwrap();
    let node_message_hash = blake3::hash(&node_message_bytes);

    let result = validate_signed_node_message(
        node_message,
        node_message_hash.as_bytes(),
        &signed_node_message.signature,
        &tx,
    )
    .await;

    assert!(matches!(
        result,
        Err(AtomaP2pNodeError::NodeSmallIdOwnershipVerificationError(_))
    ));
}

#[tokio::test]
async fn test_validate_usage_metrics_message_state_manager_error() {
    let keystore = create_test_keystore();
    let (tx, rx) = unbounded();

    // Create valid metrics
    let signed_node_message = create_test_signed_node_message(&keystore, 0, 1);
    let node_message = &signed_node_message.node_message;

    // Mock state manager channel error by dropping the receiver
    drop(rx);

    let mut node_message_bytes = Vec::new();
    ciborium::into_writer(node_message, &mut node_message_bytes).unwrap();
    let node_message_hash = blake3::hash(&node_message_bytes);

    let result = validate_signed_node_message(
        node_message,
        node_message_hash.as_bytes(),
        &signed_node_message.signature,
        &tx,
    )
    .await;
    assert!(matches!(
        result,
        Err(AtomaP2pNodeError::StateManagerError(_))
    ));
}

#[tokio::test]
async fn test_validate_usage_metrics_message_response_channel_error() {
    let keystore = create_test_keystore();
    let (tx, rx) = unbounded();

    // Create valid metrics
    let signed_node_message = create_test_signed_node_message(&keystore, 0, 1);
    let node_message = &signed_node_message.node_message;

    // Mock response channel error
    tokio::spawn(async move {
        let (event, response_sender) = rx.recv_async().await.unwrap();
        if let AtomaP2pEvent::VerifyNodeSmallIdOwnership {
            node_small_id,
            sui_address: _,
        } = event
        {
            assert_eq!(node_small_id, 1);
            // Drop the sender without sending a response
            drop(response_sender);
        }
    });

    let mut node_message_bytes = Vec::new();
    ciborium::into_writer(node_message, &mut node_message_bytes).unwrap();
    let node_message_hash = blake3::hash(&node_message_bytes);

    let result = validate_signed_node_message(
        node_message,
        node_message_hash.as_bytes(),
        &signed_node_message.signature,
        &tx,
    )
    .await;
    assert!(matches!(
        result,
        Err(AtomaP2pNodeError::NodeSmallIdOwnershipVerificationError(_))
    ));
}
