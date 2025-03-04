use crate::errors::AtomaP2pNodeError;
use crate::metrics::{
    KAD_ROUTING_TABLE_SIZE, TOTAL_DIALS_ATTEMPTED, TOTAL_DIALS_FAILED,
    TOTAL_GOSSIPSUB_MESSAGES_FORWARDED, TOTAL_GOSSIPSUB_SUBSCRIPTIONS,
    TOTAL_INVALID_GOSSIPSUB_MESSAGES_RECEIVED, TOTAL_MDNS_DISCOVERIES,
};
use crate::service::{AtomaP2pBehaviour, AtomaP2pBehaviourEvent, StateManagerEvent};
use crate::types::SerializeWithSignature;
use crate::types::SignedNodeMessage;
use crate::utils::validate_signed_node_message;
use crate::AtomaP2pEvent;
use bytes::Bytes;
use flume::Sender;
use libp2p::metrics::Metrics;
use libp2p::metrics::Recorder;
use libp2p::{gossipsub, swarm::SwarmEvent};
use libp2p::{kad, mdns, PeerId, Swarm};
use opentelemetry::KeyValue;
use tracing::{debug, error, instrument};

/// # Panics
///
/// This function will panic if:
/// - `peer_id` is `None` when unwrapping in the `Dialing` and `OutgoingConnectionError` events
#[allow(clippy::too_many_lines)]
pub async fn handle_p2p_event(
    swarm: &mut Swarm<AtomaP2pBehaviour>,
    state_manager_sender: &Sender<StateManagerEvent>,
    event: SwarmEvent<AtomaP2pBehaviourEvent>,
    metrics: &mut Metrics,
    is_client: bool,
) {
    match event {
        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Gossipsub(gossipsub::Event::Message {
            message_id,
            message,
            propagation_source,
        })) => {
            match handle_gossipsub_message(
                swarm,
                state_manager_sender,
                message.data.into(),
                &message_id,
                &propagation_source,
                is_client,
            )
            .await
            {
                Ok(()) => {
                    TOTAL_GOSSIPSUB_MESSAGES_FORWARDED.add(
                        1,
                        &[KeyValue::new("peerId", swarm.local_peer_id().to_base58())],
                    );
                }
                Err(e) => {
                    TOTAL_INVALID_GOSSIPSUB_MESSAGES_RECEIVED.add(
                        1,
                        &[KeyValue::new("peerId", swarm.local_peer_id().to_base58())],
                    );
                    error!(
                        target = "atoma-p2p",
                        event = "gossipsub_message_error",
                        error = %e,
                        "Failed to handle gossipsub message"
                    );
                }
            }
        }
        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Gossipsub(
            gossipsub::Event::Subscribed { peer_id, topic },
        )) => {
            // Record subscript metrics
            TOTAL_GOSSIPSUB_SUBSCRIPTIONS.add(1, &[KeyValue::new("topic", topic.to_string())]);
            metrics.record(&gossipsub::Event::Subscribed {
                peer_id,
                topic: topic.clone(),
            });

            debug!(
                target = "atoma-p2p",
                event = "gossipsub_subscribed",
                peer_id = %peer_id,
                topic = %topic,
                "Peer subscribed to topic"
            );
        }
        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Gossipsub(
            gossipsub::Event::Unsubscribed { peer_id, topic },
        )) => {
            // Record unsubscription metrics
            TOTAL_GOSSIPSUB_SUBSCRIPTIONS.add(-1, &[KeyValue::new("topic", topic.to_string())]);
            metrics.record(&gossipsub::Event::Unsubscribed {
                peer_id,
                topic: topic.clone(),
            });

            debug!(
                target = "atoma-p2p",
                event = "gossipsub_unsubscribed",
                peer_id = %peer_id,
                topic = %topic,
                "Peer unsubscribed from topic"
            );
        }
        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Mdns(mdns::Event::Discovered(
            discovered_peers,
        ))) => {
            let peer_count = discovered_peers.len() as u64;
            debug!(
                target = "atoma-p2p",
                event = "mdns_discovered",
                peer_count = %peer_count,
                "Mdns discovered peers"
            );
            for (peer_id, multiaddr) in discovered_peers {
                debug!(
                    target = "atoma-p2p",
                    event = "mdns_discovered_peer",
                    peer_id = %peer_id,
                    multiaddr = %multiaddr,
                    "Mdns discovered peer"
                );
                swarm
                    .behaviour_mut()
                    .kademlia
                    .add_address(&peer_id, multiaddr);
            }
            // Record discovery metrics
            TOTAL_MDNS_DISCOVERIES.add(
                peer_count,
                &[KeyValue::new("peerId", swarm.local_peer_id().to_base58())],
            );
        }
        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Mdns(mdns::Event::Expired(
            expired_peers,
        ))) => {
            debug!(
                target = "atoma-p2p",
                event = "mdns_expired",
                num_expired_peers = %expired_peers.len(),
                "Mdns expired"
            );
            for (peer_id, multiaddr) in expired_peers {
                debug!(
                    target = "atoma-p2p",
                    event = "mdns_expired_peer",
                    peer_id = %peer_id,
                    multiaddr = %multiaddr,
                    "Mdns expired peer"
                );
                swarm
                    .behaviour_mut()
                    .kademlia
                    .remove_address(&peer_id, &multiaddr);
                swarm
                    .behaviour_mut()
                    .gossipsub
                    .remove_explicit_peer(&peer_id);
            }
        }
        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Kademlia(kad::Event::RoutingUpdated {
            peer,
            is_new_peer,
            addresses,
            bucket_range,
            old_peer,
        })) => {
            debug!(
                target = "atoma-p2p",
                event = "kademlia_routing_updated",
                peer = %peer,
                is_new_peer = %is_new_peer,
                addresses = ?addresses,
                bucket_range = ?bucket_range,
                old_peer = ?old_peer,
                "Kademlia routing updated"
            );
            KAD_ROUTING_TABLE_SIZE.record(
                addresses.len() as u64,
                &[KeyValue::new("peerId", peer.to_base58())],
            );
            metrics.record(&kad::Event::RoutingUpdated {
                peer,
                is_new_peer,
                addresses,
                bucket_range,
                old_peer,
            });
        }
        SwarmEvent::ConnectionEstablished {
            peer_id,
            num_established,
            established_in,
            connection_id,
            ..
        } => {
            debug!(
                target = "atoma-p2p",
                event = "peer_connection_established",
                peer_id = %peer_id,
                num_established = %num_established,
                established_in = ?established_in,
                connection_id = %connection_id,
                "Peer connection established"
            );
        }
        SwarmEvent::ConnectionClosed {
            peer_id,
            connection_id,
            num_established,
            ..
        } => {
            debug!(
                target = "atoma-p2p",
                event = "peer_connection_closed",
                peer_id = %peer_id,
                connection_id = %connection_id,
                num_established = %num_established,
                "Peer connection closed"
            );
        }
        SwarmEvent::NewListenAddr {
            listener_id,
            address,
            ..
        } => {
            debug!(
                target = "atoma-p2p",
                event = "new_listen_addr",
                listener_id = %listener_id,
                address = %address,
                "New listen address"
            );
        }
        SwarmEvent::ExpiredListenAddr {
            listener_id,
            address,
        } => {
            debug!(
                target = "atoma-p2p",
                event = "expired_listen_addr",
                listener_id = %listener_id,
                address = %address,
                "Expired listen address"
            );
        }
        SwarmEvent::Dialing {
            peer_id,
            connection_id,
            ..
        } => {
            debug!(
                target = "atoma-p2p",
                event = "dialing",
                peer_id = ?peer_id,
                connection_id = %connection_id,
                "Dialing peer"
            );
            TOTAL_DIALS_ATTEMPTED.add(1, &[KeyValue::new("peerId", peer_id.unwrap().to_base58())]);
        }
        SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
            TOTAL_DIALS_FAILED.add(1, &[KeyValue::new("peerId", peer_id.unwrap().to_base58())]);
            error!(
                target = "atoma-p2p",
                event = "outgoing_connection_error",
                peer_id = ?peer_id,
                error = %error,
                "Outgoing connection error"
            );
        }
        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Identify(identify_event)) => {
            tracing::debug!(
                target = "atoma-p2p",
                event = "identify",
                identify_event = ?identify_event,
                "Identify event"
            );
            metrics.record(&identify_event);
        }
        SwarmEvent::Behaviour(AtomaP2pBehaviourEvent::Kademlia(kad_event)) => {
            tracing::debug!(
                target = "atoma-p2p",
                event = "kad",
                kad_event = ?kad_event,
                "Kad event"
            );
            metrics.record(&kad_event);
        }
        swarm_event => {
            tracing::debug!(
                target = "atoma-p2p",
                event = "swarm_event",
                swarm_event = ?swarm_event,
                "Swarm event"
            );
            metrics.record(&swarm_event);
        }
    }
}

/// Handles incoming gossipsub messages in the P2P network.
///
/// This method processes UsageMetrics messages by:
/// 1. Validating the message's signature and timestamp
/// 2. Verifying the node's ownership of its small ID
/// 3. Reporting the validation result to the gossipsub protocol
/// 4. Storing the node's public URL in the state manager (if this peer is a client)
///
/// # Message Flow
///
/// 1. Receives a message from the gossipsub network
/// 2. Skips processing if the message is from self
/// 3. Deserializes the message into a GossipMessage
/// 4. For UsageMetrics messages:
///    - Validates the message using `validate_usage_metrics_message`
///    - Reports validation result to gossipsub protocol
///    - If this peer is a client, stores the node's public URL in state manager
///
/// # Arguments
///
/// * `message_data` - Raw bytes of the received gossipsub message (CBOR encoded)
/// * `message_id` - Unique identifier for the gossipsub message
/// * `propagation_source` - The peer that forwarded this message
///
/// # Returns
///
/// Returns `Ok(())` if message processing succeeds, or an error if:
/// - Message deserialization fails
/// - Message validation fails
/// - Reporting validation result fails
/// - Storing public URL fails (for clients)
///
/// # Errors
///
/// This function can return the following errors:
/// * `UsageMetricsSerializeError` - If CBOR deserialization fails
/// * `StateManagerError` - If storing public URL fails (for clients)
///
/// # Security Considerations
///
/// - Messages from self are ignored to prevent message loops
/// - Messages are validated before processing
/// - Only clients store public URLs to prevent unnecessary data storage
/// - Uses CBOR for efficient binary serialization
///
/// # Example
///
/// ```rust,ignore
/// let message_data = // ... received from gossipsub ...;
/// let message_id = // ... message identifier ...;
/// let propagation_source = // ... peer ID ...;
///
/// match node.handle_gossipsub_message(&message_data, &message_id, &propagation_source).await {
///     Ok(()) => println!("Message processed successfully"),
///     Err(e) => println!("Failed to process message: {}", e),
/// }
/// ```
///
/// # Message Validation
///
/// Messages are validated by:
/// 1. Checking the URL format and timestamp freshness
/// 2. Verifying the cryptographic signature
/// 3. Confirming the node's ownership of its small ID
///
/// Invalid messages are rejected and not propagated further in the network.
#[instrument(level = "debug", skip_all)]
pub async fn handle_gossipsub_message(
    swarm: &mut Swarm<AtomaP2pBehaviour>,
    state_manager_sender: &Sender<StateManagerEvent>,
    message_data: Bytes,
    message_id: &gossipsub::MessageId,
    propagation_source: &PeerId,
    is_client: bool,
) -> Result<(), AtomaP2pNodeError> {
    debug!(
        target = "atoma-p2p",
        event = "gossipsub_message",
        message_id = %message_id,
        propagation_source = %propagation_source,
        "Received gossipsub message"
    );
    if propagation_source == swarm.local_peer_id() {
        debug!(
            target = "atoma-p2p",
            event = "gossipsub_message_from_self",
            "Gossipsub message from self"
        );
        // Do not re-publish the node's own message, just return `Ok(())
        return Ok(());
    }
    // Directly deserialize SignedNodeMessage using new method
    let signed_node_message = SignedNodeMessage::deserialize_with_signature(&message_data)?;
    let signature_len = signed_node_message.signature.len();
    debug!(
        target = "atoma-p2p",
        event = "gossipsub_message_data",
        message_id = %message_id,
        propagation_source = %propagation_source,
        "Received gossipsub message data"
    );
    let node_message = &signed_node_message.node_message;
    let node_message_hash = blake3::hash(&message_data[signature_len..]);
    let message_acceptance = match validate_signed_node_message(
        node_message,
        node_message_hash.as_bytes(),
        &signed_node_message.signature,
        state_manager_sender,
    )
    .await
    {
        Ok(()) => gossipsub::MessageAcceptance::Accept,
        Err(e) => {
            error!(
                target = "atoma-p2p",
                event = "gossipsub_message_validation_error",
                error = %e,
                "Failed to validate gossipsub message"
            );
            // NOTE: We should reject the message if it fails to validate
            // as it means the node is not being following the current protocol
            if let AtomaP2pNodeError::UrlParseError(_) = e {
                // We remove the peer from the gossipsub topic, because it is not a valid URL and therefore cannot be reached
                // by clients for processing OpenAI api compatible AI requests, so these peers are not useful for the network
                swarm
                    .behaviour_mut()
                    .gossipsub
                    .remove_explicit_peer(propagation_source);
            }
            gossipsub::MessageAcceptance::Reject
        }
    };
    // Report the message validation result to the gossipsub protocol
    let is_in_mempool = swarm
        .behaviour_mut()
        .gossipsub
        .report_message_validation_result(message_id, propagation_source, message_acceptance);
    if is_in_mempool {
        debug!(
            target = "atoma-p2p",
            event = "gossipsub_message_in_mempool",
            message_id = %message_id,
            propagation_source = %propagation_source,
            "Gossipsub message already in the mempool, no need to take further actions"
        );
        return Ok(());
    }
    // If the current peer is a client, we need to store the public URL in the state manager
    if is_client {
        let node_message = signed_node_message.node_message;
        let event = AtomaP2pEvent::NodeMetricsRegistrationEvent {
            public_url: node_message.node_metadata.node_public_url,
            node_small_id: node_message.node_metadata.node_small_id,
            timestamp: node_message.node_metadata.timestamp,
            country: node_message.node_metadata.country,
            node_metrics: node_message.node_metrics,
        };
        state_manager_sender.send((event, None)).map_err(|e| {
            error!(
                target = "atoma-p2p",
                event = "gossipsub_message_state_manager_error",
                error = %e,
                "Failed to send event to state manager"
            );
            AtomaP2pNodeError::StateManagerError(e)
        })?;
    }

    Ok(())
}
