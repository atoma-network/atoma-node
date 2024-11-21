use chrono::Utc;
use std::str::FromStr;

use anyhow::Result;
use messages::UpdatePublicAddress;
use prost::Message;
use topics::ATOMA_UPDATE_PUBLIC_ADDRESS;
use tracing::{error, info, trace};
use waku_bindings::{
    waku_default_pubsub_topic, waku_new, waku_set_event_callback, ContentFilter, Multiaddr,
    ProtocolId, Running, WakuMessage, WakuNodeHandle,
};

mod messages;
mod topics;

const NODES: &[&str] = &[
    // "/dns4/node-01.ac-cn-hongkong-c.wakuv2.test.statusim.net/tcp/30303/p2p/16Uiu2HAkvWiyFsgRhuJEb9JfjYxEkoHLgnUQmr1N5mKWnYjxYRVm",
    // "/dns4/node-01.do-ams3.wakuv2.test.statusim.net/tcp/30303/p2p/16Uiu2HAmPLe7Mzm8TsYUubgCAW1aJoeFScxrLj8ppHFivPo97bUZ",
    // "/dns4/node-01.gc-us-central1-a.wakuv2.test.statusim.net/tcp/30303/p2p/16Uiu2HAmJb2e28qLXxT5kZxVUUoJt72EMzNGXB47Rxx5hw3q4YjS"
    "/dns4/store-01.do-ams3.shards.test.status.im/tcp/30303/p2p/16Uiu2HAmAUdrQ3uwzuE4Gy4D56hX6uLKEeerJAnhKEHZ3DxF1EfT",
    "/dns4/store-02.do-ams3.shards.test.status.im/tcp/30303/p2p/16Uiu2HAm9aDJPkhGxc2SFcEACTFdZ91Q5TJjp76qZEhq9iF59x7R",
    "/dns4/store-01.gc-us-central1-a.shards.test.status.im/tcp/30303/p2p/16Uiu2HAmMELCo218hncCtTvC2Dwbej3rbyHQcR8erXNnKGei7WPZ",
    "/dns4/store-02.gc-us-central1-a.shards.test.status.im/tcp/30303/p2p/16Uiu2HAmJnVR7ZzFaYvciPVafUXuYGLHPzSUigqAmeNw9nJUVGeM",
    "/dns4/store-01.ac-cn-hongkong-c.shards.test.status.im/tcp/30303/p2p/16Uiu2HAm2M7xs7cLPc3jamawkEqbr7cUJX11uvY7LxQ6WFUdUKUT",
    "/dns4/store-02.ac-cn-hongkong-c.shards.test.status.im/tcp/30303/p2p/16Uiu2HAm9CQhsuwPR54q27kNj9iaQVfyRzTGKrhFmr94oD8ujU6P"
];

pub struct AtomaWaku {
    node_handle: WakuNodeHandle<Running>,
}

impl AtomaWaku {
    pub fn new() -> Result<Self> {
        let node_handle = Self::setup_node_handle()?;
        let this = Self { node_handle };
        this.setup_callback();
        Ok(this)
    }

    fn setup_node_handle() -> Result<WakuNodeHandle<Running>> {
        let node_handle = waku_new(None).map_err(|error| anyhow::anyhow!(error))?;
        let node_handle = node_handle
            .start()
            .map_err(|error| anyhow::anyhow!(error))?;
        for address in NODES
            .iter()
            .map(|a| Multiaddr::from_str(a))
            .collect::<Result<Vec<_>, _>>()?
        {
            let peer_id = node_handle.add_peer(&address, ProtocolId::Relay);
            match peer_id {
                Ok(peer_id) => {
                    match node_handle.connect_peer_with_id(&peer_id, None) {
                        Ok(()) => {
                            info!("Connected to peer {peer_id}");
                        }
                        Err(e) => {
                            error!("Failed to connect to peer {peer_id}: {e}");
                        }
                    };
                }
                Err(e) => {
                    error!("Failed to add peer {address}: {e}");
                }
            }
        }

        let content_filter = ContentFilter::new(Some(waku_default_pubsub_topic()), vec![]);
        match node_handle.relay_subscribe(&content_filter) {
            Ok(()) => {
                info!("Subscribed to relay");
                Ok(node_handle)
            }
            Err(e) => {
                error!("Failed to subscribe to relay : {e}");
                Err(anyhow::anyhow!(e))
            }
        }
    }

    fn setup_callback(&self) {
        waku_set_event_callback(move |signal| match signal.event() {
            waku_bindings::Event::WakuMessage(event) => {
                info!("Waku message");
                if event.waku_message().content_topic() != &ATOMA_UPDATE_PUBLIC_ADDRESS {
                    return;
                }
                match <UpdatePublicAddress as Message>::decode(event.waku_message().payload()) {
                    Ok(atoma_message) => {
                        info!(
                            "Received Atoma message from node {} with address {}",
                            atoma_message.node_small_id(),
                            atoma_message.address()
                        );
                    }
                    Err(e) => {
                        error!("{e:?}");
                    }
                }
            }
            waku_bindings::Event::Unrecognized(data) => {
                error!("Error, received unrecognized event {data}");
            }
            _ => {
                trace!("Received event");
            }
        });
    }

    pub fn send_update_public_address(
        &self,
        node_small_id: u64,
        address: String,
    ) -> Result<String> {
        let message = UpdatePublicAddress::new(node_small_id, &address);

        let mut buff = Vec::new();
        let meta = Vec::new();
        Message::encode(&message, &mut buff).unwrap();
        let waku_message = WakuMessage::new(
            buff,
            ATOMA_UPDATE_PUBLIC_ADDRESS.clone(),
            1,
            Utc::now().timestamp_nanos_opt().unwrap() as usize,
            meta,
            false,
        );
        self.node_handle
            .relay_publish_message(&waku_message, Some(waku_default_pubsub_topic()), None)
            .map_err(|error| anyhow::anyhow!(error))
    }
}
