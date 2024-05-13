use std::path::Path;

use atoma_crypto::{calculate_commitment, Blake2b};
use atoma_types::{Digest, Response};
use sui_sdk::{
    json::SuiJsonValue,
    types::base_types::{ObjectIDParseError, SuiAddress},
    wallet_context::WalletContext,
};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use crate::config::AtomaSuiClientConfig;

const GAS_BUDGET: u64 = 5_000_000; // 0.005 SUI

const MODULE_ID: &str = "settlement";
const METHOD: &str = "submit_commitment";

pub struct AtomaSuiClient {
    address: SuiAddress,
    config: AtomaSuiClientConfig,
    wallet_ctx: WalletContext,
    response_rx: mpsc::Receiver<Response>,
    output_manager_tx: mpsc::Sender<(Digest, Response)>,
}

impl AtomaSuiClient {
    pub fn new_from_config(
        config: AtomaSuiClientConfig,
        response_rx: mpsc::Receiver<Response>,
        output_manager_tx: mpsc::Sender<(Digest, Response)>,
    ) -> Result<Self, AtomaSuiClientError> {
        info!("Initializing Sui wallet..");
        let mut wallet_ctx = WalletContext::new(
            config.config_path().as_ref(),
            Some(config.request_timeout()),
            Some(config.max_concurrent_requests()),
        )?;
        let active_address = wallet_ctx.active_address()?;
        info!("Set Sui client, with active address: {}", active_address);
        Ok(Self {
            address: active_address,
            config,
            wallet_ctx,
            response_rx,
            output_manager_tx,
        })
    }

    pub fn new_from_config_file<P: AsRef<Path>>(
        config_path: P,
        response_rx: mpsc::Receiver<Response>,
        output_manager_tx: mpsc::Sender<(Digest, Response)>,
    ) -> Result<Self, AtomaSuiClientError> {
        let config = AtomaSuiClientConfig::from_file_path(config_path);
        Self::new_from_config(config, response_rx, output_manager_tx)
    }

    /// Extracts and processes data from a JSON response to generate a byte vector.
    ///
    /// This method handles two types of data structures within the JSON response:
    /// - If the JSON contains a "text" field, it converts the text to a byte vector.
    /// - If the JSON contains an array with at least three elements, it interprets:
    ///   - The first element as an array of bytes (representing an image byte content),
    ///   - The second element as the image height,
    ///   - The third element as the image width.
    ///   These are then combined into a single byte vector where the image data is followed by the height and width.
    fn get_data(&self, data: serde_json::Value) -> Result<Vec<u8>, AtomaSuiClientError> {
        // TODO: rework this when responses get same structure
        if let Some(text) = data["text"].as_str() {
            Ok(text.as_bytes().to_owned())
        } else if let Some(array) = data.as_array() {
            if array.len() < 3 {
                error!("Incomplete image data");
                return Err(AtomaSuiClientError::MissingOutputData);
            }

            let img_data = array
                .first()
                .and_then(|img| img.as_array())
                .ok_or(AtomaSuiClientError::MissingOutputData)?;
            let img = img_data
                .iter()
                .map(|b| b.as_u64().ok_or(AtomaSuiClientError::MissingOutputData))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|b| b as u8)
                .collect::<Vec<_>>();
            let height = array
                .get(1)
                .and_then(|h| h.as_u64())
                .ok_or(AtomaSuiClientError::MissingOutputData)?
                .to_le_bytes();
            let width = array
                .get(2)
                .and_then(|w| w.as_u64())
                .ok_or(AtomaSuiClientError::MissingOutputData)?
                .to_le_bytes();

            info!("Image data length: {:?}", img.len());

            let mut result = img;
            result.extend_from_slice(&height);
            result.extend_from_slice(&width);

            Ok(result)
        } else {
            error!("Invalid JSON structure for data extraction");
            return Err(AtomaSuiClientError::FailedResponseJsonParsing);
        }
    }

    /// Upon receiving a response from the `AtomaNode` service, this method extracts
    /// the output data and computes a cryptographic commitment. The commitment includes
    /// the root of an n-ary Merkle Tree built from the output data, represented as a `Vec<u8>`,
    /// and the indices of the sampled nodes used for inference.
    /// Additionally, the commitment contains a Merkle path from the node's leaf index
    /// (in the `sampled_nodes` vector) to the root.
    ///
    /// This data is then submitted to the Sui blockchain
    /// as a cryptographic commitment to the node's work on inference.
    pub async fn submit_response_commitment(
        &self,
        response: Response,
    ) -> Result<Digest, AtomaSuiClientError> {
        let request_id = response.id();
        let data = self.get_data(response.response())?;
        let (index, num_leaves) = (response.sampled_node_index(), response.num_sampled_nodes());
        let (root, pre_image) = calculate_commitment::<Blake2b<_>, _>(data, index, num_leaves);

        let client = self.wallet_ctx.get_client().await?;
        let tx = client
            .transaction_builder()
            .move_call(
                self.address,
                self.config.package_id(),
                MODULE_ID,
                METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db_id()),
                    SuiJsonValue::from_object_id(self.config.node_badge_id()),
                    SuiJsonValue::new(request_id.into())?,
                    SuiJsonValue::new(root.as_ref().into())?,
                    SuiJsonValue::new(pre_image.as_ref().into())?,
                ],
                None,
                GAS_BUDGET,
                None,
            )
            .await?;

        let tx = self.wallet_ctx.sign_transaction(&tx);
        let tx_response = self.wallet_ctx.execute_transaction_must_succeed(tx).await;

        debug!("Submitted transaction with response: {:?}", tx_response);

        let tx_digest = tx_response.digest.base58_encode();
        if let Some(events) = tx_response.events {
            for event in events.data.iter() {
                debug!("Got a transaction event: {:?}", event.type_.name.as_str());
                if event.type_.name.as_str() == "FirstSubmissionEvent" {
                    self.output_manager_tx
                        .send((tx_digest.clone(), response))
                        .await?;
                    break; // we don't need to check other events, as at this point the node knows it has been selected for
                }
            }
        }
        Ok(tx_digest)
    }

    pub async fn run(mut self) -> Result<(), AtomaSuiClientError> {
        while let Some(response) = self.response_rx.recv().await {
            info!("Received new response: {:?}", response);
            self.submit_response_commitment(response).await?;
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum AtomaSuiClientError {
    #[error("Sui Builder error: `{0}`")]
    SuiBuilderError(#[from] sui_sdk::error::Error),
    #[error("Failed to parse address: `{0}`")]
    FailedParseAddress(#[from] anyhow::Error),
    #[error("Object ID parse error: `{0}`")]
    ObjectIDParseError(#[from] ObjectIDParseError),
    #[error("Failed signature: `{0}`")]
    FailedSignature(String),
    #[error("Sender error: `{0}`")]
    SendError(#[from] mpsc::error::SendError<(Digest, Response)>),
    #[error("Failed response JSON parsing")]
    FailedResponseJsonParsing,
    #[error("No available funds")]
    NoAvailableFunds,
    #[error("Invalid sampled node")]
    InvalidSampledNode,
    #[error("Invalid request id")]
    InvalidRequestId,
    #[error("Missing output data")]
    MissingOutputData,
}
