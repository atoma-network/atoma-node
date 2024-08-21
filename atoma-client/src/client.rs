use std::path::Path;

use atoma_crypto::{calculate_commitment, Blake2b};
use atoma_types::{AtomaOutputMetadata, Digest, OutputType, Response};
use rmp_serde::Deserializer;
use serde::Deserialize;
use serde_json::Value;
use sui_sdk::{
    json::SuiJsonValue,
    types::{
        base_types::{ObjectIDParseError, SuiAddress},
        SUI_RANDOMNESS_STATE_OBJECT_ID,
    },
    wallet_context::WalletContext,
};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, instrument};

use crate::config::AtomaSuiClientConfig;

const GAS_BUDGET: u64 = 5_000_000; // 0.005 SUI

const MODULE_ID: &str = "settlement";
const METHOD: &str = "submit_commitment";

/// `AtomaSuiClient` - The interface responsible for a node to commit to a generated output, on the Sui blockchain.
pub struct AtomaSuiClient {
    /// Sui address
    address: SuiAddress,
    /// Atoma's configuration for the Sui client.
    config: AtomaSuiClientConfig,
    /// Sui's wallet context
    wallet_ctx: WalletContext,
    /// A mpsc receiver, which is responsible to receive new `Response`'s, so that the node
    /// can then commit to these.
    response_rx: mpsc::Receiver<Response>,
    /// A mpsc sender, responsible to send the actual output to the `OutputManager` service (for being shared with an end user or protocol)
    /// It sends a tuple, containing the output's metadata and the actual output (in JSON format).
    output_manager_tx: mpsc::Sender<(AtomaOutputMetadata, String)>,
}

impl AtomaSuiClient {
    /// Constructs a new instance from an `AtomaSuiClientConfig`.
    ///
    /// Inputs:
    ///     `config` - The Atoma Sui client configuration.
    ///     `response_rx` - A mpsc receiver, associated to a `Response`.
    ///     `output_manager_tx` - A mpsc sender, associated with a tuple (`Digest`, `Response`), responsible for
    ///         sharing the actual output with the `OutputManager` service.
    pub fn new_from_config(
        config: AtomaSuiClientConfig,
        response_rx: mpsc::Receiver<Response>,
        output_manager_tx: mpsc::Sender<(AtomaOutputMetadata, String)>,
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

    /// Constructs a new instance from a configuration file path (which can be deserialized into a `AtomaSuiClientConfig`)
    ///
    /// Inputs:
    ///     `config_path` - Path for the configuration file, which is deserialized into an `AtomaSuiClientConfig`.
    ///     `response_rx` - A mpsc receiver, associated to a `Response`.
    ///     `output_manager_tx` - A mpsc sender, associated with a tuple (`AtomaOutputMetadata`, `String`), responsible for
    ///         sharing the actual output with the `OutputManager` service.
    pub fn new_from_config_file<P: AsRef<Path>>(
        config_path: P,
        response_rx: mpsc::Receiver<Response>,
        output_manager_tx: mpsc::Sender<(AtomaOutputMetadata, String)>,
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
    ///
    ///   These are then combined into a single byte vector where the image data is followed by the height and width.
    #[instrument(skip(self, data))]
    fn get_data(&self, data: Value) -> Result<Vec<u8>, AtomaSuiClientError> {
        // TODO: rework this when responses get same structure
        if let Some(text) = data["text"].as_str() {
            Ok(text.as_bytes().to_owned())
        } else if let Some(img_data) = data["image_data"].as_array() {
            let img = img_data
                .iter()
                .map(|b| b.as_u64().ok_or(AtomaSuiClientError::MissingOutputData))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|b| b as u8)
                .collect::<Vec<_>>();
            let height = data["height"]
                .as_u64()
                .ok_or(AtomaSuiClientError::MissingOutputData)?
                .to_le_bytes();
            let width = data["width"]
                .as_u64()
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
    #[instrument(skip_all)]
    pub async fn submit_response_commitment(
        &self,
        response: Response,
    ) -> Result<Digest, AtomaSuiClientError> {
        let request_id = response.id();
        let data = self.get_data(response.response().clone())?;
        let (index, num_leaves) = (response.sampled_node_index(), response.num_sampled_nodes());
        let (root, pre_image) = calculate_commitment::<Blake2b<_>, _>(data, index, num_leaves);

        let num_input_tokens = response.input_tokens();
        let num_output_tokens = response.tokens_count();

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
                    SuiJsonValue::new(request_id.clone().into())?,
                    SuiJsonValue::new(num_input_tokens.to_string().into())?,
                    SuiJsonValue::new(num_output_tokens.to_string().into())?,
                    SuiJsonValue::new(root.as_ref().into())?,
                    SuiJsonValue::new(pre_image.as_ref().into())?,
                    SuiJsonValue::from_object_id(SUI_RANDOMNESS_STATE_OBJECT_ID),
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
        let hex_request_id = hex::encode(request_id.as_slice());

        if let Some(events) = tx_response.events {
            for event in events.data.iter() {
                debug!("Got a transaction event: {:?}", event.type_.name.as_str());
                if event.type_.name.as_str() == "FirstSubmissionEvent" {
                    let output = response.response();
                    let output_destination = Deserialize::deserialize(&mut Deserializer::new(
                        response.output_destination(),
                    ))?;
                    let output_type = response.output_type();
                    let output = match output_type {
                        OutputType::Text => output["text"]
                            .as_str()
                            .ok_or(AtomaSuiClientError::MissingOutputData)?
                            .to_string(),
                        OutputType::Image => {
                            todo!()
                        }
                    };
                    let output_metadata = AtomaOutputMetadata {
                        transaction_base_58: tx_digest.clone(),
                        node_public_key: self.address.to_string(),
                        ticket_id: hex_request_id,
                        num_input_tokens,
                        num_output_tokens,
                        num_sampled_nodes: num_leaves,
                        index_of_node: index,
                        time_to_generate: response.time_to_generate(),
                        commitment_root_hash: root.to_vec(),
                        leaf_hash: pre_image.to_vec(),
                        output_destination,
                        output_type,
                    };

                    self.output_manager_tx
                        .send((output_metadata, output))
                        .await
                        .map_err(Box::new)?;
                    // we don't need to check other events, as at this point the node knows it has been selected for
                    break;
                }
            }
        }
        Ok(tx_digest)
    }

    /// Responsible for running the `AtomaSuiClient` main loop.
    ///
    /// It listens to new incoming `Response`'s from the `AtomaInference` service. Once it gets
    /// a new response in, it constructs a new commitment to the `Response` that is then submitted
    /// on the Atoma smart contract, on the Sui blockchain.
    #[instrument(skip_all)]
    pub async fn run(mut self) -> Result<(), AtomaSuiClientError> {
        while let Some(response) = self.response_rx.recv().await {
            info!("Received new response: {:?}", response);
            if let Err(e) = self.submit_response_commitment(response).await {
                error!("Failed to submit response commitment: {:?}", e);
            }
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
    SendError(#[from] Box<mpsc::error::SendError<(AtomaOutputMetadata, String)>>),
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
    #[error("Rmp deserialize error: `{0}`")]
    RmpDeseriliazeError(#[from] rmp_serde::decode::Error),
}
