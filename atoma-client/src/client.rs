use std::{path::Path, str::FromStr, time::Duration};

use atoma_crypto::{calculate_commitment, Sha256};
use sui_keys::keystore::AccountKeystore;
use sui_sdk::{
    json::SuiJsonValue,
    types::{
        base_types::{ObjectID, ObjectIDParseError, SuiAddress},
        crypto::Signature,
        digests::TransactionDigest,
    },
    wallet_context::WalletContext,
};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::info;

const GAS_BUDGET: u64 = 5_000_000; // 0.005 SUI

const PACKAGE_ID: &str = "<TODO>";
const MODULE_ID: &str = "";
const METHOD: &str = "command";

pub struct AtomaSuiClient {
    address: SuiAddress,
    wallet_ctx: WalletContext,
    response_receiver: mpsc::Receiver<serde_json::Value>,
}

impl AtomaSuiClient {
    pub fn new<P: AsRef<Path>>(
        config_path: P,
        request_timeout: Option<Duration>,
        max_concurrent_requests: Option<u64>,
        response_receiver: mpsc::Receiver<serde_json::Value>,
    ) -> Result<Self, AtomaSuiClientError> {
        let mut wallet_ctx = WalletContext::new(
            config_path.as_ref(),
            request_timeout,
            max_concurrent_requests,
        )?;
        let active_address = wallet_ctx.active_address()?;
        info!("Set Sui client, with active address: {}", active_address);
        Ok(Self {
            address: active_address,
            wallet_ctx,
            response_receiver,
        })
    }

    fn get_index(
        &self,
        sampled_nodes: serde_json::Value,
    ) -> Result<(usize, usize), AtomaSuiClientError> {
        let (index, num_leaves) = if let Some(sampled_nodes) = sampled_nodes.as_array() {
            let idx = sampled_nodes
                .iter()
                .enumerate()
                .find(|(_, pk)| {
                    let pk_bytes = pk
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|u| u.as_u64().unwrap() as u8)
                        .collect::<Vec<_>>();
                    self.address.as_ref() == pk_bytes
                })
                .unwrap()
                .0;
            (idx, sampled_nodes.len())
        } else {
            return Err(AtomaSuiClientError::InvalidSampledNode);
        };
        Ok((index, num_leaves))
    }

    fn get_data(&self, data: serde_json::Value) -> Result<Vec<u8>, AtomaSuiClientError> {
        // TODO: rework this when responses get same structure
        let data = match data.as_str() {
            Some(text) => text.as_bytes().to_owned(),
            None => {
                if let Some(array) = data.as_array() {
                    array
                        .iter()
                        .map(|b| b.as_u64().unwrap() as u8)
                        .collect::<Vec<_>>()
                } else {
                    return Err(AtomaSuiClientError::FailedResponseJsonParsing);
                }
            }
        };
        Ok(data)
    }

    fn sign_merkle_root(&self, merkle_root: [u8; 32]) -> Result<Signature, AtomaSuiClientError> {
        self.wallet_ctx
            .config
            .keystore
            .sign_hashed(&self.address, merkle_root.as_slice())
            .map_err(|e| AtomaSuiClientError::FailedSignature(e.to_string()))
    }

    pub async fn submit_response_commitment(
        &self,
        response: serde_json::Value,
    ) -> Result<TransactionDigest, AtomaSuiClientError> {
        let request_id = response["id"]
            .as_u64()
            .ok_or(AtomaSuiClientError::InvalidRequestId)?;
        let data = self.get_data(response["data"].clone())?;
        let (index, num_leaves) = self.get_index(response["sampled_nodes"].clone())?;
        let (merkle_root, merkle_path) = calculate_commitment::<Sha256, _>(data, index, num_leaves);
        let signature = self.sign_merkle_root(merkle_root)?;

        let client = self.wallet_ctx.get_client().await?;
        let tx = client
            .transaction_builder()
            .move_call(
                self.address,
                ObjectID::from_str(PACKAGE_ID)?,
                MODULE_ID,
                METHOD,
                vec![],
                vec![
                    SuiJsonValue::new(request_id.into())?,
                    SuiJsonValue::new(signature.as_ref().into())?,
                    SuiJsonValue::new(
                        merkle_path
                            .proof_hashes()
                            .iter()
                            .map(|u| u.to_vec())
                            .collect::<Vec<_>>()
                            .into(),
                    )?,
                ],
                None,
                GAS_BUDGET,
                None,
            )
            .await?;

        let tx = self.wallet_ctx.sign_transaction(&tx);
        let resp = self.wallet_ctx.execute_transaction_must_succeed(tx).await;

        Ok(resp.digest)
    }

    pub async fn run(mut self) -> Result<(), AtomaSuiClientError> {
        while let Some(response) = self.response_receiver.recv().await {
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
    #[error("Failed response JSON parsing")]
    FailedResponseJsonParsing,
    #[error("No available funds")]
    NoAvailableFunds,
    #[error("Invalid sampled node")]
    InvalidSampledNode,
    #[error("Invalid request id")]
    InvalidRequestId,
}
