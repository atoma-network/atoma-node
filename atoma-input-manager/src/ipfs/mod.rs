use futures::TryStreamExt;
use ipfs_api_backend_hyper::{IpfsApi, IpfsClient, TryFromUri};
use serde::Deserialize;
use tracing::{error, info, instrument};

use crate::{config::AtomaInputManagerConfig, AtomaInputManagerError};

/// IPFS input manager
pub struct IpfsInputManager {
    /// IPFS Hyper client
    client: IpfsClient,
}

impl IpfsInputManager {
    /// Constructor
    #[tracing::instrument(skip_all)]
    pub async fn new(config: &AtomaInputManagerConfig) -> Result<Self, AtomaInputManagerError> {
        info!("Building IPFS client...");
        let client = IpfsClient::from_str(
            config
                .ipfs_api_url
                .as_str(),
        )
        .map_err(|e| AtomaInputManagerError::FailedToBuildIpfsClient(e.to_string()))?
        .with_credentials(
            config
                .ipfs_username
                .clone(),
            config
                .ipfs_password
                .clone(),
        );
        match client.version().await {
            Ok(version) => {
                info!(
                    "IPFS client built successfully, with version = {:?}",
                    version
                );
            }
            Err(e) => {
                error!("Failed to obtain IPFS client's version: {}", e);
            }
        }
        Ok(Self { client })
    }
}

impl IpfsInputManager {
    /// Fetch text from IPFS
    #[instrument(skip_all)]
    pub async fn fetch_text(&self, document_cid: &str) -> Result<String, AtomaInputManagerError> {
        let document = self
            .client
            .cat(document_cid)
            .map_ok(|chunk| chunk.to_vec())
            .try_concat()
            .await
            .map_err(|e| AtomaInputManagerError::IpfsError(e.to_string()))?;

        Ok(String::from_utf8_lossy(document.as_slice()).to_string())
    }

    /// Fetch JSON from IPFS
    #[instrument(skip_all)]
    pub async fn fetch_json(
        &self,
        document_cid: &str,
    ) -> Result<serde_json::Value, AtomaInputManagerError> {
        let document = self
            .client
            .cat(document_cid)
            .map_ok(|chunk| chunk.to_vec())
            .try_concat()
            .await
            .map_err(|e| AtomaInputManagerError::IpfsError(e.to_string()))?;

        Ok(serde_json::from_slice(document.as_slice())?)
    }

    /// Fetch image from IPFS
    #[instrument(skip_all)]
    pub async fn fetch_image(
        &self,
        document_cid: &str,
    ) -> Result<(Vec<u8>, usize, usize), AtomaInputManagerError> {
        let document = self
            .client
            .cat(document_cid)
            .map_ok(|chunk| chunk.to_vec())
            .try_concat()
            .await
            .map_err(|e| AtomaInputManagerError::IpfsError(e.to_string()))?;

        let mut height_bytes_buffer = [0; 4];
        height_bytes_buffer.copy_from_slice(&document[document.len() - 4..document.len()]);

        let mut width_bytes_buffer = [0; 4];
        width_bytes_buffer.copy_from_slice(&document[document.len() - 8..document.len() - 4]);

        let height = u32::from_be_bytes(height_bytes_buffer) as usize;
        let width = u32::from_be_bytes(width_bytes_buffer) as usize;

        let image_bytes = document[..document.len() - 8].to_vec();
        Ok((image_bytes, height, width))
    }
}
