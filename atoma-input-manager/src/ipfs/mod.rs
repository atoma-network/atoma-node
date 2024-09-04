use atoma_types::ModelInput;
use futures::TryStreamExt;
use ipfs_api_backend_hyper::{IpfsApi, IpfsClient, TryFromUri};
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

        let client = IpfsClient::from_multiaddr_str("https://ipfs.io/ipfs/")
            .map_err(|e| AtomaInputManagerError::FailedToBuildIpfsClient(e.to_string()))?;
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
    pub async fn fetch_text(
        &self,
        document_cid: &str,
    ) -> Result<ModelInput, AtomaInputManagerError> {
        let document = self
            .client
            .cat(document_cid)
            .map_ok(|chunk| chunk.to_vec())
            .try_concat()
            .await
            .map_err(|e| AtomaInputManagerError::IpfsError(e.to_string()))?;

        Ok(ModelInput::Text(
            String::from_utf8_lossy(document.as_slice()).to_string(),
        ))
    }

    /// Fetch image from IPFS
    #[instrument(skip_all)]
    pub async fn fetch_image(
        &self,
        document_cid: &str,
    ) -> Result<ModelInput, AtomaInputManagerError> {
        let document = self
            .client
            .cat(document_cid)
            .map_ok(|chunk| chunk.to_vec())
            .try_concat()
            .await
            .map_err(|e| AtomaInputManagerError::IpfsError(e.to_string()))?;

        Ok(ModelInput::ImageBytes(document))
    }
}
