use atoma_types::AtomaOutputMetadata;
use ipfs_api_backend_hyper::{IpfsApi, IpfsClient};
use serde_json::json;
use tracing::{error, info, instrument};

use crate::AtomaOutputManagerError;
pub struct IpfsOutputManager {
    client: IpfsClient,
}

impl IpfsOutputManager {
    /// Constructor
    #[instrument(skip_all)]
    pub async fn new() -> Result<Self, AtomaOutputManagerError> {
        info!("Building IPFS client...");
        let client = IpfsClient::default();
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

impl IpfsOutputManager {
    #[instrument(skip_all)]
    pub async fn handle_request(
        &self,
        output_metadata: &AtomaOutputMetadata,
        output: Vec<u8>,
    ) -> Result<(), AtomaOutputManagerError> {
        info!("Storing new data to IPFS for new request");
        let metadata_json = serde_json::to_value(output_metadata)?;
        let json_output = json!({
            "metadata": metadata_json,
            "output": output,
        });
        let cursor_output = std::io::Cursor::new(serde_json::to_vec(&json_output)?);
        match self.client.add(cursor_output).await {
            Ok(response) => {
                info!(
                    "Data stored to IPFS successfully, with hash = {:?}",
                    response.hash
                );
                Ok(())
            }
            Err(e) => {
                error!("Failed to store data to IPFS: {}", e);
                Err(AtomaOutputManagerError::IpfsError(e.to_string()))
            }
        }
    }
}
