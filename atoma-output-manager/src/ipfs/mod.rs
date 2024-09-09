use atoma_types::AtomaOutputMetadata;
use futures::{stream::FuturesUnordered, StreamExt};
use ipfs_api_backend_hyper::{IpfsApi, IpfsClient};
use serde_json::json;
use tokio::sync::mpsc;
use tracing::{error, info, instrument, trace};

use crate::AtomaOutputManagerError;

type Output = Vec<u8>;

pub struct IpfsOutputManager {
    client: IpfsClient,
    ipfs_request_rx: mpsc::UnboundedReceiver<(AtomaOutputMetadata, Output)>,
}

impl IpfsOutputManager {
    /// Constructor
    #[instrument(skip_all)]
    pub async fn new(
        client: IpfsClient,
        ipfs_request_rx: mpsc::UnboundedReceiver<(AtomaOutputMetadata, Output)>,
    ) -> Result<Self, AtomaOutputManagerError> {
        Ok(Self {
            client,
            ipfs_request_rx,
        })
    }

    /// Runs a local IPFS node, to be able to communicate with the IPFS network
    #[instrument(skip_all)]
    pub async fn run(&mut self) -> Result<(), AtomaOutputManagerError> {
        info!("Running IPFS output manager...");
        let mut futures_unordered = FuturesUnordered::new();
        loop {
            tokio::select! {
                output = self.ipfs_request_rx.recv() => {
                    if let Some((output_metadata, output)) = output {
                        let future = Self::handle_request(&self.client, output_metadata, output);
                        futures_unordered.push(future);
                    }
                },
                result = futures_unordered.next() => {
                    if let Some(result) = result {
                        match result {
                            Ok(()) => {
                                trace!("Successfully stored output to IPFS");
                            }
                            Err(e) => {
                                error!("Failed to store output to IPFS, with error: {e}");
                            }
                        }
                    }
                }
            }
        }
    }
}

impl IpfsOutputManager {
    #[instrument(skip_all)]
    pub async fn handle_request(
        client: &IpfsClient,
        output_metadata: AtomaOutputMetadata,
        output: Vec<u8>,
    ) -> Result<(), AtomaOutputManagerError> {
        info!("Storing new data to IPFS for new request");
        let metadata_json = serde_json::to_value(output_metadata)?;
        let json_output = json!({
            "metadata": metadata_json,
            "output": output,
        });
        let cursor_output = std::io::Cursor::new(serde_json::to_vec(&json_output)?);
        match client.add(cursor_output).await {
            Ok(response) => {
                info!(
                    "Data stored to IPFS successfully, with hash = {:?}",
                    response.hash
                );
                Ok(())
            }
            Err(e) => {
                error!("Failed to store data to IPFS: {e}");
                Err(AtomaOutputManagerError::IpfsError(e))
            }
        }
    }
}
