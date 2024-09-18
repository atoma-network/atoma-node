use atoma_types::{InputFormat, ModelInput};
use futures::TryStreamExt;
use ipfs_api_backend_hyper::{IpfsApi, IpfsClient};
use tokio::sync::{mpsc, oneshot::Sender};
use tracing::{info, instrument};

use crate::AtomaInputManagerError;

type Cid = String;
type IpfsRequestReceiver = mpsc::UnboundedReceiver<(
    Cid,
    InputFormat,
    Sender<Result<ModelInput, AtomaInputManagerError>>,
)>;

/// IPFS input manager
pub struct IpfsInputManager {
    /// IPFS Hyper client
    client: IpfsClient,
    /// IPFS request receiver
    ipfs_request_rx: IpfsRequestReceiver,
}

impl IpfsInputManager {
    /// Constructor
    #[tracing::instrument(skip_all)]
    pub async fn new(
        client: IpfsClient,
        ipfs_request_rx: IpfsRequestReceiver,
    ) -> Result<Self, AtomaInputManagerError> {
        Ok(Self {
            client,
            ipfs_request_rx,
        })
    }

    /// Runs the IPFS input manager background task
    #[instrument(skip_all)]
    pub async fn run(mut self) -> Result<(), AtomaInputManagerError> {
        info!("Starting IPFS input manager background task...");
        while let Some((cid, format, sender)) = self.ipfs_request_rx.recv().await {
            let model_input_result = match format {
                InputFormat::Text => self.fetch_text(cid).await,
                InputFormat::Image => self.fetch_image(cid).await,
            };
            sender
                .send(model_input_result)
                .map_err(|_| AtomaInputManagerError::SendPromptError)?;
        }
        Ok(())
    }
}

impl IpfsInputManager {
    /// Fetch text from IPFS
    #[instrument(skip_all)]
    pub async fn fetch_text(
        &self,
        document_cid: String,
    ) -> Result<ModelInput, AtomaInputManagerError> {
        let document = self
            .client
            .cat(&document_cid)
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
        document_cid: String,
    ) -> Result<ModelInput, AtomaInputManagerError> {
        let document = self
            .client
            .cat(&document_cid)
            .map_ok(|chunk| chunk.to_vec())
            .try_concat()
            .await
            .map_err(|e| AtomaInputManagerError::IpfsError(e.to_string()))?;

        Ok(ModelInput::ImageBytes(document))
    }
}
