use std::{path::Path, string::FromUtf8Error};

use atoma_helpers::Firebase;
#[cfg(feature = "supabase")]
use atoma_helpers::Supabase;
use atoma_types::{AtomaOutputMetadata, OutputDestination};
use config::AtomaOutputManagerConfig;
use firebase::FirebaseOutputManager;
use gateway::GatewayOutputManager;
use http::uri::Scheme;
use ipfs::IpfsOutputManager;
use ipfs_api_backend_hyper::{IpfsApi, IpfsClient, TryFromUri};
use thiserror::Error;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tracing::{error, info, instrument, trace};

mod config;
mod firebase;
mod gateway;
mod ipfs;
#[cfg(feature = "supabase")]
mod supabase;
#[cfg(feature = "supabase")]
use supabase::SupabaseOutputManager;

type ChatId = String;
type IpfsSendRequestError = mpsc::error::SendError<(
    AtomaOutputMetadata,
    Vec<u8>,
    oneshot::Sender<Option<String>>,
)>;
type OutputData = (ChatId, String, usize, usize, f64);

/// `AtomaOutputManager` - manages different output destination
///     requests, allowing for a flexible interaction between
///     the user and the Atoma Network.
///
/// Current available options for storing the output on behalf
/// of the user consists of Firebase and the Gateway protocol.
pub struct AtomaOutputManager {
    /// Firebase's output manager instance.
    firebase_output_manager: FirebaseOutputManager,
    /// IPFS's join handle related to the IPFS output manager background task.
    ipfs_join_handle: Option<JoinHandle<Result<(), AtomaOutputManagerError>>>,
    /// Request sender to the IPFS manager background task.
    ipfs_request_tx: mpsc::UnboundedSender<(
        AtomaOutputMetadata,
        Vec<u8>,
        oneshot::Sender<Option<String>>,
    )>,
    /// Gateway's output manager.
    gateway_output_manager: GatewayOutputManager,
    /// A mpsc receiver that receives tuples of `AtomaOutputMetadata` and
    /// the actual AI generated output, in JSON format.
    output_manager_receiver: mpsc::Receiver<(AtomaOutputMetadata, Vec<u8>)>,
    /// Chat session data receiver
    chat_message_receiver: mpsc::Receiver<OutputData>,
    #[cfg(feature = "supabase")]
    supabase_output_manager: SupabaseOutputManager,
}

impl AtomaOutputManager {
    /// Constructor
    pub async fn new<P: AsRef<Path>>(
        config_file_path: P,
        output_manager_receiver: mpsc::Receiver<(AtomaOutputMetadata, Vec<u8>)>,
        firebase: Firebase,
        chat_message_receiver: mpsc::Receiver<OutputData>,
        #[cfg(feature = "supabase")] supabase: Supabase,
    ) -> Result<Self, AtomaOutputManagerError> {
        let config = AtomaOutputManagerConfig::from_file_path(config_file_path);
        let firebase_output_manager = FirebaseOutputManager::new(firebase);
        let gateway_output_manager =
            GatewayOutputManager::new(&config.gateway_api_key, &config.gateway_bearer_token);

        info!("Building IPFS client...");
        let ipfs_host = config.ipfs_host.unwrap_or("localhost".to_string());
        let ipfs_port = config.ipfs_port;

        let client = IpfsClient::from_host_and_port(Scheme::HTTP, &ipfs_host, ipfs_port)
            .map_err(|e| AtomaOutputManagerError::FailedToBuildIpfsClient(e.to_string()))?;
        let (ipfs_request_tx, ipfs_request_rx) = mpsc::unbounded_channel();

        let ipfs_join_handle = match client.version().await {
            Ok(version) => {
                info!(
                    "IPFS client built successfully, with version = {:?}",
                    version
                );
                let ipfs_join_handle = tokio::spawn(async move {
                    let mut ipfs_output_manager =
                        IpfsOutputManager::new(client, ipfs_request_rx).await?;
                    ipfs_output_manager.run().await?;
                    Ok(())
                });
                Some(ipfs_join_handle)
            }
            Err(e) => {
                #[cfg(target_os = "windows")]
                error!(
                    "Failed to obtain IPFS client's version: {e}, most likely IPFS daemon is not running in the background. To start it, install IPFS daemon on your machine and run `ipfs daemon`."
                );
                #[cfg(not(target_os = "windows"))]
                error!(
                    "Failed to obtain IPFS client's version: {e}, most likely IPFS daemon is not running in the background. To start it, instal the IPFS daemon on your machine and run `$ ipfs daemon`."
                );
                None
            }
        };
        #[cfg(feature = "supabase")]
        let supabase_output_manager = SupabaseOutputManager::new(supabase);

        Ok(Self {
            firebase_output_manager,
            ipfs_join_handle,
            ipfs_request_tx,
            gateway_output_manager,
            output_manager_receiver,
            chat_message_receiver,
            #[cfg(feature = "supabase")]
            supabase_output_manager,
        })
    }

    /// Main loop, responsible for continuously listening to incoming
    /// AI generated outputs, together with corresponding metadata
    #[instrument(skip_all)]
    pub async fn run(mut self) -> Result<(), AtomaOutputManagerError> {
        trace!("Starting firebase service..");
        loop {
            tokio::select! {
                Some((output_metadata, output)) = self.output_manager_receiver.recv() => {
                    match output_metadata.output_destination {
                        OutputDestination::Firebase { .. } => {
                            self.firebase_output_manager
                                .handle_post_request(&output_metadata, output, None)
                                .await?
                        }
                        OutputDestination::Ipfs { .. } => {
                            let (sender, receiver) = oneshot::channel();
                            self.ipfs_request_tx
                                .send((output_metadata.clone(), output.clone(), sender))
                                .map_err(Box::new)?;
                            let ipfs_cid = match receiver.await {
                                Ok(response) => response,
                                Err(e) => {
                                    error!("Failed to receive IPFS response: {:?}", e);
                                    None
                                }
                            };
                            self.firebase_output_manager
                                .handle_post_request(&output_metadata, output.clone(), ipfs_cid)
                                .await?;
                        }
                        OutputDestination::Gateway { .. } => {
                            self.gateway_output_manager
                                .handle_request(&output_metadata, output)
                                .await?
                        }
                        #[cfg(feature = "supabase")]
                        OutputDestination::Supabase { .. } => {
                            self.supabase_output_manager
                                .handle_post_request(&output_metadata, output)
                                .await?
                        }
                    }
                },
                Some((chat_id, message, num_input_tokens, num_output_tokens, elapsed_time)) = self.chat_message_receiver.recv() => {
                    #[cfg(feature = "supabase")]
                    {
                        trace!("Handling chat message on Supabase, for chat_id = {chat_id}...");
                        self.supabase_output_manager.handle_chat_message(
                            &chat_id,
                            message,
                            num_input_tokens,
                            num_output_tokens,
                            elapsed_time
                        )
                        .await?;
                    }
                    #[cfg(not(feature = "supabase"))]
                    {
                        error!(
                            "Supabase is not enabled, but chat message was received with values: chat_id = {}, message = {}, num_input_tokens = {}, num_output_tokens = {}, elapsed_time = {}",
                            chat_id, message, num_input_tokens, num_output_tokens, elapsed_time
                        );
                    }
                }
            }
        }
    }

    /// Graceful shutdown
    #[instrument(skip_all)]
    pub async fn stop(self) -> Result<(), AtomaOutputManagerError> {
        info!("Stopping IPFS manager...");

        trace!("Dropping IPFS request tx...");
        drop(self.ipfs_request_tx);

        trace!("Aborting IPFS manager join handle...");
        if let Some(handle) = self.ipfs_join_handle {
            handle.abort();

            trace!("Waiting for IPFS manager to join...");
            match handle.await {
                Ok(_) => Ok(()),
                Err(e) if e.is_cancelled() => Ok(()),
                Err(e) => Err(AtomaOutputManagerError::JoinError(e)),
            }?;
        }

        trace!("Dropping output manager receiver...");
        drop(self.output_manager_receiver);

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum AtomaOutputManagerError {
    #[error("Deserialize JSON value error: `{0}`")]
    DeserializeError(#[from] serde_json::Error),
    #[error("Request error: `{0}`")]
    RequestError(#[from] reqwest::Error),
    #[error("GraphQl error: `{0}`")]
    GraphQlError(String),
    #[error("Invalid output destination: `{0}`")]
    InvalidOutputDestination(String),
    #[error("Failed to convert output to string: `{0}`")]
    FromUtf8Error(#[from] FromUtf8Error),
    #[error("Failed to build IPFS client: `{0}`")]
    FailedToBuildIpfsClient(String),
    #[error("Firebase authentication error: `{0}`")]
    FirebaseAuthError(#[from] atoma_helpers::FirebaseAuthError),
    #[error("Url error: `{0}`")]
    UrlError(String),
    #[error("Url parse error: `{0}`")]
    UrlParseError(#[from] url::ParseError),
    #[error("IPFS send request error: `{0}`")]
    IpfsSendRequestError(#[from] Box<IpfsSendRequestError>),
    #[error("Unknown IPFS error: `{0}`")]
    IpfsError(#[from] ipfs_api_backend_hyper::Error),
    #[error("Join error: `{0}`")]
    JoinError(#[from] tokio::task::JoinError),
    #[error("Firebase error: `{0}`")]
    FirebaseError(String),
    #[error("Failed to send the ipfs result `{0:?}` back")]
    SendError(Option<String>),
    #[cfg(feature = "supabase")]
    #[error("Supabase error: `{0}`")]
    SupabaseError(#[from] atoma_helpers::SupabaseError),
}
