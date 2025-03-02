use async_trait::async_trait;
use bytes::{BufMut, BytesMut};
use futures::io::{self, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use libp2p::request_response::Codec;
use serde::{Deserialize, Serialize};
use sui_sdk::types::crypto::Signature;

#[derive(Clone, Debug)]
pub struct StackLeaderProtocol {
    _version: u8,
}

impl Default for StackLeaderProtocol {
    fn default() -> Self {
        Self { _version: 1 }
    }
}

impl AsRef<str> for StackLeaderProtocol {
    fn as_ref(&self) -> &'static str {
        "/atoma/stack-leader/0.0.1"
    }
}

#[derive(Clone, Default)]
pub struct StackLeaderCodec();

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackSmallId(u64);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeSmallId(u64);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackAvailableComputeUnitsRequest {
    pub stack_small_id: StackSmallId,
    pub node_small_id: NodeSmallId,
    pub num_compute_units: u64,
    pub timestamp: u64,
    pub signature: Signature,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackAvailableComputeUnitsResponse {
    pub is_permissed: bool,
    pub stack_small_id: StackSmallId,
    pub timestamp: u64,
    pub signature: Signature,
    pub remaining_available_compute_units: u64,
}

#[async_trait]
impl Codec for StackLeaderCodec {
    type Request = StackAvailableComputeUnitsRequest;
    type Response = StackAvailableComputeUnitsResponse;
    type Protocol = StackLeaderProtocol;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        // Read the length prefix (4 bytes)
        let mut length_bytes = [0u8; 4];
        io.read_exact(&mut length_bytes).await?;
        let length = u32::from_be_bytes(length_bytes) as usize;

        // Read the serialized data
        let mut data = vec![0u8; length];
        io.read_exact(&mut data).await?;

        // Deserialize the request
        ciborium::de::from_reader(&data[..])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        // Read the length prefix (4 bytes)
        let mut length_bytes = [0u8; 4];
        io.read_exact(&mut length_bytes).await?;
        let length = u32::from_be_bytes(length_bytes) as usize;

        // Read the serialized data
        let mut data = vec![0u8; length];
        io.read_exact(&mut data).await?;

        // Deserialize the response
        ciborium::de::from_reader(&data[..])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        request: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let mut buffer = BytesMut::new();
        ciborium::into_writer(&request, (&mut buffer).writer())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let bytes = buffer.freeze();
        let length = u32::try_from(bytes.len())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        io.write_all(&length.to_be_bytes()).await?;
        io.write_all(&bytes).await?;
        Ok(())
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        response: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let mut buffer = BytesMut::new();
        ciborium::into_writer(&response, (&mut buffer).writer())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let bytes = buffer.freeze();
        let length = u32::try_from(bytes.len())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        io.write_all(&length.to_be_bytes()).await?;
        io.write_all(&bytes).await?;
        Ok(())
    }
}
