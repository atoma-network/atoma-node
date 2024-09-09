use std::collections::HashMap;

use atoma_types::{AtomaOutputMetadata, OutputDestination, OutputType};
use gql_client::Client;
use tracing::{info, instrument};

use crate::AtomaOutputManagerError;

const GATEWAY_GRAPHQL_ENDPOINT: &str = "https://protocol.mygateway.xyz/graphql";
const ATOMA_DATA_MODEL_ID: &str = "2a27c67f-64bc-4b23-8121-72c57e5dbb2f";

/// `GatewayOutputManager` - Wrapper around a GraphQL client to interact with the
///   Gateway protocol API.
///
/// It is used by nodes to submit generated outputs on behalf of a user request
/// (to a user's own PDA).
pub struct GatewayOutputManager {
    /// GraphQL client
    client: Client,
}

impl GatewayOutputManager {
    /// Constructor
    pub fn new(api_key: &str, bearer_token: &str) -> Self {
        let url = GATEWAY_GRAPHQL_ENDPOINT.to_string();
        let mut headers = HashMap::new();
        headers.insert("X-Api-Key", api_key.to_string());
        headers.insert("Authorization", format!("Bearer {}", bearer_token));
        let client = Client::new_with_headers(url, headers);
        info!("Created a new client for the gateway..");
        Self { client }
    }

    /// Handles a new request by submitting the inference output to the user
    /// specified PDA, on behalf of the Atoma organization on the Gateway protocol
    #[instrument(skip_all)]
    pub async fn handle_request(
        &self,
        output_metadata: &AtomaOutputMetadata,
        output: Vec<u8>,
    ) -> Result<(), AtomaOutputManagerError> {
        let query = match output_metadata.output_type {
            OutputType::Text => {
                let output = String::from_utf8(output)?;
                build_text_query(output, output_metadata)?
            }
            OutputType::Image => {
                let mut height_bytes_buffer = [0; 4];
                height_bytes_buffer.copy_from_slice(&output[output.len() - 4..output.len()]);

                let mut width_bytes_buffer = [0; 4];
                width_bytes_buffer.copy_from_slice(&output[output.len() - 8..output.len() - 4]);

                let height = u32::from_be_bytes(height_bytes_buffer) as usize;
                let width = u32::from_be_bytes(width_bytes_buffer) as usize;

                let output = output[..output.len() - 8].to_vec();
                let img_output = (output, height, width);
                build_image_query(img_output, output_metadata)?
            }
        };

        self.client
            .query::<String>(&query)
            .await
            .map_err(|e| AtomaOutputManagerError::GraphQlError(e.to_string()))?;

        Ok(())
    }
}

fn build_text_query(
    output: String,
    output_metadata: &AtomaOutputMetadata,
) -> Result<String, AtomaOutputManagerError> {
    let AtomaOutputMetadata {
        node_public_key,
        ticket_id,
        num_input_tokens,
        num_output_tokens,
        time_to_generate,
        commitment_root_hash,
        num_sampled_nodes,
        index_of_node,
        leaf_hash,
        transaction_base_58,
        output_destination,
        output_type,
        ..
    } = output_metadata;

    let gateway_user_id = if let OutputDestination::Gateway { gateway_user_id } = output_destination
    {
        gateway_user_id
    } else {
        return Err(AtomaOutputManagerError::InvalidOutputDestination(
            "Missing `gateway_user_id` from `OutputDestination".into(),
        ));
    };
    Ok(format!(
        r#"mutation createPDA {{
      createPDA(
        input: {{
          title: "Atoma's output for ticket id: {ticket_id}",
          description: "Atoma Node output for ticket id {ticket_id}",
          owner: {{ type: GATEWAY_ID, value: "{gateway_user_id}" }},
          dataModelId: "{ATOMA_DATA_MODEL_ID}",
          expirationDate: null,
          organization: {{ type: GATEWAY_ID, value: "AtomaNetwork" }},
          claim: {{
            nodePublicKey: "{node_public_key}",
            ticketId: "{ticket_id}",
            output: "{output}",
            inputTokens: {num_input_tokens},
            outputTokens: {num_output_tokens},
            timeToGenerate: {time_to_generate},
            commitmentRootHash: "{commitment_root_hash:?}",
            numSampledNodes: {num_sampled_nodes},
            indexSubmissionNode: {index_of_node},
            leafHash: "{leaf_hash:?}",
            transactionBase58: "{transaction_base_58}",
            outputType: "{output_type}"
          }}
        }}
      ) {{
        id
        arweaveUrl
        dataAsset {{
          owner {{
            id
            gatewayId
          }}
          issuer {{
            id
            gatewayId
          }}
        }}
      }}
    }}"#
    ))
}

fn build_image_query(
    output: (Vec<u8>, usize, usize),
    output_metadata: &AtomaOutputMetadata,
) -> Result<String, AtomaOutputManagerError> {
    let AtomaOutputMetadata {
        node_public_key,
        ticket_id,
        num_input_tokens,
        num_output_tokens,
        time_to_generate,
        commitment_root_hash,
        num_sampled_nodes,
        index_of_node,
        leaf_hash,
        transaction_base_58,
        output_destination,
        output_type,
        ..
    } = output_metadata;

    let gateway_user_id = if let OutputDestination::Gateway { gateway_user_id } = output_destination
    {
        gateway_user_id
    } else {
        return Err(AtomaOutputManagerError::InvalidOutputDestination(
            "Missing `gateway_user_id` from `OutputDestination".into(),
        ));
    };
    Ok(format!(
        r#"mutation createPDA {{
      createPDA(
        input: {{
          title: "Atoma's output for ticket id: {ticket_id}",
          description: "Atoma Node output for ticket id {ticket_id}",
          owner: {{ type: GATEWAY_ID, value: "{gateway_user_id}" }},
          dataModelId: "{ATOMA_DATA_MODEL_ID}",
          expirationDate: null,
          organization: {{ type: GATEWAY_ID, value: "AtomaNetwork" }},
          claim: {{
            nodePublicKey: "{node_public_key}",
            ticketId: "{ticket_id}",
            output: "{output:?}",
            inputTokens: {num_input_tokens},
            outputTokens: {num_output_tokens},
            timeToGenerate: {time_to_generate},
            commitmentRootHash: "{commitment_root_hash:?}",
            numSampledNodes: {num_sampled_nodes},
            indexSubmissionNode: {index_of_node},
            leafHash: "{leaf_hash:?}",
            transactionBase58: "{transaction_base_58}",
            outputType: "{output_type}"
          }}
        }}
      ) {{
        id
        arweaveUrl
        dataAsset {{
          owner {{
            id
            gatewayId
          }}
          issuer {{
            id
            gatewayId
          }}
        }}
      }}
    }}"#
    ))
}
