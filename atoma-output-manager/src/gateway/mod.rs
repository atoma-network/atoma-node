use std::collections::HashMap;

use atoma_types::{AtomaOutputMetadata, OutputDestination};
use gql_client::Client;
use serde_json::Value;
use tracing::info;

use crate::AtomaOutputManagerError;

const GATEWAY_GRAPHQL_ENDPOINT: &str = "https://protocol.mygateway.xyz/graphql";

/// Gateway

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
    pub async fn handle_request(
        &self,
        output_metadata: &AtomaOutputMetadata,
        output: Value,
    ) -> Result<(), AtomaOutputManagerError> {
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
        } = output_metadata;

        let gateway_user_id =
            if let OutputDestination::Gateway { gateway_user_id } = output_destination {
                gateway_user_id
            } else {
                return Err(AtomaOutputManagerError::InvalidOutputDestiny(
                    "Missing `gateway_user_id` from `OutputDestinty".into(),
                ));
            };
        let query = format!(
            r#"mutation createPDA {{
            createPDA(
              input: {{
                title: "Atoma's output for ticket id: {ticket_id}",
                description: "Atoma Node output for ticket id {ticket_id}",
                owner: {{ type: GATEWAY_ID, value: "{gateway_user_id}" }},
                dataModelId: "d5011a1f-d6df-41ec-970f-36477e554dc2",
                expirationDate: null,
                organization: {{ type: GATEWAY_ID, value: "AtomaNetwork" }},
                claim: {{
                  nodePublicKey: "{node_public_key}",
                  ticketId: "{ticket_id}",
                  output: "{output}",
                  inputTokens: "{num_input_tokens}",
                  outputTokens: "{num_output_tokens}",
                  timeToGenerate: "{time_to_generate}",
                  commitmentRootHash: "{commitment_root_hash:?}",
                  numSampledNodes: "{num_sampled_nodes}",
                  indexSubmissionNode: "{index_of_node}",
                  leafHash: "{leaf_hash:?}",
                  transactionBase58: "{transaction_base_58}"
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
        );

        self.client
            .query::<String>(&query)
            .await
            .map_err(|e| AtomaOutputManagerError::GraphQlError(e.to_string()))?;

        Ok(())
    }
}
