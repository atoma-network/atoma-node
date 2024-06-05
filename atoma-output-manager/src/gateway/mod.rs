use std::collections::HashMap;

use atoma_types::AtomaOutputMetadata;
use gql_client::Client;
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::info;

use crate::AtomaOutputManagerError;

const GATEWAY_GRAPHQL_ENDPOINT: &str = "https://protocol.mygateway.xyz/graphql";

/// `GatewayOutputManager` - Wrapper around a GraphQL client to interact with the
///   Gateway protocol API.
///
/// It is used by nodes to submit generated outputs on behalf of a user request
/// (to a user's own PDA).
pub struct GatewayOutputManager {
    /// GraphQL client
    client: Client,
    /// A mpsc receiver of a tuple containing the output's metadata and actual content (in JSON format)
    output_manager_rx: mpsc::Receiver<(AtomaOutputMetadata, Value)>,
}

impl GatewayOutputManager {
    /// Constructor
    pub fn new(
        api_key: &str,
        bearer_token: &str,
        output_manager_rx: mpsc::Receiver<(AtomaOutputMetadata, Value)>,
    ) -> Self {
        let url = format!("{}", GATEWAY_GRAPHQL_ENDPOINT);
        let mut headers = HashMap::new();
        headers.insert("X-Api-Key", format!("{}", api_key));
        headers.insert("Authorization", format!("Bearer {}", bearer_token));
        let client = Client::new_with_headers(url, headers);
        info!("Created a new client for the gateway..");
        Self {
            client,
            output_manager_rx,
        }
    }

    /// Main loop, it listens to newly received responses from the Node's client.
    /// It then is responsible to send the request to Gateway's storage, via its API.
    pub async fn run(mut self) -> Result<(), AtomaOutputManagerError> {
        info!("Starting firebase service..");
        while let Some((output_metadata, output)) = self.output_manager_rx.recv().await {
            info!("Received a new output to be submitted to Firebase..");
            self.handle_request(output_metadata, output).await?;
        }
        Ok(())
    }

    async fn handle_request(
        &self,
        output_metadata: AtomaOutputMetadata,
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
        } = output_metadata;
        let query = format!(
            r#"mutation createPDA {{
            createPDA(
              input: {{
                title: "Atoma's output for ticket id: {:?}",
                description: "Atoma Node output for ticket id {:?}",
                owner: {{ type: GATEWAY_ID, value: "{}" }},
                dataModelId: "d5011a1f-d6df-41ec-970f-36477e554dc2",
                expirationDate: null,
                organization: {{ type: GATEWAY_ID, value: "AtomaNetwork" }},
                claim: {{
                  nodePublicKey: "{:?}",
                  ticketId: "{:?}",
                  output: "{:?}",
                  inputTokens: "{:?}",
                  outputTokens: "{:?}",
                  timeToGenerate: "{:?}",
                  commitmentRootHash: "{:?}",
                  numSampledNodes: "{:?}",
                  indexSubmissionNode: "{:?}",
                  leafHash: "{:?}",
                  transactionBase58: "{:?}"
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
          }}"#,
            ticket_id,
            ticket_id,
            gateway_user_id,
            node_public_key,
            ticket_id,
            output,
            num_input_tokens,
            num_output_tokens,
            time_to_generate,
            commitment_root_hash,
            num_sampled_nodes,
            index_of_node,
            leaf_hash,
            transaction_base_58
        );

        self.client
            .query::<String>(&query)
            .await
            .map_err(|e| AtomaOutputManagerError::GraphQlError(e.to_string()))?;

        Ok(())
    }
}
