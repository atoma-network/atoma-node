use gql_client::GqlClient;
use tokio::sync::mpsc;
use tracing::info;

const GATEWAY_GRAPHQL_ENDPOINT: &str = "https://protocol.mygateway.xyz/graphql";

pub struct GatewayOutputManager {
    client: GqlClient,
    output_manager_rx: mpsc::Receiver<(Vec<u8>, Response)>
}

impl GatewayOutputManager {
    pub fn new(api_key: &str, bearer_token: &str, output_manager_rx: mpsc::Receiver<(Vec<u8>, Response)>) -> Self {
        let url = format!("{}", GATEWAY_GRAPHQL_ENDPOINT);
        let mut headers = HashMap::new();
        headers.insert("X-Api-Key", format!("{}", api_key));
        headers.insert("Authorization", format!("Bearer {}", bearer_token));
        let client = Client::new_with_headers(url, headers);
        info!("Created a new client for the gateway..");
        Self { client, output_manager_rx }
    }

    pub async fn run(mut self) -> Result<(), AtomaOutputManagerError> {
        info!("Starting firebase service..");
        while let Some((tx_digest, response)) = self.output_manager_rx.recv().await {
            info!("Received a new output to be submitted to Firebase..");
            self.handle_request(tx_digest, data).await?;
        }
    }

    async fn handle_request(tx_digest: Vec<u8>, response: Response) -> Result<(), AtomaOutputManagerError> {
        let ticket_id = response.ticket_id();
        let sampled_node_index = response.sampled_node_index();
        let num_sampled_nodes = response.num_sampled_nodes();
        let time_to_generate = response.time_to_generate();
        let query = format!(r#"mutation createPDA {{
            createPDA(
              input: {{
                title: "My Personal Data Asset"
                description: "Atoma Node output for {:?}"
                owner: {{ type: GATEWAY_ID, value: "{}" }}
                dataModelId: "d5011a1f-d6df-41ec-970f-36477e554dc2"
                expirationDate: null
                organization: {{ type: GATEWAY_ID, value: "AtomaNetwork" }}
                claim: {{
                  nodePublicKey: "{:?}",
                  ticketId: "{:?}",
                  outputText: "{:?}",
                  inputTokens: "{:?}",
                  outputTokens: "{:?}",
                  timeToGenerate: "{:?}",
                  commitmentRootHash: "{:?}",
                  numSampledNodes: "{:?}",
                  indexSubmissionNode: "{:?}",
                }
              }
            ) {
              id
              arweaveUrl
              dataAsset {
                owner {
                  id
                  gatewayId
                }
                issuer {
                  id
                  gatewayId
                }
              }
            }
          }"#, tx_digest, );
        Ok(())
    }
}

