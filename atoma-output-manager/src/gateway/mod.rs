use gql_client::GqlClient;
use tokio::sync::mpsc;

const GATEWAY_GRAPHQL_ENDPOINT: &str = "https://protocol.mygateway.xyz/graphql";

pub struct GatewayOutputManager {
    client: GqlClient,
    output_manager_rx: mpsc::Receiver<(Vec<u8>, Vec<u8>)>
}

impl GatewayOutputManager {
    pub fn new(api_key: &str, bearer_token: &str, output_manager_rx: mpsc::Receiver<(Vec<u8>, Vec<u8>)>) -> Self {
        let url = format!("{}/{}", GATEWAY_GRAPHQL_ENDPOINT, api_key);
        let mut headers = HashMap::new();
        headers.insert("authorization", format!("Bearer {}", bearer_token));
        let client = GqlClient::new_with_headers(url, headers);
        Self { client, output_manager_rx }
    }

    pub async fn run(mut self) -> Result<(), AtomaOutputManagerError> {
        info!("Starting firebase service..");
        while let Some(response) = self.output_manager_rx.recv().await {
            info!("Received a new output to be submitted to Firebase..");
            let tx_digest = response.0;
            let response = response.1;
            let data = serde_json::to_value(response)?;
            self.handle_request(tx_digest, data).await?;
        }
    }

    async fn handle_request(tx_digest: Vec<u8>, data: Value) -> 
}

