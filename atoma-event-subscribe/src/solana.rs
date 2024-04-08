use futures_util::StreamExt;
use solana_client::{
    nonblocking::pubsub_client::PubsubClient,
    rpc_config::{RpcTransactionLogsConfig, RpcTransactionLogsFilter},
    rpc_response::RpcLogsResponse,
};
use solana_sdk::commitment_config::CommitmentConfig;

pub async fn solana() -> Result<(), Box<dyn std::error::Error>> {
    let ws_url = "wss://api.mainnet-beta.solana.com";
    let ps_client = PubsubClient::new(ws_url).await?;
    // TODO: We should `Mentions` instead of `All` to filter what we need.
    let filter = RpcTransactionLogsFilter::All;
    let config = RpcTransactionLogsConfig {
        commitment: Some(CommitmentConfig::finalized()),
    };

    let (mut accounts, unsubscriber) = ps_client.logs_subscribe(filter, config).await?;

    while let Some(response) = accounts.next().await {
        let response: RpcLogsResponse = response.value;
        if response.err.is_none() {
            println!("{:?}", response.logs);
            println!();
        }
    }

    unsubscriber().await;

    Ok(())
}
