use std::{path::Path, str::FromStr};

use sui_sdk::{
    rpc_types::{SuiObjectDataOptions, SuiObjectResponseQuery, SuiTransactionBlockResponseOptions},
    types::{
        base_types::SuiAddress,
        programmable_transaction_builder::ProgrammableTransactionBuilder,
        quorum_driver_types::ExecuteTransactionRequestType,
        transaction::{Transaction, TransactionData},
    },
    SuiClient, SuiClientBuilder,
};
use thiserror::Error;
use tracing::info;

use crate::config::SuiConfig;

const AMOUNT: u64 = 100_000_000; // 0.1 SUI
const GAS_BUDGET: u64 = 5_000_000; // 0.005 SUI

pub struct AtomaSuiClient {
    client: SuiClient,
    atoma_contract_address: SuiAddress,
    address: SuiAddress,
}

impl AtomaSuiClient {
    pub async fn new(
        address: &str,
        atoma_contract_address: &str,
        http_url: &str,
        ws_url: Option<&str>,
    ) -> Result<Self, AtomaSuiClientError> {
        let sui_client_builder = SuiClientBuilder::default();
        info!("Starting sui client..");
        let client = sui_client_builder.build(http_url).await?;
        Ok(Self {
            client,
            address: SuiAddress::from_str(address)?,
            atoma_contract_address: SuiAddress::from_str(atoma_contract_address)?,
        })
    }

    pub async fn new_from_config<P: AsRef<Path>>(
        config_path: P,
    ) -> Result<Self, AtomaSuiClientError> {
        let config = SuiConfig::from_file_path(config_path);
        let address = config.address();
        let atoma_contract_address = config.atoma_contract_address();
        let http_url = config.http_addr();
        let ws_url = config.ws_addr();
        Self::new(&address, &atoma_contract_address, &http_url, Some(&ws_url)).await
    }

    pub async fn sign_transaction(
        &self,
        response: serde_json::Value,
    ) -> Result<(), AtomaSuiClientError> {
        let coins_response = self
            .client
            .read_api()
            .get_owned_objects(
                self.address,
                Some(SuiObjectResponseQuery::new_with_options(
                    SuiObjectDataOptions::new().with_type(),
                )),
                None,
                None,
            )
            .await?;
        info!("Wallet coins: {:?}", coins_response);

        let coin = coins_response
            .data
            .iter()
            .find(|obj| obj.data.as_ref().unwrap().is_gas_coin())
            .ok_or(AtomaSuiClientError::NoAvailableFunds)?;
        let coin = coin.data.as_ref().unwrap();
        info!("Wallet SUI: {:?}", coin);

        let balance = self
            .client
            .coin_read_api()
            .get_coins(self.address, None, None, None)
            .await?;
        let coin_balance = balance.data.into_iter().next().unwrap();
        info!("SUI balance: {:?}", coin_balance);

        let pt = {
            let mut builder = ProgrammableTransactionBuilder::new();
            // TODO: we don't want to pay SUI, only submit the commitment
            builder.pay_sui(vec![self.atoma_contract_address], vec![AMOUNT])?;
            builder.finish()
        };

        let gas_price = self.client.read_api().get_reference_gas_price().await?;

        // Create the transaction data that will be sent to the network
        let tx_data = TransactionData::new_programmable(
            self.address,
            vec![coin.object_ref()],
            pt,
            GAS_BUDGET,
            gas_price,
        );

        let signature = keystore.sign_secure(&self.address, &tx_data, Intent::sui_transaction())?;

        info!("\nExecuting transaction...");
        let transaction_response = self
            .client
            .quorum_driver_api()
            .execute_transaction_block(
                Transaction::from_data(tx_data, vec![signature]),
                SuiTransactionBlockResponseOptions::full_content(),
                Some(ExecuteTransactionRequestType::WaitForLocalExecution),
            )
            .await?;
        info!(
            "done!\n\nTransaction information: {:?}",
            transaction_response
        );
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum AtomaSuiClientError {
    #[error("Sui Builder error: `{0}`")]
    SuiBuilderError(#[from] sui_sdk::error::Error),
    #[error("Failed to parse address: `{0}`")]
    FailedParseAddress(#[from] anyhow::Error),
    #[error("No available funds")]
    NoAvailableFunds,
}
