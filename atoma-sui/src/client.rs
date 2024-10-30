use std::path::Path;
use sui_sdk::{
    json::SuiJsonValue, rpc_types::SuiData, types::base_types::ObjectID,
    wallet_context::WalletContext,
};
use thiserror::Error;
use tracing::{error, info, instrument};

use crate::config::AtomaSuiConfig;

type Result<T> = std::result::Result<T, AtomaSuiClientError>;

/// The gas budget for the node registration transaction
const GAS_BUDGET: u64 = 5_000_000; // 0.005 SUI

/// The Atoma's contract module name
const MODULE_ID: &str = "db";
/// The Atoma's contract method name for node registration
const NODE_REGISTRATION_METHOD: &str = "register_node_entry";
/// The Atoma's contract method name for node model subscription
const NODE_MODEL_SUBSCRIPTION_METHOD: &str = "add_node_to_model";
/// The Atoma's contract method name for node task subscription
const NODE_TASK_SUBSCRIPTION_METHOD: &str = "subscribe_node_to_task";
/// The Atoma's contract method name for node task unsubscription
const NODE_TASK_UNSUBSCRIPTION_METHOD: &str = "unsubscribe_node_from_task";
/// The Atoma's contract method name for trying to settle a stack
const TRY_SETLE_STACK_METHOD: &str = "try_settle_stack";
/// The Atoma's contract method name for stack settlement attestation
const STACK_SETTLEMENT_ATTESTATION_METHOD: &str = "submit_stack_settlement_attestation";
/// The Atoma's contract method name for starting an attestation dispute
const START_ATTESTATION_DISPUTE_METHOD: &str = "start_attestation_dispute";
/// The Atoma's contract method name for claiming funds
const CLAIM_FUNDS_METHOD: &str = "claim_funds";

/// A client for interacting with the Atoma network using the Sui blockchain.
///
/// The `AtomaSuiClient` struct provides methods to perform various operations
/// in the Atoma network, such as registering nodes, subscribing to models and tasks,
/// and managing transactions. It maintains a wallet context and optionally stores
/// a node badge representing the client's node registration status.
pub struct AtomaSuiClient {
    /// Configuration settings for the Atoma client, including paths and timeouts.
    config: AtomaSuiConfig,
    /// The wallet context used for managing blockchain interactions.
    wallet_ctx: WalletContext,
    /// An optional tuple containing the ObjectID and small ID of the node badge,
    /// which represents the node's registration in the Atoma network.
    node_badge: Option<(ObjectID, u64)>,
}

impl AtomaSuiClient {
    /// Constructor
    pub async fn new(config: AtomaSuiConfig) -> Result<Self> {
        let sui_config_path = config.sui_config_path();
        let sui_config_path = Path::new(&sui_config_path);
        let mut wallet_ctx = WalletContext::new(
            sui_config_path,
            config.request_timeout(),
            config.max_concurrent_requests(),
        )?;
        let node_badge = utils::get_node_badge(
            &wallet_ctx.get_client().await?,
            config.atoma_package_id(),
            wallet_ctx.active_address()?,
        )
        .await;
        Ok(Self {
            config,
            wallet_ctx,
            node_badge,
        })
    }

    /// Creates a new `AtomaSuiClient` instance from a configuration file.
    ///
    /// This method reads the configuration from the specified file path and initializes
    /// a new `AtomaSuiClient` with the loaded configuration.
    ///
    /// # Arguments
    ///
    /// * `config_path` - A path-like type that represents the location of the configuration file.
    ///
    /// # Returns
    ///
    /// * `Result<Self>` - A Result containing the new `AtomaSuiClient` instance if successful,
    ///   or an error if the configuration couldn't be read.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// * The configuration file cannot be read or parsed.
    pub async fn new_from_config<P: AsRef<Path>>(config_path: P) -> Result<Self> {
        let config = AtomaSuiConfig::from_file_path(config_path);
        Self::new(config).await
    }

    /// Submits a transaction to register a node in the Atoma network.
    ///
    /// This method creates and submits a transaction that registers the current wallet address
    /// as a node in the Atoma network. Upon successful registration, a node badge is created
    /// and stored in the client state.
    ///
    /// # Arguments
    ///
    /// * `gas` - Optional ObjectID to use as gas for the transaction. If None, the system will
    ///           automatically select a gas object.
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to GAS_BUDGET
    ///                  constant (5,000,000 MIST = 0.005 SUI).
    /// * `gas_price` - Optional gas price for the transaction. If None, uses the network's
    ///                 reference gas price.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the registration is successful, or an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - The node badge cannot be found after registration
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use sui_sdk::types::base_types::ObjectID;
    /// # async fn example(client: &mut AtomaSuiClient) -> Result<()> {
    /// // Register with default gas settings
    /// client.submit_node_registration_tx(None, None, None).await?;
    ///
    /// // Or with custom gas settings
    /// let gas_object = ObjectID::new([1; 32]);
    /// client.submit_node_registration_tx(
    ///     Some(gas_object),
    ///     Some(10_000_000),  // 0.01 SUI
    ///     Some(1000)
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(level = "info", skip_all, fields(
        address = %self.wallet_ctx.active_address().unwrap()
    ))]
    pub async fn submit_node_registration_tx(
        &mut self,
        gas: Option<ObjectID>,
        gas_budget: Option<u64>,
        gas_price: Option<u64>,
    ) -> Result<String> {
        let client = self.wallet_ctx.get_client().await?;
        let active_address = self.wallet_ctx.active_address()?;
        let tx = client
            .transaction_builder()
            .move_call(
                active_address,
                self.config.atoma_package_id(),
                MODULE_ID,
                NODE_REGISTRATION_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::from_object_id(self.config.toma_package_id()),
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;
        info!("Submitting node registration transaction...");
        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_must_succeed(tx).await;

        info!(
            "Node registration transaction submitted successfully. Transaction digest: {:?}",
            response.digest
        );

        let created_object = utils::get_node_badge(
            &client,
            self.config.atoma_package_id(),
            self.wallet_ctx.active_address()?,
        )
        .await
        .ok_or(AtomaSuiClientError::FailedToFindNodeBadge)?;
        self.node_badge = Some(created_object);

        Ok(response.digest.to_string())
    }

    /// Submits a transaction to subscribe a node to a specific model in the Atoma network.
    ///
    /// This method creates and submits a transaction that subscribes a node to a model with
    /// a specified echelon level. The node must have a valid node badge to perform this operation.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to subscribe to
    /// * `echelon` - The echelon level for the subscription (0-255)
    /// * `node_badge_id` - Optional ObjectID of the node badge. If None, uses the client's stored badge
    /// * `gas` - Optional ObjectID to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to GAS_BUDGET
    /// * `gas_price` - Optional gas price for the transaction. If None, uses network's reference price
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the subscription is successful, or an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use sui_sdk::types::base_types::ObjectID;
    /// # async fn example(client: &mut AtomaSuiClient) -> Result<()> {
    /// // Subscribe to model with default gas settings
    /// client.submit_node_model_subscription_tx(
    ///     "my_model",
    ///     1,
    ///     None,
    ///     None,
    ///     None,
    ///     None
    /// ).await?;
    ///
    /// // Or with custom gas settings and specific node badge
    /// let node_badge = ObjectID::new([1; 32]);
    /// let gas_object = ObjectID::new([2; 32]);
    /// client.submit_node_model_subscription_tx(
    ///     "my_model",
    ///     1,
    ///     Some(node_badge),
    ///     Some(gas_object),
    ///     Some(10_000_000),  // 0.01 SUI
    ///     Some(1000)
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(level = "info", skip_all, fields(
        model_name = %model_name,
        echelon = %echelon,
        address = %self.wallet_ctx.active_address().unwrap()
    ))]
    pub async fn submit_node_model_subscription_tx(
        &mut self,
        model_name: &str,
        echelon: u64,
        node_badge_id: Option<ObjectID>,
        gas: Option<ObjectID>,
        gas_budget: Option<u64>,
        gas_price: Option<u64>,
    ) -> Result<String> {
        let client = self.wallet_ctx.get_client().await?;
        let active_address = self.wallet_ctx.active_address()?;
        let node_badge_id = node_badge_id.unwrap_or(
            self.node_badge
                .as_ref()
                .ok_or(AtomaSuiClientError::FailedToFindNodeBadge)?
                .0,
        );
        let tx = client
            .transaction_builder()
            .move_call(
                active_address,
                self.config.atoma_package_id(),
                MODULE_ID,
                NODE_MODEL_SUBSCRIPTION_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::from_object_id(node_badge_id),
                    SuiJsonValue::new(model_name.into())?,
                    SuiJsonValue::new(echelon.to_string().into())?,
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;
        info!("Submitting model subscription transaction...");
        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_must_succeed(tx).await;

        info!(
            "Node model subscription transaction submitted successfully. Transaction digest: {:?}",
            response.digest
        );

        Ok(response.digest.to_string())
    }

    /// Submits a transaction to subscribe a node to a specific task in the Atoma network.
    ///
    /// This method creates and submits a transaction that subscribes a node to a task with
    /// specified pricing parameters. The node must have a valid node badge to perform this operation.
    ///
    /// # Arguments
    ///
    /// * `task_small_id` - The small ID of the task to subscribe to
    /// * `node_small_id` - Optional small ID of the node. If None, uses the client's stored badge ID
    /// * `price_per_compute_unit` - The price per compute unit the node is willing to charge
    /// * `max_num_compute_units` - Maximum number of compute units the node is willing to provide
    /// * `gas` - Optional ObjectID to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to GAS_BUDGET
    /// * `gas_price` - Optional gas price for the transaction. If None, uses network's reference price
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the subscription is successful, or an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use sui_sdk::types::base_types::ObjectID;
    /// # async fn example(client: &mut AtomaSuiClient) -> Result<()> {
    /// // Subscribe to task with default gas settings
    /// client.submit_node_task_subscription_tx(
    ///     123,                    // task_small_id
    ///     None,                   // use stored node_small_id
    ///     1000,                   // price per compute unit
    ///     5000,                   // max compute units
    ///     None,                   // default gas
    ///     None,                   // default gas budget
    ///     None                    // default gas price
    /// ).await?;
    ///
    /// // Or with custom gas settings and specific node ID
    /// let gas_object = ObjectID::new([1; 32]);
    /// client.submit_node_task_subscription_tx(
    ///     123,                    // task_small_id
    ///     Some(456),              // specific node_small_id
    ///     1000,                   // price per compute unit
    ///     5000,                   // max compute units
    ///     Some(gas_object),       // specific gas object
    ///     Some(10_000_000),       // 0.01 SUI gas budget
    ///     Some(1000)              // specific gas price
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(level = "info", skip_all, fields(
        address = %self.wallet_ctx.active_address().unwrap(),
        price_per_compute_unit = %price_per_compute_unit,
        max_num_compute_units = %max_num_compute_units,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_node_task_subscription_tx(
        &mut self,
        task_small_id: u64,
        node_small_id: Option<u64>,
        price_per_compute_unit: u64,
        max_num_compute_units: u64,
        gas: Option<ObjectID>,
        gas_budget: Option<u64>,
        gas_price: Option<u64>,
    ) -> Result<String> {
        let client = self.wallet_ctx.get_client().await?;
        let active_address = self.wallet_ctx.active_address()?;
        let node_small_id = node_small_id.unwrap_or(
            self.node_badge
                .as_ref()
                .ok_or(AtomaSuiClientError::FailedToFindNodeBadge)?
                .1,
        );
        let tx = client
            .transaction_builder()
            .move_call(
                active_address,
                self.config.atoma_package_id(),
                MODULE_ID,
                NODE_TASK_SUBSCRIPTION_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::new(node_small_id.into())?,
                    SuiJsonValue::new(task_small_id.into())?,
                    SuiJsonValue::new(price_per_compute_unit.to_string().into())?,
                    SuiJsonValue::new(max_num_compute_units.to_string().into())?,
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;
        info!("Submitting node task subscription transaction...");
        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_must_succeed(tx).await;

        info!(
            "Node task subscription transaction submitted successfully. Transaction digest: {:?}",
            response.digest
        );

        Ok(response.digest.to_string())
    }

    /// Submits a transaction to unsubscribe a node from a specific task in the Atoma network.
    ///
    /// This method creates and submits a transaction that unsubscribes a node from a task they were
    /// previously subscribed to. The node must have a valid node badge to perform this operation.
    ///
    /// # Arguments
    ///
    /// * `task_small_id` - The small ID of the task to unsubscribe from
    /// * `node_small_id` - Optional small ID of the node. If None, uses the client's stored badge ID
    /// * `gas` - Optional ObjectID to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to GAS_BUDGET
    /// * `gas_price` - Optional gas price for the transaction. If None, uses network's reference price
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the unsubscription is successful, or an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use sui_sdk::types::base_types::ObjectID;
    /// # async fn example(client: &mut AtomaSuiClient) -> Result<()> {
    /// // Unsubscribe from task with default gas settings
    /// client.submit_unsubscribe_node_from_task_tx(
    ///     123,                    // task_small_id
    ///     None,                   // use stored node_small_id
    ///     None,                   // default gas
    ///     None,                   // default gas budget
    ///     None                    // default gas price
    /// ).await?;
    ///
    /// // Or with custom gas settings and specific node ID
    /// let gas_object = ObjectID::new([1; 32]);
    /// client.submit_unsubscribe_node_from_task_tx(
    ///     123,                    // task_small_id
    ///     Some(456),              // specific node_small_id
    ///     Some(gas_object),       // specific gas object
    ///     Some(10_000_000),       // 0.01 SUI gas budget
    ///     Some(1000)              // specific gas price
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(level = "info", skip_all, fields(
        address = %self.wallet_ctx.active_address().unwrap()
    ))]
    pub async fn submit_unsubscribe_node_from_task_tx(
        &mut self,
        task_small_id: u64,
        node_small_id: Option<u64>,
        gas: Option<ObjectID>,
        gas_budget: Option<u64>,
        gas_price: Option<u64>,
    ) -> Result<String> {
        let client = self.wallet_ctx.get_client().await?;
        let active_address = self.wallet_ctx.active_address()?;
        let node_small_id = node_small_id.unwrap_or(
            self.node_badge
                .as_ref()
                .ok_or(AtomaSuiClientError::FailedToFindNodeBadge)?
                .1,
        );
        let tx = client
            .transaction_builder()
            .move_call(
                active_address,
                self.config.atoma_package_id(),
                MODULE_ID,
                NODE_TASK_UNSUBSCRIPTION_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::new(node_small_id.into())?,
                    SuiJsonValue::new(task_small_id.into())?,
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;
        info!("Submitting node try settle stack transaction...");
        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_must_succeed(tx).await;

        info!(
            "Node try settle stack transaction submitted successfully. Transaction digest: {:?}",
            response.digest
        );
        Ok(response.digest.to_string())
    }

    /// Submits a transaction to try to settle a stack in the Atoma network.
    ///
    /// This method creates and submits a transaction that attempts to settle a stack with
    /// the provided proof and merkle leaf data. The node must have a valid node badge to
    /// perform this operation.
    ///
    /// # Arguments
    ///
    /// * `stack_small_id` - The small ID of the stack to settle
    /// * `node_small_id` - Optional small ID of the node. If None, uses the client's stored badge ID
    /// * `num_claimed_compute_units` - The number of compute units being claimed for this stack
    /// * `committed_stack_proof` - The proof data for the committed stack
    /// * `stack_merkle_leaf` - The merkle leaf data for the stack
    /// * `gas` - Optional ObjectID to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to GAS_BUDGET
    /// * `gas_price` - Optional gas price for the transaction. If None, uses network's reference price
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the stack settlement attempt is successful, or an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use sui_sdk::types::base_types::ObjectID;
    /// # async fn example(client: &mut AtomaSuiClient) -> Result<()> {
    /// // Try to settle stack with default gas settings
    /// client.submit_try_settle_stack_tx(
    ///     123,                    // stack_small_id
    ///     None,                   // use stored node_small_id
    ///     1000,                   // num_claimed_compute_units
    ///     vec![1, 2, 3],         // committed_stack_proof
    ///     vec![4, 5, 6],         // stack_merkle_leaf
    ///     None,                   // default gas
    ///     None,                   // default gas budget
    ///     None                    // default gas price
    /// ).await?;
    ///
    /// // Or with custom gas settings and specific node ID
    /// let gas_object = ObjectID::new([1; 32]);
    /// client.submit_try_settle_stack_tx(
    ///     123,                    // stack_small_id
    ///     Some(456),              // specific node_small_id
    ///     1000,                   // num_claimed_compute_units
    ///     vec![1, 2, 3],         // committed_stack_proof
    ///     vec![4, 5, 6],         // stack_merkle_leaf
    ///     Some(gas_object),       // specific gas object
    ///     Some(10_000_000),       // 0.01 SUI gas budget
    ///     Some(1000)              // specific gas price
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(level = "info", skip_all, fields(
        address = %self.wallet_ctx.active_address().unwrap(),
        num_claimed_compute_units = %num_claimed_compute_units,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_try_settle_stack_tx(
        &mut self,
        stack_small_id: u64,
        node_small_id: Option<u64>,
        num_claimed_compute_units: u64,
        committed_stack_proof: Vec<u8>,
        stack_merkle_leaf: Vec<u8>,
        gas: Option<ObjectID>,
        gas_budget: Option<u64>,
        gas_price: Option<u64>,
    ) -> Result<String> {
        let client = self.wallet_ctx.get_client().await?;
        let active_address = self.wallet_ctx.active_address()?;
        let node_small_id = node_small_id.unwrap_or(
            self.node_badge
                .as_ref()
                .ok_or(AtomaSuiClientError::FailedToFindNodeBadge)?
                .1,
        );
        let tx = client
            .transaction_builder()
            .move_call(
                active_address,
                self.config.atoma_package_id(),
                MODULE_ID,
                TRY_SETLE_STACK_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::new(node_small_id.into())?,
                    SuiJsonValue::new(stack_small_id.into())?,
                    SuiJsonValue::new(num_claimed_compute_units.to_string().into())?,
                    SuiJsonValue::new(committed_stack_proof.into())?,
                    SuiJsonValue::new(stack_merkle_leaf.into())?,
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;

        info!("Submitting node try settle stack transaction...");
        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_must_succeed(tx).await;

        info!(
            "Node try settle stack transaction submitted successfully. Transaction digest: {:?}",
            response.digest
        );

        Ok(response.digest.to_string())
    }

    /// Submits a transaction to attest to a stack settlement in the Atoma network.
    ///
    /// This method creates and submits a transaction that provides attestation for a stack settlement,
    /// including the proof and merkle leaf data. The node must have a valid node badge to perform
    /// this operation.
    ///
    /// # Arguments
    ///
    /// * `stack_small_id` - The small ID of the stack being attested
    /// * `node_small_id` - Optional small ID of the node. If None, uses the client's stored badge ID
    /// * `committed_stack_proof` - The proof data for the committed stack
    /// * `stack_merkle_leaf` - The merkle leaf data for the stack
    /// * `gas` - Optional ObjectID to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to GAS_BUDGET
    /// * `gas_price` - Optional gas price for the transaction. If None, uses network's reference price
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the attestation is successful, or an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use sui_sdk::types::base_types::ObjectID;
    /// # async fn example(client: &mut AtomaSuiClient) -> Result<()> {
    /// // Submit attestation with default gas settings
    /// client.submit_stack_settlement_attestation_tx(
    ///     123,                    // stack_small_id
    ///     None,                   // use stored node_small_id
    ///     vec![1, 2, 3],         // committed_stack_proof
    ///     vec![4, 5, 6],         // stack_merkle_leaf
    ///     None,                   // default gas
    ///     None,                   // default gas budget
    ///     None                    // default gas price
    /// ).await?;
    ///
    /// // Or with custom gas settings and specific node ID
    /// let gas_object = ObjectID::new([1; 32]);
    /// client.submit_stack_settlement_attestation_tx(
    ///     123,                    // stack_small_id
    ///     Some(456),              // specific node_small_id
    ///     vec![1, 2, 3],         // committed_stack_proof
    ///     vec![4, 5, 6],         // stack_merkle_leaf
    ///     Some(gas_object),       // specific gas object
    ///     Some(10_000_000),       // 0.01 SUI gas budget
    ///     Some(1000)              // specific gas price
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(level = "info", skip_all, fields(
        address = %self.wallet_ctx.active_address().unwrap()
    ))]
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_stack_settlement_attestation_tx(
        &mut self,
        stack_small_id: u64,
        node_small_id: Option<u64>,
        committed_stack_proof: Vec<u8>,
        stack_merkle_leaf: Vec<u8>,
        gas: Option<ObjectID>,
        gas_budget: Option<u64>,
        gas_price: Option<u64>,
    ) -> Result<String> {
        let client = self.wallet_ctx.get_client().await?;
        let active_address = self.wallet_ctx.active_address()?;
        let node_small_id = node_small_id.unwrap_or(
            self.node_badge
                .as_ref()
                .ok_or(AtomaSuiClientError::FailedToFindNodeBadge)?
                .1,
        );
        let tx = client
            .transaction_builder()
            .move_call(
                active_address,
                self.config.atoma_package_id(),
                MODULE_ID,
                STACK_SETTLEMENT_ATTESTATION_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::new(node_small_id.into())?,
                    SuiJsonValue::new(stack_small_id.into())?,
                    SuiJsonValue::new(committed_stack_proof.into())?,
                    SuiJsonValue::new(stack_merkle_leaf.into())?,
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;

        info!("Submitting stack settlement attestation transaction...");

        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_must_succeed(tx).await;

        info!(
            "Stack settlement attestation transaction submitted successfully. Transaction digest: {:?}",
            response.digest
        );

        Ok(response.digest.to_string())
    }

    /// Submits a transaction to start an attestation dispute in the Atoma network.
    ///
    /// This method creates and submits a transaction that initiates a dispute regarding
    /// a stack settlement attestation, providing proof data to support the dispute claim.
    /// The node must have a valid node badge to perform this operation.
    ///
    /// # Arguments
    ///
    /// * `stack_small_id` - The small ID of the stack being disputed
    /// * `node_small_id` - Optional small ID of the node. If None, uses the client's stored badge ID
    /// * `committed_stack_proof` - The proof data for the committed stack that supports the dispute
    /// * `gas` - Optional ObjectID to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to GAS_BUDGET
    /// * `gas_price` - Optional gas price for the transaction. If None, uses network's reference price
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the dispute initiation is successful, or an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use sui_sdk::types::base_types::ObjectID;
    /// # async fn example(client: &mut AtomaSuiClient) -> Result<()> {
    /// // Start dispute with default gas settings
    /// client.submit_start_attestation_dispute_tx(
    ///     123,                    // stack_small_id
    ///     None,                   // use stored node_small_id
    ///     vec![1, 2, 3],         // committed_stack_proof
    ///     None,                   // default gas
    ///     None,                   // default gas budget
    ///     None                    // default gas price
    /// ).await?;
    ///
    /// // Or with custom gas settings and specific node ID
    /// let gas_object = ObjectID::new([1; 32]);
    /// client.submit_start_attestation_dispute_tx(
    ///     123,                    // stack_small_id
    ///     Some(456),              // specific node_small_id
    ///     vec![1, 2, 3],         // committed_stack_proof
    ///     Some(gas_object),       // specific gas object
    ///     Some(10_000_000),       // 0.01 SUI gas budget
    ///     Some(1000)              // specific gas price
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(level = "info", skip_all, fields(
        address = %self.wallet_ctx.active_address().unwrap()
    ))]
    pub async fn submit_start_attestation_dispute_tx(
        &mut self,
        stack_small_id: u64,
        node_small_id: Option<u64>,
        committed_stack_proof: Vec<u8>,
        gas: Option<ObjectID>,
        gas_budget: Option<u64>,
        gas_price: Option<u64>,
    ) -> Result<()> {
        let client = self.wallet_ctx.get_client().await?;
        let active_address = self.wallet_ctx.active_address()?;
        let node_small_id = node_small_id.unwrap_or(
            self.node_badge
                .as_ref()
                .ok_or(AtomaSuiClientError::FailedToFindNodeBadge)?
                .1,
        );
        let tx = client
            .transaction_builder()
            .move_call(
                active_address,
                self.config.atoma_package_id(),
                MODULE_ID,
                START_ATTESTATION_DISPUTE_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::new(node_small_id.into())?,
                    SuiJsonValue::new(stack_small_id.into())?,
                    SuiJsonValue::new(committed_stack_proof.into())?,
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;

        info!("Submitting start attestation dispute transaction...");

        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_must_succeed(tx).await;

        info!(
            "Start attestation dispute transaction submitted successfully. Transaction digest: {:?}",
            response.digest
        );

        Ok(())
    }

    /// Submits a transaction to claim funds for settled tickets in the Atoma network.
    ///
    /// This method creates and submits a transaction that claims funds for a list of settled ticket IDs.
    /// The node must have a valid node badge to perform this operation.
    ///
    /// # Arguments
    ///
    /// * `settled_ticket_ids` - A vector of ticket IDs that have been settled and are ready for claiming
    /// * `node_small_id` - Optional small ID of the node. If None, uses the client's stored badge ID
    /// * `gas` - Optional ObjectID to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to GAS_BUDGET
    /// * `gas_price` - Optional gas price for the transaction. If None, uses network's reference price
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the funds claim is successful, or an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use sui_sdk::types::base_types::ObjectID;
    /// # async fn example(client: &mut AtomaSuiClient) -> Result<()> {
    /// // Claim funds with default gas settings
    /// client.submit_claim_funds_tx(
    ///     vec![123, 456],         // settled_ticket_ids
    ///     None,                   // use stored node_small_id
    ///     None,                   // default gas
    ///     None,                   // default gas budget
    ///     None                    // default gas price
    /// ).await?;
    ///
    /// // Or with custom gas settings and specific node ID
    /// let gas_object = ObjectID::new([1; 32]);
    /// client.submit_claim_funds_tx(
    ///     vec![123, 456],         // settled_ticket_ids
    ///     Some(789),              // specific node_small_id
    ///     Some(gas_object),       // specific gas object
    ///     Some(10_000_000),       // 0.01 SUI gas budget
    ///     Some(1000)              // specific gas price
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(level = "info", skip_all, fields(
        address = %self.wallet_ctx.active_address().unwrap()
    ))]
    pub async fn submit_claim_funds_tx(
        &mut self,
        settled_ticket_ids: Vec<u64>,
        node_small_id: Option<u64>,
        gas: Option<ObjectID>,
        gas_budget: Option<u64>,
        gas_price: Option<u64>,
    ) -> Result<String> {
        let client = self.wallet_ctx.get_client().await?;
        let active_address = self.wallet_ctx.active_address()?;
        let node_small_id = node_small_id.unwrap_or(
            self.node_badge
                .as_ref()
                .ok_or(AtomaSuiClientError::FailedToFindNodeBadge)?
                .1,
        );
        let tx = client
            .transaction_builder()
            .move_call(
                active_address,
                self.config.atoma_package_id(),
                MODULE_ID,
                CLAIM_FUNDS_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::new(node_small_id.into())?,
                    SuiJsonValue::new(settled_ticket_ids.into())?,
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;

        info!("Submitting claim funds transaction...");

        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_must_succeed(tx).await;

        info!(
            "Claim funds transaction submitted successfully. Transaction digest: {:?}",
            response.digest
        );

        Ok(response.digest.to_string())
    }
}

#[derive(Debug, Error)]
pub enum AtomaSuiClientError {
    #[error("Failed to create wallet context")]
    WalletContextError(#[from] anyhow::Error),
    #[error("Failed to submit node registration transaction")]
    NodeRegistrationFailed,
    #[error("Failed to find node badge")]
    FailedToFindNodeBadge,
    #[error("Sui client error: `{0}`")]
    AtomaSuiClientError(#[from] sui_sdk::error::Error),
    #[error("Node is not subscribed to model {0}")]
    NodeNotSubscribedToModel(String),
}

pub(crate) mod utils {
    use super::*;
    use sui_sdk::{
        rpc_types::{Page, SuiObjectDataFilter, SuiObjectDataOptions, SuiObjectResponseQuery},
        types::base_types::{ObjectType, SuiAddress},
        SuiClient,
    };
    use tracing::error;

    /// The name of the Atoma's contract node badge type
    const DB_NODE_TYPE_NAME: &str = "NodeBadge";
    /// The page size for querying a user's owned objects
    const PAGE_SIZE: usize = 100;

    /// Retrieves the node badge (ObjectID and small_id) associated with a given address.
    ///
    /// This function queries the Sui blockchain to find a NodeBadge object owned by the specified
    /// address that was created by the specified package. The NodeBadge represents a node's
    /// registration in the Atoma network.
    ///
    /// # Arguments
    ///
    /// * `client` - A reference to the SuiClient used to interact with the blockchain
    /// * `package` - The ObjectID of the Atoma package that created the NodeBadge
    /// * `active_address` - The SuiAddress to query for owned NodeBadge objects
    ///
    /// # Returns
    ///
    /// Returns `Option<(ObjectID, u64)>` where:
    /// - `Some((object_id, small_id))` if a NodeBadge is found, where:
    ///   - `object_id` is the unique identifier of the NodeBadge object
    ///   - `small_id` is the node's numeric identifier in the Atoma network
    /// - `None` if:
    ///   - No NodeBadge is found
    ///   - The query fails
    ///   - The object data cannot be parsed
    ///
    /// # Example
    ///
    /// ```ignore
    /// use sui_sdk::SuiClient;
    /// use sui_sdk::types::base_types::{ObjectID, SuiAddress};
    ///
    /// async fn example(client: &SuiClient) {
    ///     let package_id = ObjectID::new([1; 32]);
    ///     let address = SuiAddress::random_for_testing_only();
    ///     
    ///     match get_node_badge(client, package_id, address).await {
    ///         Some((badge_id, small_id)) => {
    ///             println!("Found NodeBadge: ID={}, small_id={}", badge_id, small_id);
    ///         }
    ///         None => {
    ///             println!("No NodeBadge found for address");
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// # Implementation Notes
    ///
    /// - The function queries up to 100 objects at a time
    /// - The function filters objects by package and looks for the specific NodeBadge type
    /// - Object content is parsed to extract the small_id from the Move object's fields
    pub(crate) async fn get_node_badge(
        client: &SuiClient,
        package: ObjectID,
        active_address: SuiAddress,
    ) -> Option<(ObjectID, u64)> {
        let mut cursor = None;
        loop {
            let Page {
                data,
                has_next_page,
                next_cursor,
            } = match client
                .read_api()
                .get_owned_objects(
                    active_address,
                    Some(SuiObjectResponseQuery {
                        filter: Some(SuiObjectDataFilter::Package(package)),
                        options: Some(SuiObjectDataOptions {
                            show_type: true,
                            show_content: true,
                            ..Default::default()
                        }),
                    }),
                    cursor,
                    Some(PAGE_SIZE),
                )
                .await
            {
                Ok(page) => page,
                Err(e) => {
                    error!("Failed to get node badge: {:?}", e);
                    return None;
                }
            };

            if let Some(object) = data.into_iter().find_map(|resp| {
                let object = resp.data?;

                let ObjectType::Struct(type_) = object.type_? else {
                    return None;
                };

                if type_.module().as_str() == MODULE_ID
                    && type_.name().as_str() == DB_NODE_TYPE_NAME
                {
                    let id = object
                        .content?
                        .try_as_move()?
                        .clone()
                        .fields
                        .to_json_value();

                    Some((
                        object.object_id,
                        id["small_id"]["inner"].as_str()?.parse().ok()?,
                    ))
                } else {
                    None
                }
            }) {
                return Some(object);
            }

            // Check if there is a next page
            if !has_next_page {
                break;
            }
            cursor = next_cursor;
        }
        None
    }
}
