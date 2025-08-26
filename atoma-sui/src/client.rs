use std::path::Path;
use sui_keys::keystore::{AccountKeystore, Keystore};
use sui_sdk::{
    json::SuiJsonValue,
    types::{base_types::ObjectID, crypto::Signature},
    wallet_context::WalletContext,
};
use thiserror::Error;
use tracing::{error, info, instrument};

use crate::{config::Config as SuiConfig, events::NodePublicKeyCommittmentEvent};

type Result<T> = std::result::Result<T, AtomaSuiClientError>;

/// The gas budget for the node registration transaction
const GAS_BUDGET: u64 = 50_000_000; // 0.05 SUI

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
const TRY_SETTLE_STACK_METHOD: &str = "try_settle_stack";

/// The Atoma's contract method name for stack settlement attestation
const STACK_SETTLEMENT_ATTESTATION_METHOD: &str = "submit_stack_settlement_attestation";

/// The Atoma's contract method name for starting an attestation dispute
const START_ATTESTATION_DISPUTE_METHOD: &str = "start_attestation_dispute";

/// The Atoma's contract method name for claiming funds
const CLAIM_FUNDS_METHOD: &str = "claim_funds";

/// The Atoma's contract method name for claiming funds for stacks
const CLAIM_FUNDS_FOR_STACKS_METHOD: &str = "claim_funds_for_stacks";

/// The Atoma's contract method name for updating a node task subscription
const UPDATE_NODE_TASK_SUBSCRIPTION_METHOD: &str = "update_node_subscription";

/// The Atoma's contract method name for rotating the protocol's nodes public key
const ROTATE_NODE_PUBLIC_KEY: &str = "rotate_node_public_key";

/// The key rotation counter field name for the `NewKeyRotationEvent` event
const KEY_ROTATION_COUNTER_FIELD: &str = "key_rotation_counter";

/// The nonce field name for the `NewKeyRotationEvent` event
const NONCE_FIELD: &str = "nonce";

/// Client for interacting with Atoma's Sui blockchain functionality
pub struct Client {
    /// Configuration settings for the Atoma client
    config: SuiConfig,
    /// The Sui client for blockchain interactions
    wallet_ctx: WalletContext,
    /// An optional tuple containing the `ObjectID` and small ID of the node badge,
    /// used for authentication
    node_badge: Option<(ObjectID, u64)>,
    /// The `ObjectID` of the USDC wallet address
    usdc_wallet: Option<ObjectID>,
}

impl Client {
    /// Creates a new Sui client instance
    ///
    /// # Arguments
    /// * `config` - Configuration settings for the client
    ///
    /// # Returns
    /// A new client instance
    ///
    /// # Errors
    /// Returns an error if:
    /// - Failed to initialize wallet context
    /// - Failed to get client from wallet context
    /// - Failed to get active address
    /// - Failed to retrieve node badge
    #[instrument(level = "info", skip_all, err, fields(
        config = %config.sui_config_path()
    ))]
    pub async fn new(config: SuiConfig) -> Result<Self> {
        let sui_config_path = config.sui_config_path();
        let sui_config_path = Path::new(&sui_config_path);
        let mut wallet_ctx = WalletContext::new(sui_config_path)?;
        if let Some(request_timeout) = config.request_timeout() {
            wallet_ctx = wallet_ctx.with_request_timeout(request_timeout);
        }
        if let Some(max_concurrent_requests) = config.max_concurrent_requests() {
            wallet_ctx = wallet_ctx.with_max_concurrent_requests(max_concurrent_requests);
        }
        let active_address = wallet_ctx.active_address()?;
        info!("Current active address: {}", active_address);
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
            usdc_wallet: None,
        })
    }

    /// Creates a new client instance
    ///
    /// # Errors
    /// - If Sui client initialization fails
    /// - If keystore operations fail
    /// - If network connection fails
    pub async fn new_from_config<P: AsRef<Path> + Send>(config_path: P) -> Result<Self> {
        let config = SuiConfig::from_file_path(config_path);
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
    /// * `gas` - Optional `ObjectID` to use as gas for the transaction. If None, the system will
    ///           automatically select a gas object.
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to `GAS_BUDGET`
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
    ///
    /// # Errors
    /// This function returns an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - The node badge cannot be found after registration
    ///
    /// # Panics
    /// This function panics if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - The node badge cannot be found after registration
    #[instrument(level = "info", skip_all, err, fields(
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
                vec![SuiJsonValue::from_object_id(self.config.atoma_db())],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;
        info!("Submitting node registration transaction...");
        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_may_fail(tx).await?;

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
    /// * `node_badge_id` - Optional `ObjectID` of the node badge. If None, uses the client's stored badge
    /// * `gas` - Optional `ObjectID` to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to `GAS_BUDGET`
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
    ///
    /// # Errors
    /// This function returns an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Panics
    /// This function panics if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    #[instrument(level = "info", skip_all, err, fields(
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
        let response = self.wallet_ctx.execute_transaction_may_fail(tx).await?;

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
    /// * `node_badge_id` - Optional `ObjectID` of the node badge. If None, uses the client's stored badge ID
    /// * `price_per_one_million_compute_units` - The price per compute unit the node is willing to charge
    /// * `gas` - Optional `ObjectID` to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to `GAS_BUDGET`
    /// * `gas_price` - Optional gas price for the transaction. If None, uses the network's reference price
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
    ///     Some(gas_object),       // specific gas object
    ///     Some(10_000_000),       // 0.01 SUI gas budget
    ///     Some(1000)              // specific gas price
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// This function returns an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Panics
    /// This function panics if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    #[instrument(level = "info", skip_all, err, fields(
        address = %self.wallet_ctx.active_address().unwrap(),
        price_per_one_million_compute_units = %price_per_one_million_compute_units,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_node_task_subscription_tx(
        &mut self,
        task_small_id: u64,
        node_badge_id: Option<ObjectID>,
        price_per_one_million_compute_units: u64,
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
                NODE_TASK_SUBSCRIPTION_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::from_object_id(node_badge_id),
                    SuiJsonValue::new(task_small_id.to_string().into())?,
                    SuiJsonValue::new(price_per_one_million_compute_units.to_string().into())?,
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;
        info!("Submitting node task subscription transaction...");
        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_may_fail(tx).await?;

        info!(
            "Node task subscription transaction submitted successfully. Transaction digest: {:?}",
            response.digest
        );

        Ok(response.digest.to_string())
    }

    /// Submits a transaction to update a node's task subscription in the Atoma network.
    ///
    /// This method creates and submits a transaction that updates a node's task subscription
    /// with new pricing parameters. The node must have a valid node badge to perform this operation.
    ///
    /// # Arguments
    ///
    /// * `task_small_id` - The small ID of the task to update the subscription for
    /// * `node_badge_id` - Optional `ObjectID` of the node badge. If None, uses the client's stored badge ID
    /// * `price_per_one_million_compute_units` - The new price per compute unit for the task subscription
    /// * `gas` - Optional `ObjectID` to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to `GAS_BUDGET`
    /// * `gas_price` - Optional gas price for the transaction. If None, uses the network's reference price
    ///
    /// # Returns
    ///
    /// Returns `Result<String>` where the String is the transaction digest if successful.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use sui_sdk::types::base_types::ObjectID;
    /// # async fn example(client: &mut AtomaSuiClient) -> Result<()> {
    ///     // Update task subscription with default gas settings
    ///     let tx_digest = client.submit_update_node_task_subscription_tx(
    ///         123,                    // task_small_id
    ///         None,                   // use stored node_badge_id
    ///         1000,                   // new price per compute unit
    ///         None,                   // default gas
    ///         None,                   // default gas budget
    ///         None                    // default gas price
    ///     ).await?;
    ///
    ///     // Or with custom gas settings and specific node badge ID
    ///     let gas_object = ObjectID::new([1; 32]);
    ///     let node_badge = ObjectID::new([2; 32]);
    ///     let tx_digest = client.submit_update_node_task_subscription_tx(
    ///         123,                    // task_small_id
    ///         Some(node_badge),       // specific node_badge_id
    ///         1000,                   // new price per compute unit
    ///         Some(gas_object),       // specific gas object
    ///         Some(10_000_000),       // 0.01 SUI gas budget
    ///         Some(1000)              // specific gas price
    ///     ).await?;
    ///
    ///     println!("Task subscription updated: {}", tx_digest);
    ///     Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// This function returns an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Panics
    /// This function panics if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    #[instrument(level = "info", skip_all, err, fields(
        method = %UPDATE_NODE_TASK_SUBSCRIPTION_METHOD,
        price_per_one_million_compute_units = %price_per_one_million_compute_units,
        address = %self.wallet_ctx.active_address().unwrap()
    ))]
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_update_node_task_subscription_tx(
        &mut self,
        task_small_id: u64,
        node_badge_id: Option<ObjectID>,
        price_per_one_million_compute_units: u64,
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
                UPDATE_NODE_TASK_SUBSCRIPTION_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::from_object_id(node_badge_id),
                    SuiJsonValue::new(task_small_id.to_string().into())?,
                    SuiJsonValue::new(price_per_one_million_compute_units.to_string().into())?,
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;
        info!("Submitting node task update subscription transaction...");
        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_may_fail(tx).await?;

        info!(
            "Node task update subscription transaction submitted successfully. Transaction digest: {:?}",
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
    /// * `node_badge_id` - Optional `ObjectID` of the node badge. If None, uses the client's stored badge ID
    /// * `gas` - Optional `ObjectID` to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to `GAS_BUDGET`
    /// * `gas_price` - Optional gas price for the transaction. If None, uses the network's reference price
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
    ///     None,                   // use stored node_badge_id
    ///     None,                   // default gas
    ///     None,                   // default gas budget
    ///     None                    // default gas price
    /// ).await?;
    ///
    /// // Or with custom gas settings and specific node ID
    /// let gas_object = ObjectID::new([1; 32]);
    /// client.submit_unsubscribe_node_from_task_tx(
    ///     123,                    // task_small_id
    ///     Some("0xabc123"),        // specific node_badge_id
    ///     Some(gas_object),         // specific gas object
    ///     Some(10_000_000),         // 0.01 SUI gas budget
    ///     Some(1000)                // specific gas price
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// This function returns an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Panics
    /// This function panics if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    #[instrument(level = "info", skip_all, err, fields(
        address = %self.wallet_ctx.active_address().unwrap()
    ))]
    pub async fn submit_unsubscribe_node_from_task_tx(
        &mut self,
        task_small_id: u64,
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
                NODE_TASK_UNSUBSCRIPTION_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::from_object_id(node_badge_id),
                    SuiJsonValue::new(task_small_id.to_string().into())?,
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;
        info!("Submitting node try settle stack transaction...");
        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_may_fail(tx).await?;

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
    /// * `node_badge_id` - Optional `ObjectID` of the node badge. If None, uses the client's stored badge ID
    /// * `num_claimed_compute_units` - The number of compute units being claimed for this stack
    /// * `committed_stack_proof` - The proof data for the committed stack
    /// * `stack_merkle_leaf` - The merkle leaf data for the stack
    /// * `gas` - Optional `ObjectID` to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to `GAS_BUDGET`
    /// * `gas_price` - Optional gas price for the transaction. If None, uses the network's reference price
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
    ///     None,                   // use stored node_badge_id
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
    ///     Some(456),              // specific node_badge_id
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
    ///
    /// # Errors
    /// This function returns an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Panics
    /// This function panics if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    #[instrument(level = "info", skip_all, err, fields(
        address = %self.wallet_ctx.active_address().unwrap(),
        num_claimed_compute_units = %num_claimed_compute_units,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_try_settle_stack_tx(
        &mut self,
        stack_small_id: u64,
        node_badge_id: Option<ObjectID>,
        num_claimed_compute_units: u64,
        committed_stack_proof: Vec<u8>,
        stack_merkle_leaf: Vec<u8>,
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
                TRY_SETTLE_STACK_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::from_object_id(node_badge_id),
                    SuiJsonValue::new(stack_small_id.to_string().into())?,
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
        let response = self.wallet_ctx.execute_transaction_may_fail(tx).await?;

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
    /// * `node_badge_id` - Optional `ObjectID` of the node badge. If None, uses the client's stored badge ID
    /// * `committed_stack_proof` - The proof data for the committed stack
    /// * `stack_merkle_leaf` - The merkle leaf data for the stack
    /// * `gas` - Optional `ObjectID` to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to `GAS_BUDGET`
    /// * `gas_price` - Optional gas price for the transaction. If None, uses the network's reference price
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
    ///     None,                   // use stored node_badge_id
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
    ///
    /// # Errors
    /// This function returns an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Panics
    /// This function panics if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    #[instrument(level = "info", skip_all, err, fields(
        address = %self.wallet_ctx.active_address().unwrap()
    ))]
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_stack_settlement_attestation_tx(
        &mut self,
        stack_small_id: u64,
        node_badge_id: Option<ObjectID>,
        committed_stack_proof: Vec<u8>,
        stack_merkle_leaf: Vec<u8>,
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
                STACK_SETTLEMENT_ATTESTATION_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::from_object_id(node_badge_id),
                    SuiJsonValue::new(stack_small_id.to_string().into())?,
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
        let response = self.wallet_ctx.execute_transaction_may_fail(tx).await?;

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
    /// * `node_badge_id` - Optional `ObjectID` of the node badge. If None, uses the client's stored badge ID
    /// * `committed_stack_proof` - The proof data for the committed stack that supports the dispute
    /// * `gas` - Optional `ObjectID` to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to `GAS_BUDGET`
    /// * `gas_price` - Optional gas price for the transaction. If None, uses the network's reference price
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
    ///     None,                   // use stored node_badge_id
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
    ///     Some(0xabc123),          // specific node_badge_id
    ///     vec![1, 2, 3],         // committed_stack_proof
    ///     Some(gas_object),       // specific gas object
    ///     Some(10_000_000),       // 0.01 SUI gas budget
    ///     Some(1000)              // specific gas price
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// This function returns an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Panics
    /// This function panics if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    #[instrument(level = "info", skip_all, err, fields(
        address = %self.wallet_ctx.active_address().unwrap()
    ))]
    pub async fn submit_start_attestation_dispute_tx(
        &mut self,
        stack_small_id: u64,
        node_badge_id: Option<ObjectID>,
        committed_stack_proof: Vec<u8>,
        gas: Option<ObjectID>,
        gas_budget: Option<u64>,
        gas_price: Option<u64>,
    ) -> Result<()> {
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
                START_ATTESTATION_DISPUTE_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::from_object_id(node_badge_id),
                    SuiJsonValue::new(stack_small_id.to_string().into())?,
                    SuiJsonValue::new(committed_stack_proof.into())?,
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;

        info!("Submitting start attestation dispute transaction...");

        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_may_fail(tx).await?;

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
    /// * `node_badge_id` - Optional `ObjectID` of the node badge. If None, uses the client's stored badge ID
    /// * `gas` - Optional `ObjectID` to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to `GAS_BUDGET`
    /// * `gas_price` - Optional gas price for the transaction. If None, uses the network's reference price
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
    ///     None,                   // use stored node_badge_id
    ///     None,                   // default gas
    ///     None,                   // default gas budget
    ///     None                    // default gas price
    /// ).await?;
    ///
    /// // Or with custom gas settings and specific node ID
    /// let gas_object = ObjectID::new([1; 32]);
    /// client.submit_claim_funds_tx(
    ///     vec![123, 456],         // settled_ticket_ids
    ///     Some(0xabc123),          // specific node_badge_id
    ///     Some(gas_object),       // specific gas object
    ///     Some(10_000_000),       // 0.01 SUI gas budget
    ///     Some(1000)              // specific gas price
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// This function returns an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Panics        
    /// This function panics if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    #[instrument(level = "info", skip_all, err, fields(
        address = %self.wallet_ctx.active_address().unwrap()
    ))]
    pub async fn submit_claim_funds_tx(
        &mut self,
        settled_ticket_ids: Vec<u64>,
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
                CLAIM_FUNDS_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::from_object_id(node_badge_id),
                    SuiJsonValue::new(settled_ticket_ids.into())?,
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;

        info!("Submitting claim funds transaction...");

        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_may_fail(tx).await?;

        info!(
            "Claim funds transaction submitted successfully. Transaction digest: {:?}",
            response.digest
        );

        Ok(response.digest.to_string())
    }

    /// Submits a transaction to claim funds for stacks in the Atoma network.
    ///
    /// This method creates and submits a transaction that claims funds for a list of stacks.
    /// The node must have a valid node badge to perform this operation.
    ///
    /// # Arguments
    ///
    /// * `stack_small_ids` - A vector of stack small IDs that have been settled and are ready for claiming
    /// * `node_badge_id` - Optional `ObjectID` of the node badge. If None, uses the client's stored badge ID
    /// * `num_claimed_compute_units` - A vector of the number of compute units claimed for each stack
    /// * `gas` - Optional `ObjectID` to use as gas for the transaction
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to `GAS_BUDGET`
    /// * `gas_price` - Optional gas price for the transaction. If None, uses the network's reference price
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the funds claim is successful, or an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The node badge is not found
    /// - The wallet context operations fail
    /// - The transaction submission fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use sui_sdk::types::base_types::ObjectID;
    /// # async fn example(client: &mut AtomaSuiClient) -> Result<()> {
    /// // Claim funds with default gas settings
    /// client.submit_claim_funds_for_stacks_tx(
    ///     vec![123, 456],         // stack_small_ids
    ///     None,                   // use stored node_badge_id
    ///     vec![1000, 2000],      // num_claimed_compute_units
    ///     None,                   // default gas
    ///     None,                   // default gas budget
    ///     None                    // default gas price
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// This function returns an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Panics
    /// This function panics if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    #[instrument(
        level = "info",
        skip_all,
        err,
        fields(
            address = %self.wallet_ctx.active_address().unwrap(),
            stack_small_ids = ?stack_small_ids,
            node_badge_id = ?node_badge_id,
            num_claimed_compute_units = ?num_claimed_compute_units,
    ))]
    pub async fn submit_claim_funds_for_stacks_tx(
        &mut self,
        stack_small_ids: Vec<u64>,
        node_badge_id: Option<ObjectID>,
        num_claimed_compute_units: Vec<u64>,
        gas: Option<ObjectID>,
        gas_budget: Option<u64>,
        gas_price: Option<u64>,
    ) -> Result<String> {
        let client = self.wallet_ctx.get_client().await?;
        let active_address = self.wallet_ctx.active_address()?;
        let node_badge_id = node_badge_id.unwrap_or(
            self.node_badge
                .ok_or(AtomaSuiClientError::FailedToFindNodeBadge)?
                .0,
        );
        info!(
            target = "atoma-sui-client",
            level = "info",
            "Node badge ID: {node_badge_id}"
        );
        if stack_small_ids.len() != num_claimed_compute_units.len() {
            return Err(AtomaSuiClientError::InvalidInputForClaimFundsForStacks {
                stack_small_ids_len: stack_small_ids.len(),
                num_claimed_compute_units_len: num_claimed_compute_units.len(),
            });
        }
        info!(
            target = "atoma-sui-client",
            level = "info",
            "Building claim funds for stacks transaction..."
        );
        let stack_small_ids = stack_small_ids
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>();
        let num_claimed_compute_units = num_claimed_compute_units
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>();
        let tx = client
            .transaction_builder()
            .move_call(
                active_address,
                self.config.atoma_package_id(),
                MODULE_ID,
                CLAIM_FUNDS_FOR_STACKS_METHOD,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::from_object_id(node_badge_id),
                    SuiJsonValue::new(stack_small_ids.into())?,
                    SuiJsonValue::new(num_claimed_compute_units.into())?,
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;

        info!("Submitted claim funds for stacks transaction...");

        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_may_fail(tx).await?;

        info!(
            "Claim funds for stacks transaction submitted successfully. Transaction digest: {:?}",
            response.digest
        );

        Ok(response.digest.to_string())
    }

    /// Submits a transaction to rotate a node's key with remote attestation in the Atoma network.
    ///
    /// This method creates and submits a transaction that rotates a node's key using remote
    /// attestation. The node must have a valid node badge to perform this operation. The method requires
    /// the new public key bytes and attestation report bytes.
    ///
    /// # Arguments
    ///
    /// * `public_key_bytes` - A 32-byte array containing the new public key
    /// * `attestation_report_bytes` - A vector of bytes containing the attestation report
    /// * `key_rotation_counter` - The key rotation counter value
    /// * `device_type` - The device type identifier (as a u16)
    /// * `task_small_id` - Optional small ID of the task
    /// * `gas` - Optional `ObjectID` to use as gas for the transaction. If None, the system will
    ///           automatically select a gas object
    /// * `gas_budget` - Optional gas budget for the transaction. If None, defaults to `GAS_BUDGET`
    /// * `gas_price` - Optional gas price for the transaction. If None, uses the network's reference price
    ///
    /// # Returns
    ///
    /// Returns `Result<(String, u64)>` where the String is the transaction digest and the u64 is
    /// the key rotation counter if successful.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The node badge is not found
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - Failed to find the key rotation event in the transaction response
    /// - Failed to parse the event data
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// async fn example(client: &mut Client) -> Result<()> {
    ///     let public_key = [0u8; 32];       // Your new public key
    ///     let attestation_report = vec![1, 2, 3, 4]; // Your attestation report bytes
    ///     let key_rotation_counter = 1;
    ///     let device_type = 1;
    ///
    ///     // Submit with default gas settings
    ///     let (tx_digest, new_counter) = client.submit_key_rotation_remote_attestation(
    ///         public_key,
    ///         attestation_report,
    ///         key_rotation_counter,
    ///         device_type,
    ///         None,    // task_small_id
    ///         None,    // default gas
    ///         None,    // default gas budget
    ///         None,    // default gas price
    ///     ).await?;
    ///
    ///     println!("Key rotation submitted: {}, new counter: {}", tx_digest, new_counter);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Panics
    /// This function panics if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Errors
    /// This function returns an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "info", skip_all, err, fields(
        address = %self.wallet_ctx.active_address().unwrap(),
        public_key = %hex::encode(public_key_bytes),
        device_type = %device_type,
        key_rotation_counter = %key_rotation_counter,
    ))]
    pub async fn submit_key_rotation_remote_attestation(
        &mut self,
        public_key_bytes: [u8; 32],
        evidence_data_bytes: Vec<u8>,
        key_rotation_counter: u64,
        device_type: u16,
        gas: Option<ObjectID>,
        gas_budget: Option<u64>,
        gas_price: Option<u64>,
    ) -> Result<(String, u64)> {
        let client = self.wallet_ctx.get_client().await?;
        let active_address = self.wallet_ctx.active_address()?;
        let node_badge_id = self
            .node_badge
            .as_ref()
            .ok_or(AtomaSuiClientError::FailedToFindNodeBadge)?
            .0;

        let tx = client
            .transaction_builder()
            .move_call(
                active_address,
                self.config.atoma_package_id(),
                MODULE_ID,
                ROTATE_NODE_PUBLIC_KEY,
                vec![],
                vec![
                    SuiJsonValue::from_object_id(self.config.atoma_db()),
                    SuiJsonValue::from_object_id(node_badge_id),
                    SuiJsonValue::new(public_key_bytes.to_vec().into())?,
                    SuiJsonValue::new(evidence_data_bytes.into())?,
                    SuiJsonValue::new(key_rotation_counter.to_string().into())?,
                    SuiJsonValue::new(device_type.into())?,
                ],
                gas,
                gas_budget.unwrap_or(GAS_BUDGET),
                gas_price,
            )
            .await?;

        info!("Submitting key rotation remote attestation transaction...");

        let tx = self.wallet_ctx.sign_transaction(&tx);
        let response = self.wallet_ctx.execute_transaction_may_fail(tx).await?;
        let digest = response.digest.to_string();
        let events = response.events;
        if let Some(tx_block_events) = events {
            let event_data = tx_block_events.data;
            if let Some(event) = event_data.into_iter().next() {
                let node_key_rotation_event: NodePublicKeyCommittmentEvent =
                    serde_json::from_value(event.parsed_json)?;
                let key_rotation_counter = node_key_rotation_event.key_rotation_counter;
                return Ok((digest, key_rotation_counter));
            }
        }
        Err(AtomaSuiClientError::FailedToFindNewKeyRotationEvent)
    }

    /// Get or load the USDC wallet object ID
    ///
    /// This method checks if the USDC wallet object ID is already loaded and returns it if so.
    /// Otherwise, it loads the USDC wallet object ID by finding the most balance USDC coin for the active address.
    ///
    /// # Returns
    ///
    /// Returns the USDC wallet object ID.
    ///
    /// # Errors
    ///
    /// Returns an error if no USDC wallet is found for the active address.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut client = AtomaProxy::new(config).await?;
    /// let usdc_wallet_id = client.get_or_load_usdc_wallet_object_id().await?;
    /// ```
    ///
    /// # Errors
    /// This function returns an error if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    ///
    /// # Panics
    /// This function panics if:
    /// - The wallet context operations fail
    /// - The transaction submission fails
    /// - No node badge is found when one is not explicitly provided
    #[instrument(level = "info", skip_all, err, fields(
        endpoint = "get_or_load_usdc_wallet_object_id",
        address = %self.wallet_ctx.active_address().unwrap()
    ))]
    pub async fn get_or_load_usdc_wallet_object_id(&mut self) -> Result<ObjectID> {
        if let Some(usdc_wallet_id) = self.usdc_wallet {
            Ok(usdc_wallet_id)
        } else {
            let active_address = self.wallet_ctx.active_address()?;
            match utils::find_usdc_token_wallet(
                &self.wallet_ctx.get_client().await?,
                self.config.usdc_package_id(),
                active_address,
            )
            .await
            {
                Ok(usdc_wallet) => {
                    self.usdc_wallet = Some(usdc_wallet);
                    Ok(usdc_wallet)
                }
                Err(e) => Err(e),
            }
        }
    }

    /// Retrieves the latest key rotation counter and nonce from the Atoma DB object.
    ///
    /// This method queries the Atoma DB shared object to get the current key rotation counter
    /// and nonce values, which are used for tracking key rotation events in the system.
    ///
    /// # Returns
    ///
    /// Returns `Result<Option<(u64, u64)>>` where:
    /// - `Ok(Some((key_rotation_counter, nonce)))` if the values are successfully retrieved
    /// - `Ok(None)` if the values are not found in the object
    /// - `Err(AtomaSuiClientError)` if there was an error retrieving or parsing the object
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Failed to get the client from wallet context
    /// - Failed to retrieve the Atoma DB object
    /// - Failed to access the content of the Atoma DB object
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut client = Client::new(config).await?;
    /// match client.get_last_key_rotation_event().await? {
    ///     Some((counter, nonce)) => {
    ///         println!("Current key rotation counter: {}, nonce: {}", counter, nonce);
    ///     },
    ///     None => {
    ///         println!("No key rotation counter found");
    ///     }
    /// }
    /// ```
    ///
    /// # Errors
    /// This function returns an error if:
    /// - Failed to get the client from wallet context
    /// - Failed to retrieve the Atoma DB object
    /// - Failed to access the content of the Atoma DB object
    ///
    /// # Panics
    /// This function panics if:
    /// - Failed to get the client from wallet context
    /// - Failed to retrieve the Atoma DB object
    /// - Failed to access the content of the Atoma DB object
    #[instrument(level = "info", skip_all, err, fields(
        endpoint = "get_last_key_rotation_event",
        address = %self.wallet_ctx.active_address().unwrap()
    ))]
    pub async fn get_last_key_rotation_event(&mut self) -> Result<Option<(u64, u64)>> {
        let client = self.wallet_ctx.get_client().await?;
        let events = client
            .read_api()
            .get_object_with_options(
                self.config.atoma_db(),
                sui_sdk::rpc_types::SuiObjectDataOptions {
                    show_type: true,
                    show_content: true,
                    ..Default::default()
                },
            )
            .await?
            .data;
        if let Some(atoma_db) = events {
            let Some(content) = atoma_db.content else {
                return Err(AtomaSuiClientError::FailedToRetrieveAtomaDbContent);
            };
            if let sui_sdk::rpc_types::SuiParsedData::MoveObject(object) = content {
                let object_fields = object.fields.to_json_value();
                let key_rotation_counter = object_fields
                    .get(KEY_ROTATION_COUNTER_FIELD)
                    .and_then(serde_json::Value::as_str)
                    .and_then(|s| s.parse::<u64>().ok());
                let nonce = object_fields
                    .get(NONCE_FIELD)
                    .and_then(serde_json::Value::as_str)
                    .and_then(|s| s.parse::<u64>().ok());
                if let (Some(key_rotation_counter), Some(nonce)) = (key_rotation_counter, nonce) {
                    return Ok(Some((key_rotation_counter, nonce)));
                }
            }
        }
        Ok(None)
    }

    /// Signs a hashed message using the active wallet's private key.
    ///
    /// This method retrieves the active address from the wallet context and uses the keystore
    /// to sign the provided hash.
    ///
    /// # Arguments
    ///
    /// * `hash` - A byte array representing the hash to be signed.
    ///
    /// # Returns
    ///
    /// Returns `Result<Signature>` where:
    /// - `Ok(Signature)` if the signing is successful
    /// - `Err(AtomaSuiClientError)` if there is an error signing the hash
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The active address cannot be retrieved from the wallet context
    /// - The keystore fails to sign the hash
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The active address cannot be retrieved from the wallet context
    /// - The keystore fails to sign the hash
    ///
    /// # Panics
    ///
    /// This function panics if:
    /// - The active address cannot be retrieved from the wallet context
    /// - The keystore fails to sign the hash
    #[instrument(level = "info", skip_all, err, fields(
        endpoint = "sign_hashed",
        address = %self.wallet_ctx.active_address().unwrap()
    ))]
    pub async fn sign_hashed(&mut self, hash: &[u8]) -> Result<Signature> {
        let active_address = self.wallet_ctx.active_address().unwrap();
        let signature = match &self.wallet_ctx.config.keystore {
            Keystore::File(keystore) => keystore.sign_hashed(&active_address, hash).unwrap(),
            Keystore::InMem(keystore) => keystore.sign_hashed(&active_address, hash).unwrap(),
        };
        Ok(signature)
    }
}

#[derive(Debug, Error)]
pub enum AtomaSuiClientError {
    #[error("Anyhow error: `{0}`")]
    AnyhowError(#[from] anyhow::Error),
    #[error("Failed to submit node registration transaction")]
    NodeRegistrationFailed,
    #[error("Failed to find node badge")]
    FailedToFindNodeBadge,
    #[error("Sui client error: `{0}`")]
    AtomaSuiClientError(#[from] sui_sdk::error::Error),
    #[error("Node is not subscribed to model `{0}`")]
    NodeNotSubscribedToModel(String),
    #[error("No USDC wallet found")]
    NoUsdcWalletFound,
    #[error("No USDC tokens found")]
    NoUsdcTokensFound,
    #[error("Failed to find new key rotation event")]
    FailedToFindNewKeyRotationEvent,
    #[error("Failed to parse event: `{0}`")]
    FailedToParseEvent(#[from] serde_json::Error),
    #[error("Failed to retrieve AtomaDB shared object content")]
    FailedToRetrieveAtomaDbContent,
    #[error("Invalid input for claim funds for stacks: stack_small_ids length `{stack_small_ids_len}` != num_claimed_compute_units length `{num_claimed_compute_units_len}`")]
    InvalidInputForClaimFundsForStacks {
        stack_small_ids_len: usize,
        num_claimed_compute_units_len: usize,
    },
}

pub(crate) mod utils {
    use super::{AtomaSuiClientError, ObjectID, Result, MODULE_ID};
    use sui_sdk::{
        rpc_types::{
            Page, SuiData, SuiObjectDataFilter, SuiObjectDataOptions, SuiObjectResponseQuery,
        },
        types::base_types::{ObjectType, SuiAddress},
        SuiClient,
    };
    use tracing::{error, instrument};

    /// The name of the Atoma's contract node badge type
    const DB_NODE_TYPE_NAME: &str = "NodeBadge";

    /// Retrieves the node badge (`ObjectID` and `small_id`) associated with a given address.
    ///
    /// This function queries the Sui blockchain to find a `NodeBadge` object owned by the specified
    /// address that was created by the specified package. The `NodeBadge` represents a node's
    /// registration in the Atoma network.
    ///
    /// # Arguments
    ///
    /// * `client` - A reference to the `SuiClient` used to interact with the blockchain
    /// * `package` - The `ObjectID` of the Atoma package that created the `NodeBadge`
    /// * `active_address` - The `SuiAddress` to query for owned `NodeBadge` objects
    ///
    /// # Returns
    ///
    /// Returns `Option<(ObjectID, u64)>` where:
    /// - `Some((object_id, small_id))` if a `NodeBadge` is found, where:
    ///   - `object_id` is the unique identifier of the `NodeBadge` object
    ///   - `small_id` is the node's numeric identifier in the Atoma network
    /// - `None` if:
    ///   - No `NodeBadge` is found
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
    /// - The function filters objects by package and looks for the specific `NodeBadge` type
    /// - Object content is parsed to extract the `small_id` from the Move object's fields
    #[instrument(level = "info", skip_all, fields(
        endpoint = "get_node_badge",
        address = %active_address
    ))]
    pub async fn get_node_badge(
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
                    None,
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

    /// Find the USDC token wallet for the given address
    ///
    /// # Returns
    ///
    /// Returns the USDC token wallet object ID.
    ///
    /// # Errors
    ///
    /// Returns an error if no USDC wallet is found for the active address.
    #[instrument(level = "info", skip_all, fields(
        endpoint = "find_usdc_token_wallet",
        address = %active_address
    ))]
    pub async fn find_usdc_token_wallet(
        client: &SuiClient,
        usdc_package: ObjectID,
        active_address: SuiAddress,
    ) -> Result<ObjectID> {
        let Page { data: coins, .. } = client
            .coin_read_api()
            .get_coins(
                active_address,
                Some(format!("{usdc_package}::usdc::USDC")),
                None,
                None,
            )
            .await?;
        coins
            .into_iter()
            .max_by_key(|coin| coin.balance)
            .map(|coin| coin.coin_object_id)
            .ok_or_else(|| AtomaSuiClientError::NoUsdcTokensFound)
    }
}
