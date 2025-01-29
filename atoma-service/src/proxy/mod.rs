use config::ProxyConfig;
use reqwest::Client;
use serde_json::json;
use sui_keys::keystore::FileBasedKeystore;

use crate::server::utils::sign_response_body;

pub mod config;

/// Country field in the registration request
const COUNTRY: &str = "country";

/// Data field in the registration request
const DATA: &str = "data";

/// Node public address field in the registration request
const NODE_PUBLIC_ADDRESS: &str = "public_address";

/// Node small ID field in the registration request
const NODE_SMALL_ID: &str = "node_small_id";

/// Signature field in the registration request
const SIGNATURE: &str = "signature";

/// Registers the node on the proxy server
///
/// # Arguments
///
/// * `config` - Proxy configuration
/// * `node_small_id` - Small ID of the node
/// * `keystore` - Keystore for signing the registration request
/// * `address_index` - Index of the address to use for signing
///
/// # Errors
///
/// This function will return an error if:
/// - The request to the proxy server fails
/// - The server returns a non-success status code
/// - The signature generation fails
/// - The HTTP request fails to be sent
pub async fn register_on_proxy(
    config: &ProxyConfig,
    node_small_id: u64,
    keystore: &FileBasedKeystore,
    address_index: usize,
) -> anyhow::Result<()> {
    let client = Client::new();
    let url = format!("{}/v1/nodes", config.proxy_address);
    tracing::info!(
        target = "atoma-service",
        event = "register_on_proxy",
        url = url,
        "Registering on proxy server"
    );

    let data = json!({
      NODE_SMALL_ID: node_small_id,
      NODE_PUBLIC_ADDRESS: config.node_public_address,
      COUNTRY: config.country,
    });

    let (_, signature) = sign_response_body(&data, keystore, address_index)?;

    let body = json!({
        DATA: data,
        SIGNATURE: signature,
    });

    let res = client.post(&url).json(&body).send().await.map_err(|e| {
        tracing::error!(
            target = "atoma-service",
            event = "register_on_proxy_error",
            error = ?e,
            "Failed to register on proxy server"
        );
        anyhow::anyhow!("Failed to register on proxy server: {}", e)
    })?;

    if !res.status().is_success() {
        tracing::error!(
            target = "atoma-service",
            event = "register_on_proxy_error",
            error = ?res.status(),
            "Failed to register on proxy server"
        );
        anyhow::bail!("Failed to register on proxy server: {}", res.status());
    }
    Ok(())
}
