use atoma_utils::constants::SIGNATURE;
use config::ProxyConfig;
use reqwest::Client;
use serde_json::json;
use sui_keys::keystore::FileBasedKeystore;

use crate::server::utils::sign_response_body;

pub mod config;

/// Registers the node on the proxy server
///
/// # Arguments
///
/// * `config` - Proxy configuration
/// * `node_small_id` - Small ID of the node
/// * `keystore` - Keystore for signing the registration request
/// * `address_index` - Index of the address to use for signing
pub async fn register_on_proxy(
    config: &ProxyConfig,
    node_small_id: u64,
    keystore: &FileBasedKeystore,
    address_index: usize,
) -> anyhow::Result<()> {
    let client = Client::new();
    let url = format!("{}/node/registration", config.proxy_address);

    let body = json!({
      "node_small_id": node_small_id,
      "public_address": config.node_public_address,
      "country": config.country,
    });

    let (_, signature) = sign_response_body(&body, keystore, address_index)?;

    let res = client
        .post(&url)
        .header(SIGNATURE, signature)
        .json(&body)
        .send()
        .await?;

    if !res.status().is_success() {
        anyhow::bail!("Failed to register on proxy server: {}", res.status());
    }
    Ok(())
}
