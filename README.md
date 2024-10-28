# Atoma Node infrastructure

![Atoma Logo](https://github.com/atoma-network/atoma-node/blob/atoma-image/atoma-assets/atoma-symbol.jpg)

[![Discord](https://img.shields.io/discord/1172593757586214964?label=Discord&logo=discord&logoColor=white)]
[![Twitter](https://img.shields.io/twitter/follow/Atoma_Network?style=social)](https://x.com/Atoma_Network)
[![Documentation](https://img.shields.io/badge/docs-gitbook-blue)](https://atoma.gitbook.io/atoma-docs)
[![License](https://img.shields.io/github/license/atoma-network/atoma-node)](LICENSE)

## Introduction

Atoma is a decentralized cloud compute network for AI that enables:

- **Verifiable Compute**: Transparent and trustworthy AI model execution, for both inference, text embeddings, multi-modality, etc.
- **Private Inference**: Secure processing with strong privacy guarantees, through the use of secure hardware enclaves (see [Atoma's private compute paper](https://arxiv.org/abs/2410.13752))
- **Decentralized Infrastructure**: A permissionless network of compute nodes, orchestrated by Atoma's smart contract on the Sui blockchain (see [repo](https://github.com/atoma-network/atoma-contracts))
- **LLM Focus**: Specialized in serving Large Language Models compute.

This repository contains the node software that enables node operators to participate in the Atoma Network. By running an Atoma node, you can:

1. Contribute with your hardware to provide computing power to the network;
2. Earn rewards for processing AI workloads;    
3. Help build a more accessible and democratic AI infrastructure.

### Community Links
- 🌐 [Official Website](https://www.atoma.network)
- 📖 [Documentation](https://atoma.gitbook.io/atoma-docs)
- 🐦 [Twitter](https://x.com/Atoma_Network)
- 💬 [Discord](https://discord.com/channels/1172593757586214964/1258484557083054081)

## Spawn an Atoma Node

### Docker Deployment

#### Quickstart

1. Configure `.env`, using as a template `.env.example`

2. Fill the `config.toml` file, using `config.example.toml` as a template

3. Start container

```
docker compose up
```

### Manual deployment 

#### 1. Installing Rust

Install Rust using rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Follow the prompts and restart your terminal. Verify the installation:

```bash
rustc --version
cargo --version
```

#### 2. Cloning the Repository

```bash
git clone https://github.com/atoma-network/atoma-node.git
cd atoma-node
```

#### 3. Configuring the Node

The application uses a TOML configuration file with the following sections:

##### `[atoma-service]`
- `inference_service_url` (optional): Endpoint URL for the inference service. At least one of the service URLs must be provided.
- `embeddings_service_url` (optional): Endpoint URL for the embeddings service. At least one of the service URLs must be provided.
- `multimodal_service_url` (optional): Endpoint URL for the multimodal service. At least one of the service URLs must be provided.
- `models`: List of model names deployed by the Atoma Service
- `revisions`: List of model revisions supported by the service
- `service_bind_address`: Address and port for the Atoma Service to bind to

##### `[atoma-sui]`
- `http_rpc_node_addr`: HTTP URL for a Sui RPC node, that the Atoma Sui's subscriber will use to listen to events on the Sui network.
- `atoma_db`: ObjectID for Atoma's DB on the Sui network
- `atoma_package_id`: ObjectID for Atoma's package on the Sui network
- `toma_package_id`: ObjectID for Atoma's TOMA token package
- `request_timeout` (optional): Duration for request timeouts
- `max_concurrent_requests` (optional): Maximum number of concurrent Sui client requests
- `limit` (optional): Limit for dynamic fields retrieval per event subscriber loop
- `node_small_ids`: List of node small IDs controlled by the current Sui wallet. Node small IDs are assigned to each node upon registration on the Atoma's smart contract.
- `task_small_ids`: List of task small IDs controlled by the current Sui wallet. Recommended to be an empty list.
- `sui_config_path`: Path to the Sui configuration file
- `sui_keystore_path`: Path to the Sui keystore file, it should be at the same directory level as the Sui configuration file.

##### `[atoma-state]`
- `database_url`: SQLite database connection URL

##### Example Configuration

```toml
[atoma-service]
inference_service_url = "<INFERENCE_SERVICE_URL>"
embeddings_service_url = "<EMBEDDINGS_SERVICE_URL>"
multimodal_service_url = "<MULTIMODAL_SERVICE_URL>"
models = ["<MODEL_1>", "<MODEL_2>"]
revisions = ["<REVISION_1>", "<REVISION_2>"]
service_bind_address = "<HOST>:<PORT>"

[atoma-sui]
http_rpc_node_addr = "<SUI_RPC_NODE_URL>"
atoma_db = "<ATOMA_DB_OBJECT_ID>"
atoma_package_id = "<ATOMA_PACKAGE_OBJECT_ID>"
toma_package_id = "<TOMA_PACKAGE_OBJECT_ID>"
request_timeout = { secs = 300, nanos = 0 }
max_concurrent_requests = 10
limit = 100
node_small_ids = [0, 1, 2]  # List of node IDs under control
task_small_ids = []  # List of task IDs under control
sui_config_path = "<PATH_TO_SUI_CONFIG>" # Example: "~/.sui/sui_config/client.yaml" (default)
sui_keystore_path = "<PATH_TO_SUI_KEYSTORE>" # Example: "~/.sui/sui_config/sui.keystore" (default)

[atoma-state]
database_url = "sqlite:///<PATH_TO_DATABASE>"
```

#### 4. Running the Atoma Node

After configuring your node, you can run it using the following command:

```bash
cargo run --bin atoma -- \
  --config-path /path/to/config.toml \
  --address-index 0 # Optional, defaults to 0
```

Or if you've built the binary:

```bash
./target/release/atoma \
  --config-path /path/to/config.toml \
  --keystore-path /path/to/sui.keystore \
  --address-index 0 # Optional, defaults to 0
```

Command line arguments:
- `--config-path` (`-c`): Path to your TOML configuration file
- `--address-index` (`-a`): Index of the address to use from the keystore (defaults to 0)

#### 5. Spawn the background inference service

We currenlty support the following inference services:

- [atoma-inference-service](https://github.com/atoma-network/atoma-inference-service)
- [vLLM](https://github.com/vllm-project/vllm)

Please refer to the documentation of the inference service you want to use to spawn the service. Make sure to set the correct inference service URL in the Atoma Node configuration, above.

#### 6. Managing Logs

The Atoma node uses a comprehensive logging system that writes to both console and files:

##### Log Location
- Logs are stored in the `./logs` directory
- The main log file is named `atoma-node-service.log`
- Logs rotate daily to prevent excessive file sizes

##### Log Formats
- **Console Output**: Human-readable format with pretty printing, ideal for development
- **File Output**: JSON format with detailed metadata, perfect for log aggregation systems

##### Log Levels
The default logging level is `info`, but you can adjust it using the `RUST_LOG` environment variable:

```bash
# Set specific log levels
export RUST_LOG=debug,atoma_node_service=trace

# Run with custom log level
RUST_LOG=debug cargo run --bin atoma -- [args]
```

Common log levels (from most to least verbose):
- `trace`: Very detailed debugging information
- `debug`: Useful debugging information
- `info`: General information about operation
- `warn`: Warning messages
- `error`: Error messages

##### Viewing Logs
You can use standard Unix tools to view and analyze logs:

```bash
# View latest logs
tail -f ./logs/atoma-node-service.log

# Search for specific events
grep "event_name" ./logs/atoma-node-service.log

# View JSON logs in a more readable format (requires jq)
cat ./logs/atoma-node-service.log | jq '.'
```

##### Log Rotation
- Logs automatically rotate daily
- Old logs are preserved with the date appended to the filename
- You may want to set up log cleanup periodically to manage disk space:

```bash
# Example cleanup script for logs older than 30 days
find ./logs -name "atoma-node-service.log.*" -mtime +30 -delete
```
