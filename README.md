# Atoma Node Infrastructure

![Atoma Banner](https://github.com/atoma-network/atoma-node/blob/ja-improve-readme/atoma-assets/atoma-banner.png)

[![Discord](https://img.shields.io/discord/1172593757586214964?label=Discord&logo=discord&logoColor=white)](https://discord.com/channels/1172593757586214964/1258484557083054081)
[![Twitter](https://img.shields.io/twitter/follow/Atoma_Network?style=social)](https://x.com/Atoma_Network)
[![Documentation](https://img.shields.io/badge/docs-mintify-blue?logo=mintify)](https://docs.atoma.network)
[![License](https://img.shields.io/github/license/atoma-network/atoma-node)](LICENSE)

## Introduction

Atoma is a decentralized cloud compute network for AI that enables:

- **Verifiable Compute**: Transparent and trustworthy AI model execution, for both inference, text embeddings, multi-modality, etc, through Atoma's novel Sampling Consensus algorithm (see Atoma's [whitepaper](https://github.com/atoma-network/atoma-docs/blob/main/papers/atoma_whitepaper.pdf))
- **Private Inference**: Secure processing with strong privacy guarantees, through the use of secure hardware enclaves (see [Atoma's confidential compute paper](https://arxiv.org/abs/2410.13752))
- **Decentralized Infrastructure**: A permissionless network of compute nodes, orchestrated by Atoma's smart contract on the Sui blockchain (see [repo](https://github.com/atoma-network/atoma-contracts)). It includes payments, request authentication, load balancing, and more
- **Governance**: Atoma's governance is fully decentralized, with all the network participants being able to vote on the future of the network.
- **LLM Focus**: Specialized in serving Large Language Models compute, through a fully compatible OpenAI API.
- **Application Layer**: Atoma's node software is designed to be modular and easy to integrate with other AI services. In particular, you can build any AI application at scale, through the Atoma's API. This includes AI agents, chatbots, image generation applications, personal assistants, etc. All of these applications can leverage the best in class open source LLM models, offering full data privacy and security to the end user.

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

### Install the Sui client locally

The first step in setting up an Atoma node is installing the Sui client locally. Please refer to the [Sui installation guide](https://docs.sui.io/build/install) for more information.

Once you have the Sui client installed, locally, you need to connect to a Sui RPC node to be able to interact with the Sui blockchain and therefore the Atoma smart contract. Please refer to the [Connect to a Sui Network guide](https://docs.sui.io/guides/developer/getting-started/connect) for more information.

You then need to create a wallet and fund it with some testnet SUI. Please refer to the [Sui wallet guide](https://docs.sui.io/guides/developer/getting-started/get-address) for more information. If you plan to run the Atoma node on Sui's testnet, you can request testnet SUI tokens by following the [docs](https://docs.sui.io/guides/developer/getting-started/get-coins).

### Register with the Atoma Testnet smart contract

Please refer to the [setup script](https://github.com/atoma-network/atoma-contracts/pull/132/files#diff-00fdca2a8379052d26447658c20d8c7e93f8b7468979e4c96704c55e91406b96) to register with the Atoma Testnet smart contract. This will assign you a node badge and a package ID, which you'll need to configure in the `config.toml` file.

### Docker Deployment

#### Prerequisites

- Docker and Docker Compose (>= v2.22) installed
- NVIDIA Container Toolkit installed (for GPU support)
- Access to HuggingFace models (and token if using gated models)
- Sui wallet configuration

#### Quickstart

1. Clone the repository

```bash
git clone https://github.com/atoma-network/atoma-node.git
cd atoma-node
```

1. Configure environment variables by creating `.env` file, you'll need a hugging face token use `.env.example` for reference:

```bash
# Hugging Face Configuration
HF_CACHE_PATH=~/.cache/huggingface
HF_TOKEN=   # Required for gated models

# Inference Server Configuration
INFERENCE_SERVER_PORT=50000    # External port for vLLM service
MODEL=meta-llama/Llama-3.1-70B-Instruct
MAX_MODEL_LEN=4096            # Context length
GPU_COUNT=1                   # Number of GPUs to use
TENSOR_PARALLEL_SIZE=1        # Should be equal to GPU_COUNT

# Sui Configuration
SUI_CONFIG_PATH=~/.sui/sui_config

# Atoma Node Service Configuration
ATOMA_SERVICE_PORT=3000       # External port for Atoma service
```

3. Configure `config.toml`, using `config.example.toml` as template:

```toml
[atoma_service]
chat_completions_service_url = "http://chat-completions:8000" # Internal Docker network URL
embeddings_service_url = "http://embeddings:80"
image_generations_service_url = "http://image-generations:80"
# List of models to be used by the service, the current value here is just a placeholder, please change it to the models you want to deploy
models = ["meta-llama/Llama-3.2-3B-Instruct"]
revisions = ["main"]
service_bind_address = "0.0.0.0:3000"

[atoma_sui]
http_rpc_node_addr = "https://fullnode.testnet.sui.io:443"                              # Current RPC node address for testnet
atoma_db = "0x7b8f40e38698deb650519a51f9c1a725bf8cfdc074d1552a4dc85976c2b414be"         # Current ATOMA DB object ID for testnet
atoma_package_id = "0xc05bae323433740c969d8cf938c48d7559490be5f8dde158792e7a0623787013" # Current ATOMA package ID for testnet
usdc_package_id = "0xa1ec7fc00a6f40db9693ad1415d0c193ad3906494428cf252621037bd7117e29"  # Current USDC package ID for testnet
request_timeout = { secs = 300, nanos = 0 }                                             # Some reference value
max_concurrent_requests = 10                                                            # Some reference value
limit = 100                                                                             # Some reference value
node_small_ids = [1]                                                                    # List of node IDs under control of the node wallet
sui_config_path = "/root/.sui/sui_config/client.yaml"                                   # Path to the Sui client configuration file, accessed from the docker container (if this is not the case, pass in the full path, on your host machine which is by default ~/.sui/sui_config/client.yaml)
sui_keystore_path = "/root/.sui/sui_config/sui.keystore"                                # Path to the Sui keystore file, accessed from the docker container (if this is not the case, pass in the full path, on your host machine which is by default ~/.sui/sui_config/sui.keystore)
cursor_path = "./cursor.toml"                                                           # Path to the Sui events cursor file

[atoma_state]
# Path inside the container
# Replace the placeholder values with the ones for your local environment (in the .env file)
database_url = "postgres://<POSTGRES_USER>:<POSTGRES_PASSWORD>@postgres-db:5432/<POSTGRES_DB>"

[atoma_daemon]
# WARN: Do not expose this port to the public internet, as it is used only for internal communication between the Atoma Node and the Atoma Network
service_bind_address = "0.0.0.0:3001"
# Replace the placeholder values with the actual node badge and small ID assigned by the Atoma's smart contract, upon node registration
node_badges = [
    [
        "0x268e6af9502dcdcaf514bb699c880b37fa1e8d339293bc4f331f2dde54180600",
        1,
    ],
] # List of node badges, where each badge is a tuple of (badge_id, small_id), both values are assigned once the node registers itself

[proxy_server]
# replace this with the public url address of the Atoma proxy server (currently https://api.atomacloud.com)
proxy_address = "https://api.atomacloud.com"
# replace this with the public url address of this node
node_public_address = ""
# replace this with the country code (ISO 3166-1 alpha-2) of the node
country = ""
```

1. Create required directories

```bash
mkdir -p data logs
```

1. Start the containers with the desired inference services, please note if you don't have a GPU, you'll need to use the you will need to use a `vllm_cpu`  backend, but these are only compatible with  `x86_64` architectures. Otherwise we recommend using the `mistral` for CPUs.

We currently support the following inference services:

##### Chat Completions

| Backend                                                  | Architecture/Platform | Docker Compose Profile           |
| -------------------------------------------------------- | --------------------- | -------------------------------- |
| [vLLM](https://github.com/vllm-project/vllm)             | CUDA                  | `chat_completions_vllm`          |
| [vLLM](https://github.com/vllm-project/vllm)             | x86_64                | `chat_completions_vllm_cpu`      |
| [vLLM](https://github.com/vllm-project/vllm)             | ROCm                  | `chat_completions_vllm_rocm`     |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | x86_64, aarch64       | `chat_completions_mistralrs_cpu` |

##### Embeddings

| Backend                                                                               | Architecture/Platform | Docker Compose Profile |
| ------------------------------------------------------------------------------------- | --------------------- | ---------------------- |
| [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) | CUDA                  | `embeddings_tei`       |

##### Image Generations

| Backend                                                  | Architecture/Platform | Docker Compose Profile        |
| -------------------------------------------------------- | --------------------- | ----------------------------- |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | CUDA                  | `image_generations_mistralrs` |

Additionally, we offer the flexibility to run Atoma's node in two different modes:

- **Confidential**: This mode allows to run the Atoma node infrastructure in a confidential mode, meaning that the node will only be able to process requests that have been authenticated by the Atoma's smart contract, through secure hardware enclaves, allowing for full data privacy and security. This mode is the most secure one, and it is the recommended mode for most applications, but it requires operating on the latest hardware (e.g. NVIDIA Hopper and Blackwell GPUs).
- **Non-Confidential**: This mode is the default one, and it runs the Atoma's node in a non-confidential mode, meaning that the node will be able to process requests without any further privacy guarantees, even though Atoma still offers strong compute integrity guarantees through our novel [Sampling Consensus algorithm](https://github.com/atoma-network/atoma-docs/blob/main/papers/atoma_whitepaper.pdf).

To run the Atoma node in a confidential mode, you need to pass the `confidential` profile to the `docker compose up` command:

```bash
# Build and start all services
COMPOSE_PROFILES=chat_completions_vllm,embeddings_tei,image_generations_mistralrs,confidential docker compose up --build

# Only start one service
COMPOSE_PROFILES=chat_completions_vllm,confidential docker compose up --build

# Run in detached mode
COMPOSE_PROFILES=chat_completions_vllm,embeddings_tei,image_generations_mistralrs,confidential docker compose up -d --build
```

Similarly, to run the Atoma node in a non-confidential mode, you need to pass the `non-confidential` profile to the `docker compose up` command:

```bash
# Build and start all services
COMPOSE_PROFILES=chat_completions_vllm,embeddings_tei,image_generations_mistralrs,non-confidential docker compose up --build

# Only start one service
COMPOSE_PROFILES=chat_completions_vllm,non-confidential docker compose up --build

# Run in detached mode
COMPOSE_PROFILES=chat_completions_vllm,embeddings_tei,image_generations_mistralrs,non-confidential docker compose up -d --build
```

#### Container Architecture

The deployment consists of two main services:

- **LLM Inference Service**: Handles the AI model inference
- **Atoma Node**: Manages the node operations and connects to the Atoma Network

#### Service URLs
Atoma Node: `http://localhost:3000` (configured via ATOMA_SERVICE_PORT). You are free to change the port to any other available port, as long as it is not already in use by another service. Moreover, in order for your node to be accessible by the Atoma Network, you need to make sure that the port is open to the public internet, through your router's firewall and NAT configuration. Moreover, it is recommended to use a static IP address for your node, in order to avoid having to reconfigure your router's NAT table every time you restart your node. The Atoma Node service handles all the required authentication and authorization for the LLM Inference Service, ensuring that only authenticated (and already paid for) requests are processed.


#### Volume Mounts

- HuggingFace cache: `~/.cache/huggingface:/root/.cache/huggingface`
- Sui configuration: `~/.sui/sui_config:/root/.sui/sui_config`
- Logs: `./logs:/app/logs`
- PostgreSQL database: `./data:/app/data`

#### Managing the Deployment

Check service status:

```bash
docker compose ps
```

View logs:

```bash
# All services
docker compose logs

# Specific service
docker compose logs atoma-node-confidential # Confidential mode
docker compose logs atoma-node-non-confidential # Non-confidential mode
docker compose logs vllm # vLLM service

# Follow logs
docker compose logs -f
```

Stop services:

```bash
docker compose down
```

#### Troubleshooting

1. Check if services are running:

```bash
docker compose ps
```

1. Test vLLM service:

```bash
curl http://localhost:50000/health
```

1. Test Atoma Node service:

```bash
curl http://localhost:3000/health
```

1. Check GPU availability:

```bash
docker compose exec vllm nvidia-smi
```

1. View container networks:

```bash
docker network ls
docker network inspect atoma-network
```

#### Security Considerations

1. Firewall Configuration

```bash
# Allow Atoma service port
sudo ufw allow 3000/tcp

# Allow vLLM service port
sudo ufw allow 50000/tcp
```

1. HuggingFace Token

- Store HF_TOKEN in .env file
- Never commit .env file to version control
- Consider using Docker secrets for production deployments

1. Sui Configuration

- Ensure Sui configuration files have appropriate permissions
- Keep keystore file secure and never commit to version control

### Testing

Since the `AtomaStateManager` instance relies on a PostgreSQL database, we need to have a local instance running to run the tests. You can spawn one using the `docker-compose.test.yaml` file:

```bash
docker compose -f docker-compose.test.yaml up --build -d
```

It might be necessary that you clean up the database before or after running the tests. You can do so by running:

```bash
docker compose -f docker-compose.test.yaml down
```

and remove the specific postgres volumes:

```bash
docker system prune -af --volumes
```

Notice that by running the above commands you will lose all the data stored in the database.

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

- `chat_completions_service_url` (optional): Endpoint URL for the inference service. At least one of the service URLs must be provided.
- `embeddings_service_url` (optional): Endpoint URL for the embeddings service. At least one of the service URLs must be provided.
- `image_generations_service_url` (optional): Endpoint URL for the image generations service. At least one of the service URLs must be provided.
- `models`: List of model names deployed by the Atoma Service
- `revisions`: List of model revisions supported by the service
- `service_bind_address`: Address and port for the Atoma Service to bind to

##### `[atoma-sui]`

- `http_rpc_node_addr`: HTTP URL for a Sui RPC node, that the Atoma Sui's subscriber will use to listen to events on the Sui network.
- `atoma_db`: ObjectID for Atoma's DB on the Sui network
- `atoma_package_id`: ObjectID for Atoma's package on the Sui network
- `usdc_package_id`: ObjectID for USDC token package
- `request_timeout` (optional): Duration for request timeouts
- `max_concurrent_requests` (optional): Maximum number of concurrent Sui client requests
- `limit` (optional): Limit for dynamic fields retrieval per event subscriber loop
- `node_small_ids`: List of node small IDs controlled by the current Sui wallet. Node small IDs are assigned to each node upon registration on the Atoma's smart contract.
- `task_small_ids`: List of task small IDs controlled by the current Sui wallet. Recommended to be an empty list.
- `sui_config_path`: Path to the Sui configuration file
- `sui_keystore_path`: Path to the Sui keystore file, it should be at the same directory level as the Sui configuration file.

##### `[atoma-state]`

- `database_url`: PostgreSQL database connection URL

##### Example Configuration

```toml
[atoma-service]
chat_completions_service_url = "<chat_completions_service_url>"
embeddings_service_url = "<EMBEDDINGS_SERVICE_URL>"
image_generations_service_url = "<image_generations_service_url>"
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
# Path inside the container
# Replace the placeholder values with the ones for your local environment (in the .env file)
database_url = "postgres://<POSTGRES_USER>:<POSTGRES_PASSWORD>@localhost:5432/<POSTGRES_DB>"
```

#### 4. Running the Atoma Node

After configuring your node, you can run it using the following command:

```bash
RUST_LOG=debug cargo run --release --bin atoma-node -- \
  --config-path /path/to/config.toml
```

Or if you've built the binary:

```bash
./target/release/atoma-node \
  --config-path /path/to/config.toml
```

Command line arguments:

- `--config-path` (`-c`): Path to your TOML configuration file
- `--address-index` (`-a`): Index of the address to use from the keystore (defaults to 0)

#### 5. Spawn the background inference service

We currently support the following inference services:

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
RUST_LOG=debug cargo run --release --bin atoma-node -- [args]
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
