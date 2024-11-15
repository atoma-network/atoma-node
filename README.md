# Atoma Node infrastructure

<img src="https://github.com/atoma-network/atoma-node/blob/update-read-me/atoma-assets/atoma-pfp.jpg" alt="Logo" height="500"/>

[![Discord](https://img.shields.io/discord/1172593757586214964?label=Discord&logo=discord&logoColor=white)]
[![Twitter](https://img.shields.io/twitter/follow/Atoma_Network?style=social)](https://x.com/Atoma_Network)
[![Documentation](https://img.shields.io/badge/docs-gitbook-blue)](https://atoma.gitbook.io/atoma-docs)
[![License](https://img.shields.io/github/license/atoma-network/atoma-node)](LICENSE)

## Introduction

Atoma is a decentralized cloud compute network for AI that enables:

- **Verifiable Compute**: Transparent and trustworthy AI model execution, for both inference, text embeddings, multi-modality, etc, through Atoma's novel Sampling Consensus algorithm (see Atoma's [whitepaper](https://github.com/atoma-network/atoma-docs/blob/main/papers/atoma_whitepaper.pdf))
- **Private Inference**: Secure processing with strong privacy guarantees, through the use of secure hardware enclaves (see [Atoma's confidential compute paper](https://arxiv.org/abs/2410.13752))
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

### Install the Sui client locally

The first step in setting up an Atoma node is installing the Sui client locally. Please refer to the [Sui installation guide](https://docs.sui.io/build/install) for more information.

Once you have the Sui client installed, locally, you need to connect to a Sui RPC node to be able to interact with the Sui blockchain and therefore the Atoma smart contract. Please refer to the [Connect to a Sui Network guide](https://docs.sui.io/guides/developer/getting-started/connect) for more information.

You then need to create a wallet and fund it with some testnet SUI. Please refer to the [Sui wallet guide](https://docs.sui.io/guides/developer/getting-started/get-address) for more information. If you are plan to run the Atoma node on Sui's testnet, you can request testnet SUI tokens by following the [docs](https://docs.sui.io/guides/developer/getting-started/get-coins).

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

2. Configure environment variables by creating `.env` file, use `.env.example` for reference:

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
[atoma-service]
chat_completions_service_url = "http://chat-completions:8000"    # Internal Docker network URL
embeddings_service_url = "http://embeddings:80"
image_generations_service_url = "http://image-generations:80"
image_generations_service_url = ""
models = ["meta-llama/Llama-3.1-70B-Instruct"]
revisions = [""]
service_bind_address = "0.0.0.0:3000"         # Bind to all interfaces

[atoma-sui]
http_rpc_node_addr = ""
atoma_db = ""
atoma_package_id = ""
toma_package_id = ""
request_timeout = { secs = 300, nanos = 0 }
max_concurrent_requests = 10
limit = 100
node_small_ids = [0, 1, 2]  # List of node IDs under control
task_small_ids = []         # List of task IDs under control
sui_config_path = "/root/.sui/sui_config/client.yaml"
sui_keystore_path = "/root/.sui/sui_config/sui.keystore"

[atoma-state]
database_url = "sqlite:///app/data/atoma.db"
```

4. Create required directories

```bash
mkdir -p data logs
```

5. Start the containers with the desired inference services

We currenlty support the following inference services:

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

```bash
# Build and start all services
COMPOSE_PROFILES=chat_completions_vllm,embeddings_tei,image_generations_mistralrs docker compose up --build

# Only start one service
COMPOSE_PROFILES=chat_completions_vllm docker compose up --build

# Run in detached mode
COMPOSE_PROFILES=chat_completions_vllm,embeddings_tei,image_generations_mistralrs docker compose up -d --build
```

#### Container Architecture

The deployment consists of two main services:

- **vLLM Service**: Handles the AI model inference
- **Atoma Node**: Manages the node operations and connects to the Atoma Network

#### Service URLs

- vLLM Service: `http://localhost:50000` (configured via INFERENCE_SERVER_PORT)
- Atoma Node: `http://localhost:3000` (configured via ATOMA_SERVICE_PORT)

#### Volume Mounts

- HuggingFace cache: `~/.cache/huggingface:/root/.cache/huggingface`
- Sui configuration: `~/.sui/sui_config:/root/.sui/sui_config`
- Logs: `./logs:/app/logs`
- SQLite database: `./data:/app/data`

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
docker compose logs atoma-node
docker compose logs vllm

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

2. Test vLLM service:

```bash
curl http://localhost:50000/health
```

3. Test Atoma Node service:

```bash
curl http://localhost:3000/health
```

4. Check GPU availability:

```bash
docker compose exec vllm nvidia-smi
```

5. View container networks:

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

2. HuggingFace Token

- Store HF_TOKEN in .env file
- Never commit .env file to version control
- Consider using Docker secrets for production deployments

3. Sui Configuration

- Ensure Sui configuration files have appropriate permissions
- Keep keystore file secure and never commit to version control

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
