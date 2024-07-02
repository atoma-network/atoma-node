# Atoma Node infrastructure

![Atoma Logo](https://github.com/atoma-network/atoma-node/blob/atoma-image/atoma-assets/atoma-symbol.jpg)

## Introduction

The present repository contains the logic to run an Atoma node. Atoma nodes empower the Atoma Network, a decentralized network
for verifiable AI inference. Nodes can lend their GPU compute to the protocol, so large language models (LLMs for short) can be
run across a decentralized network, at a cheaper cost. Moreover, nodes can be organized in order to provide verifiability guarantees
of the correctness of their generated outputs. This means that the Atoma Network can empower smart contracts, deployed on any blockchain,
to run verifiable inference and guarantee an intelligence layer to Web3. 

## Run a node

In order to run an Atoma node, you should provide enough evidence of holding a powerful enough machine for AI inference. We
allow a wide range of possible hardware, including NVIDIA GPU series (RTX3090, RTX4090, A100, etc), as well as a good enough
Internet bandwidth connection. We further highly encourage nodes to register on Atoma contracts on supported blockchains
(these include Arbitrum, Solana, Sui, etc). In order to register, we suggest the reader to follow the instructions in the
Atoma contract [repo](https://github.com/atoma-network/atoma-contracts).

Once registration has been completed, the user is required to:

1. Clone this repo (it is assumed that Rust is installed, otherwise follow the instructions [here](https://www.rust-lang.org/tools/install)).
2. Create configuration files for both the model inference service, the event listener service and the blockchain client service. Notice
that the event listener and blockchain client services need to be for the same blockchain, in which the node has previously registered. That said, a single node can be registered in multiple blockchains and listen to events on each of these (for higher accrued rewards).
3. The model inference service follows schematically (in toml format):

```toml
api_key = "YOUR_HUGGING_FACE_API_KEY" # for downloading models
cache_dir = "PATH_FOR_MODEL_STORAGE" # where you want the downloaded models to be stored
flush_storage = FLUSH_STORAGE # bool value, when the user stops the Atoma node, it flushes or not the downloaded models
jrpc_port = JRPC_PORT # Atoma node port for JSON rpc service
models = [MODEL_CONFIG] # Specifications for each model the user wants to operate, as an Atoma Node
tracing = TRACING # bool value, allows for tracing
```

4. In the above paragraph, the `MODEL_CONFIG` refers to a set of supported model configurations, see below, as follows

```
[DEVICE, PRECISION, MODEL_TYPE, USE_FLASH_ATTENTION]
```

- `DEVICE` is an integer referring to the device index of GPU operating (if your machine only supports one single GPU cards, device should be 0, or if using cpu or metal devices). 
- `PRECISION` refers to the model inference precision, supported values are
`"f32"`, `"bf16"`, `"f16"` (if you host quantized models, this field is not relevant). 
- `MODEL_TYPE` is a string referring to the name
of the model to be hosted (on the given device), a full list of model names can be found here. 
- `USE_FLASH_ATTENTION` is a boolean value which allows to run inference with the optimized flash attention algorithm.

5. The event subscriber service configuration file is specified as (in toml format):

```toml
http_url = "RPC_NODE_HTTP_URL" # to connect via http to a rpc node on the blockchain
ws_url = "RPC_NODE_WEB_SOCKET_URL" # to connect via web socket to a rpc node on the blockchain, relevant for listening to events
package_id = "SUI_PACKAGE_ID" # the Atoma contract object id, on Sui.
small_id = JRPC_PORT # a unique identifier provided to the node, upon on-chain registration

[request_timeout] # a request timeout parameter
secs = 300
nanos = 0
```

6. The Atoma blockchain client service configuration file is specified as (in toml format):

```toml
config_path = "SUI_CLIENT_CONFIG_PATH" # the path to the sui client configuration path (for connecting the user's wallet)
atoma_db_id = "ATOMA_DB_ID" # the Atoma db object id, this value is publicly available (see below)
node_badge_id = "NODE_BADGE_ID" # the node's own badge object id, this value is provided to the node upon registration
package_id = "ATOMA_CALL_CONTRACT_ID" # the Atoma's contract package id, this value is publicly available (see below)
small_id = SMALL_ID # a unique identifier provided to the node, upon on-chain registration (same as above)

max_concurrent_requests = 1000 # how many concurrent requests are supported by the Sui's client service

[request_timeout] # a request timeout parameter
secs = 300
nanos = 0
```

7. Once the node is registered and the configuration files set, the node then just needs to run the following commands:

```sh
$ cd atoma-node
$ RUST_LOG=info cargo run --release --features <YOUR_GPU_ENV> -- --atoma-sui-client-config-path <PATH_TO_ATOMA_SUI_CLIENT_CONFIG> --model-config-path <PATH_TO_MODEL_CONFIG> --sui-subscriber-path <PATH_TO_SUI_EVENT_SUBSCRIBER_CONFIG>
```

The value `YOUR_GPU_ENV` could be either `cuda`, `metal`, `flash-attn` or if you wish to run inference on the CPU, remove the `--features <YOUR_GPU_ENV>`. If you set `use_flash_attention = true` in 4. above, you should execute the binary as

```sh
$ 
RUST_LOG=info cargo run --release --features flash-attn -- --atoma-sui-client-config-path <PATH_TO_ATOMA_SUI_CLIENT_CONFIG> --model-config-path <PATH_TO_MODEL_CONFIG> --sui-subscriber-path <PATH_TO_SUI_EVENT_SUBSCRIBER_CONFIG>
```

## Supported models

The supported models currently are:

| Model Type                         | Hugging Face model name                  |
|------------------------------------|------------------------------------------|
| falcon_7b                          | tiiuae/falcon-7b                         |
| falcon_40b                         | tiiuae/falcon-40b                        |
| falcon_180b                        | tiiuae/falcon-180b                       |
| llama_v1                           | Narsil/amall-7b                          |
| llama_v2                           | meta-llama/Llama-2-7b-hf                 |
| llama_solar_10_7b                  | upstage/SOLAR-10.7B-v1.0                 |
| llama_tiny_llama_1_1b_chat         | TinyLlama/TinyLlama-1.1B-Chat-v1.0       |
| llama3_8b                          | meta-llama/Meta-Llama-3-8B               |
| llama3_instruct_8b                 | meta-llama/Meta-Llama-3-8B-Instruct      |
| llama3_70b                         | meta-llama/Meta-Llama-3-70B              |
| mamba_130m                         | state-spaces/mamba-130m                  |
| mamba_370m                         | state-spaces/mamba-370m                  |
| mamba_790m                         | state-spaces/mamba-790m                  |
| mamba_1-4b                         | state-spaces/mamba-1.4b                  |
| mamba_2-8b                         | state-spaces/mamba-2.8b                  |
| mistral_7bv01                      | mistralai/Mistral-7B-v0.1                |
| mistral_7bv02                      | mistralai/Mistral-7B-v0.2                |
| mistral_7b-instruct-v01            | mistralai/Mistral-7B-Instruct-v0.1       |
| mistral_7b-instruct-v02            | mistralai/Mistral-7B-Instruct-v0.2       |
| mixtral_8x7b-v01                   | mistralai/Mixtral-8x7B-v0.1              |
| mixtral_8x7b-instruct-v01          | mistralai/Mixtral-8x7B-Instruct-v0.1     |
| mixtral_8x22b-v01                  | mistralai/Mixtral-8x22B-v0.1             |
| mixtral_8x22b-instruct-v01         | mistralai/Mixtral-8x22B-Instruct-v0.1    |
| phi_3-mini                         | microsoft/Phi-3-mini-4k-instruct         |
| stable_diffusion_v1-5              | runwayml/stable-diffusion-v1-5           |
| stable_diffusion_v2-1              | stabilityai/stable-diffusion-2-1         |
| stable_diffusion_xl                | stabilityai/stable-diffusion-xl-base-1.0 |
| stable_diffusion_turbo             | stabilityai/sdxl-turbo                   |
| quantized_7b                       | TheBloke/Llama-2-7B-GGML                 |
| quantized_13b                      | TheBloke/Llama-2-13B-GGML                |
| quantized_70b                      | TheBloke/Llama-2-70B-GGML                |
| quantized_7b-chat                  | TheBloke/Llama-2-7B-Chat-GGML            |
| quantized_13b-chat                 | TheBloke/Llama-2-13B-Chat-GGML           |
| quantized_70b-chat                 | TheBloke/Llama-2-70B-Chat-GGML           |
| quantized_7b-code                  | TheBloke/CodeLlama-7B-GGUF               |
| quantized_13b-code                 | TheBloke/CodeLlama-13B-GGUF              |
| quantized_32b-code                 | TheBloke/CodeLlama-34B-GGUF              |
| quantized_7b-leo                   | TheBloke/leo-hessianai-7B-GGUF           |
| quantized_13b-leo                  | TheBloke/leo-hessianai-13B-GGUF          |
| quantized_7b-mistral               | TheBloke/Mistral-7B-v0.1-GGUF            |
| quantized_7b-mistral-instruct      | TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF |
| quantized_7b-mistral-instruct-v0.2 | TheBloke/Mistral-7B-Instruct-v0.2-GGUF   |
| quantized_7b-zephyr-a              | TheBloke/zephyr-7B-alpha-GGUF            |
| quantized_7b-zephyr-b              | TheBloke/zephyr-7B-beta-GGUF             |
| quantized_7b-open-chat-3.5         | TheBloke/openchat_3.5-GGUF               |
| quantized_7b-starling-a            | TheBloke/Starling-LM-7B-alpha-GGUF       |
| quantized_mixtral                  | TheBloke/Mixtral-8x7B-v0.1-GGUF          |
| quantized_mixtral-instruct         | TheBloke/Mistral-7B-Instruct-v0.1-GGUF   |
| quantized_llama3-8b                | QuantFactory/Meta-Llama-3-8B-GGUF        |
| qwen_w0.5b                         | Qwen/Qwen1.5-0.5B                        |
| qwen_w1.8b                         | Qwen/Qwen1.5-1.8B                        |
| qwen_w4b                           | Qwen/Qwen1.5-4B                          |
| qwen_w7b                           | qwen/Qwen1.5-7B                          |
| qwen_w14b                          | qwen/Qwen1.5-14B                         |
| qwen_w72b                          | qwen/Qwen1.5-72B                         |
| qwen_moe_a2.7b                     | qwen/Qwen1.5-MoE-A2.7B                   |


For example, if a user wants to run a node hosting a quantized 7b mistral model, it can do so simply by setting

```
MODEL_TYPE = "quantized_7b-mistral"
```
