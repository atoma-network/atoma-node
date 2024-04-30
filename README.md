# Atoma Node infrastructure

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
api_key = "<YOUR_HUGGING_FACE_API_KEY>" # for downloading models
cache_dir = "<PATH_FOR_MODEL_STORAGE>" # where you want the downloaded models to be stored
flush_storage = true # when the user stops the Atoma node, it flushes or not the downloaded models
jrpc_port = 3000 # Atoma node JRPC port
models = [[device, precision, model_type, revision, use_flash_attention], ...] # Specifications for each model the user wants to operate, as an Atoma Node
tracing = true # Allows for tracing
```
4. The event subscriber service configuration file is specified as (in toml format):

```toml
http_url = "RPC_NODE_HTTP_URL" # to connect via http to a rpc node on the blockchain
ws_url = "RPC_NODE_WEB_SOCKET_URL" # to connect via web socket to a rpc node on the blockchain, relevant for listening to events
package_id = "SUI_PACKAGE_ID" # the Atoma contract object id, on Sui.
small_id = 28972375 # a unique identifier provided to the node, upon on-chain registration

[request_timeout] # a request timeout parameter
secs = 300
nanos = 0

```