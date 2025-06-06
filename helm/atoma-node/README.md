# Atoma Node Helm Chart

This Helm chart deploys the Atoma Node and its dependencies, including VLLM and SGLang inference servers, along with monitoring stack (Prometheus, Grafana, Loki, and Tempo).

## Prerequisites

- Kubernetes cluster with GPU support (for VLLM and SGLang)
- Helm 3.x
- NVIDIA device plugin installed in the cluster
- Storage class with sufficient capacity for persistent volumes
- MetalLB (for LoadBalancer services)
- Ingress NGINX (for external access)
- Cert-Manager (for TLS certificates)

## Infrastructure Components

The infrastructure components are managed by a separate Helm chart in the `infrastructure` directory. These components include:

- MetalLB: For LoadBalancer services
- NVIDIA Device Plugin: For GPU support
- Ingress NGINX: For external access
- Cert-Manager: For TLS certificate management

To install the infrastructure components:

```bash
# Add required repositories
helm repo add metallb https://metallb.github.io/metallb
helm repo add nvidia https://nvidia.github.io/k8s-device-plugin
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo add jetstack https://charts.jetstack.io
helm repo update

# Install infrastructure
helm dependency update ./helm/infrastructure
helm install infrastructure ./helm/infrastructure \
    --namespace infrastructure \
    --create-namespace
```

## Local Testing with Minikube

For local testing, you can use the provided scripts to set up a Minikube environment:

1. Make the scripts executable:
```bash
chmod +x scripts/setup-minikube.sh
chmod +x scripts/cleanup-minikube.sh
```

2. Run the setup script:
```bash
./scripts/setup-minikube.sh
```

3. Install the Atoma Node chart:
```bash
helm install atoma-node ./helm/atoma-node -f values-local.yaml -n atoma
```

4. To clean up:
```bash
./scripts/cleanup-minikube.sh
```

Note: The Minikube setup script will:
- Start Minikube with GPU support
- Install all required infrastructure components
- Configure MetalLB for LoadBalancer services
- Create necessary namespaces and secrets
- Generate a values file for local development

## Installation

1. Add the required Helm repositories:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
```

2. Create a values file for your environment (e.g., `my-values.yaml`):

```yaml
# Example values file
atomaNode:
  image:
    repository: ghcr.io/atoma-network/atoma-node
    tag: latest

  config:
    environment: "production"
    heartbeatUrl: "your-heartbeat-url"
    # Add other configuration as needed

vllm:
  enabled: true
  replicas: 8  # Adjust based on your GPU count
  model: "your-model-name"
  maxModelLen: 4096

sglang:
  enabled: true
  modelPath: "your-model-path"

# Configure monitoring stack
prometheus:
  enabled: true
  server:
    persistentVolume:
      size: 50Gi

grafana:
  enabled: true
  adminPassword: "your-secure-password"

loki:
  enabled: true
  persistence:
    size: 50Gi

tempo:
  enabled: true
  persistence:
    size: 50Gi
```

3. Create required secrets:

```bash
# Create SUI config secret
kubectl create secret generic atoma-node-sui-config \
  --from-file=client.yaml=/path/to/sui/client.yaml \
  --from-file=sui.keystore=/path/to/sui/sui.keystore

# Create SGLang secrets
kubectl create secret generic atoma-node-sglang-secrets \
  --from-literal=hf-token=your-huggingface-token
```

4. Install the chart:

```bash
helm install atoma-node ./helm/atoma-node \
  -f my-values.yaml \
  --namespace atoma \
  --create-namespace
```

## Configuration

### Atoma Node

The Atoma Node can be configured through the `atomaNode` section in values:

- `image`: Container image configuration
- `resources`: Resource requests and limits
- `service`: Service configuration
- `config`: Application configuration
- `persistence`: Storage configuration

### VLLM

VLLM inference servers can be configured through the `vllm` section:

- `enabled`: Enable/disable VLLM deployment
- `replicas`: Number of VLLM instances
- `resources`: Resource configuration including GPU requests
- `model`: Model name to load
- `maxModelLen`: Maximum model context length

### SGLang

SGLang inference server can be configured through the `sglang` section:

- `enabled`: Enable/disable SGLang deployment
- `resources`: Resource configuration including GPU requests
- `modelPath`: Path to the model

### Monitoring Stack

The monitoring stack includes:

- Prometheus: Metrics collection
- Grafana: Metrics visualization
- Loki: Log aggregation
- Tempo: Distributed tracing

Each component can be enabled/disabled and configured through their respective sections in values.

## Accessing Services

- Atoma Node API: `http://atoma-node:3000`
- Grafana: `http://grafana:3000`
- Prometheus: `http://prometheus:9090`
- Loki: `http://loki:3100`
- Tempo: `http://tempo:3200`

## Upgrading

To upgrade the deployment:

```bash
helm upgrade atoma-node ./helm/atoma-node \
  -f my-values.yaml \
  --namespace atoma
```

## Uninstalling

To uninstall the deployment:

```bash
helm uninstall atoma-node --namespace atoma
```

## Troubleshooting

1. Check pod status:
```bash
kubectl get pods -n atoma
```

2. Check pod logs:
```bash
kubectl logs -n atoma <pod-name>
```

3. Check persistent volume claims:
```bash
kubectl get pvc -n atoma
```

4. Check services:
```bash
kubectl get svc -n atoma
```

5. Check GPU availability:
```bash
kubectl describe node | grep nvidia.com/gpu
```

## Notes

- Ensure your cluster has sufficient GPU resources for VLLM and SGLang
- Adjust resource requests and limits based on your cluster capacity
- Configure appropriate storage classes for persistent volumes
- Set up proper network policies for security
- Consider using an ingress controller for external access
- For local testing, use the provided Minikube scripts
- Make sure MetalLB is properly configured for LoadBalancer services
