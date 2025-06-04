#!/bin/bash

# Exit on error
set -e

# Check if minikube is installed
if ! command -v minikube &> /dev/null; then
    echo "Minikube is not installed. Please install it first."
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install it first."
    exit 1
fi

# Check if helm is installed
if ! command -v helm &> /dev/null; then
    echo "Helm is not installed. Please install it first."
    exit 1
fi

# Start minikube with GPU support
echo "Starting Minikube with GPU support..."
minikube start \
    --driver=docker \
    --container-runtime=containerd \
    --feature-gates=DevicePlugins=true \
    --addons=ingress \
    --cpus=8 \
    --memory=16384 \
    --gpus=1

# Enable GPU support
echo "Enabling GPU support..."
minikube ssh "sudo nvidia-smi"

# Add required Helm repositories
echo "Adding Helm repositories..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add metallb https://metallb.github.io/metallb
helm repo add nvidia https://nvidia.github.io/k8s-device-plugin
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo add jetstack https://charts.jetstack.io
helm repo update

# Install infrastructure components
echo "Installing infrastructure components..."
helm dependency update ./helm/infrastructure
helm install infrastructure ./helm/infrastructure \
    --namespace infrastructure \
    --create-namespace \
    --wait

# Configure MetalLB for Minikube
echo "Configuring MetalLB..."
kubectl apply -f - <<EOF
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: default
  namespace: metallb-system
spec:
  addresses:
  - 192.168.49.0/24
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: default
  namespace: metallb-system
spec:
  ipAddressPools:
  - default
EOF

# Create namespace for Atoma Node
echo "Creating namespace for Atoma Node..."
kubectl create namespace atoma --dry-run=client -o yaml | kubectl apply -f -

# Create required secrets
echo "Creating required secrets..."
# Note: You'll need to provide the actual paths to your SUI config and keystore files
kubectl create secret generic atoma-node-sui-config \
    --from-file=client.yaml=/path/to/sui/client.yaml \
    --from-file=sui.keystore=/path/to/sui/sui.keystore \
    --namespace atoma \
    --dry-run=client -o yaml | kubectl apply -f -

# Create a values file for local development
echo "Creating values file for local development..."
cat > values-local.yaml <<EOF
atomaNode:
  image:
    repository: ghcr.io/atoma-network/atoma-node
    tag: latest

  config:
    environment: "development"
    heartbeatUrl: "http://localhost:3000/heartbeat"

  service:
    type: LoadBalancer

vllm:
  enabled: true
  replicas: 1  # Reduced for local testing
  model: "your-model-name"
  maxModelLen: 4096

sglang:
  enabled: true
  modelPath: "your-model-path"

prometheus:
  enabled: true
  server:
    persistentVolume:
      size: 10Gi

grafana:
  enabled: true
  persistence:
    size: 5Gi
  adminPassword: "admin"

loki:
  enabled: true
  persistence:
    size: 10Gi

tempo:
  enabled: true
  persistence:
    size: 10Gi
EOF

echo "Minikube setup complete! You can now install the Atoma Node chart:"
echo "helm install atoma-node ./helm/atoma-node -f values-local.yaml -n atoma"