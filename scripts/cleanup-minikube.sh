#!/bin/bash

# Exit on error
set -e

echo "Cleaning up Minikube environment..."

# Delete Atoma Node release
echo "Deleting Atoma Node release..."
helm uninstall atoma-node -n atoma || true

# Delete infrastructure release
echo "Deleting infrastructure release..."
helm uninstall infrastructure -n infrastructure || true

# Delete namespaces
echo "Deleting namespaces..."
kubectl delete namespace atoma || true
kubectl delete namespace infrastructure || true

# Stop Minikube
echo "Stopping Minikube..."
minikube stop

# Delete Minikube cluster
echo "Deleting Minikube cluster..."
minikube delete

echo "Cleanup complete!"