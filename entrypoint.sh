#!/bin/bash

# Ensure the directory exists
mkdir -p /root/.sui/sui_config

# If client.yaml exists, modify it
if [ -f /root/.sui/sui_config/client.yaml ]; then
    sed -i 's|File: .*|File: /root/.sui/sui_config/sui.keystore|' /root/.sui/sui_config/client.yaml
fi

# Execute the main command passed to the container
exec "$@"
