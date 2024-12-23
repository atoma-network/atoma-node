#!/bin/bash

# 1) Ensure the container's config directory exists
mkdir -p /root/.sui/sui_config

# 2) If the named volume is empty, copy the entire folder from the host
if [ -z "$(ls -A /root/.sui/sui_config)" ] && [ -d /tmp/.sui/sui_config ]; then
    echo "Seeding /root/.sui/sui_config from /tmp/.sui/sui_config..."
    cp -r /tmp/.sui/sui_config/. /root/.sui/sui_config/
fi

# 3) Further modify specific files in the container:
if [ -f /root/.sui/sui_config/client.yaml ]; then
    echo "Modifying client.yaml..."
    sed -i 's|File: .*|File: /root/.sui/sui_config/sui.keystore|' /root/.sui/sui_config/client.yaml
fi

# 4) Run the main command/args
exec "$@"
