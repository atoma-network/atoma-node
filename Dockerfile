# Builder stage
FROM --platform=$BUILDPLATFORM rust:1.76-slim-bullseye AS builder

ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TRACE_LEVEL

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    libc6-dev-arm64-cross \
    && rm -rf /var/lib/apt/lists/*

# Set up cross-compilation
RUN case "$TARGETPLATFORM" in \
    "linux/arm64") \
    echo "aarch64-unknown-linux-gnu" > /rust_target.txt && \
    export PKG_CONFIG_ALLOW_CROSS=1 && \
    export OPENSSL_DIR=/usr/aarch64-linux-gnu && \
    export OPENSSL_LIB_DIR=/usr/lib/aarch64-linux-gnu && \
    export OPENSSL_INCLUDE_DIR=/usr/include/aarch64-linux-gnu \
    ;; \
    "linux/amd64") \
    echo "x86_64-unknown-linux-gnu" > /rust_target.txt && \
    export PKG_CONFIG_ALLOW_CROSS=1 && \
    export OPENSSL_DIR=/usr/x86_64-linux-gnu && \
    export OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu && \
    export OPENSSL_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
    ;; \
    *) exit 1 ;; \
    esac

RUN rustup target add $(cat /rust_target.txt)

WORKDIR /usr/src/atoma-node

COPY . .

# Build the application
RUN RUST_LOG=${TRACE_LEVEL} cargo build --release --bin atoma-node --target $(cat /rust_target.txt)

# Final stage
FROM --platform=$TARGETPLATFORM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Copy the built binary from builder stage
COPY --from=builder /usr/src/atoma-node/target/release/atoma-node /usr/local/bin/atoma-node

# Copy configuration file
COPY config.toml ./config.toml

RUN chmod +x /usr/local/bin/atoma-node

# Copy and set up entrypoint script
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy host client.yaml and modify keystore path
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Use full path in CMD
CMD ["/usr/local/bin/atoma-node", "--config-path", "/app/config.toml"]
