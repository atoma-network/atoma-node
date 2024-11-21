# Builder stage
FROM --platform=$BUILDPLATFORM rust:1.76-slim-bullseye AS builder

ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TRACE_LEVEL

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up cross-compilation
RUN case "$TARGETPLATFORM" in \
    "linux/amd64") echo "x86_64-unknown-linux-gnu" > /rust_target.txt ;; \
    "linux/arm64") echo "aarch64-unknown-linux-gnu" > /rust_target.txt ;; \
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
