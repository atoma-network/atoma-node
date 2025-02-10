# Builder stage
FROM --platform=$BUILDPLATFORM rust:1.83-slim-bullseye AS builder

# Add platform-specific arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TARGETARCH
ARG ENABLE_TDX
ARG ENABLE_SEV_SNP

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    curl \
    libssl-dev \
    libssl1.1 \
    && if [ "$ENABLE_TDX" = "true" ] || [ "$ENABLE_SEV_SNP" = "true" ]; then \
       apt-get install -y libtss2-dev; \
    fi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/atoma-node

COPY . .

# Compile
RUN if [ "$ENABLE_TDX" = "true" ] && [ "$ENABLE_SEV_SNP" = "false" ]; then \
        RUST_LOG=${TRACE_LEVEL} cargo build --release --bin atoma-node --features tdx; \
    elif [ "$ENABLE_SEV_SNP" = "true" ] && [ "$ENABLE_TDX" = "false" ]; then \
        RUST_LOG=${TRACE_LEVEL} cargo build --release --bin atoma-node --features sev-snp; \
    else \
        RUST_LOG=${TRACE_LEVEL} cargo build --release --bin atoma-node; \
    fi

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

RUN chmod +x /usr/local/bin/atoma-node

# Copy and set up entrypoint script
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/usr/local/bin/atoma-node", "--config-path", "/app/config.toml"]
