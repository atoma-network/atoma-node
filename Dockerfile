# Builder stage
FROM --platform=$BUILDPLATFORM ubuntu:24.04 AS builder

# Add platform-specific arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TARGETARCH
ARG ENABLE_CC

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    curl \
    libssl-dev \
    libssl3 \
    ca-certificates \
    && if [ "$ENABLE_CC" = "true" ]; then \
    apt-get install -y libtss2-dev; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Increase system limits for the build
RUN echo "* soft nofile 65535" >> /etc/security/limits.conf && \
    echo "* hard nofile 65535" >> /etc/security/limits.conf && \
    echo "* soft nproc 65535" >> /etc/security/limits.conf && \
    echo "* hard nproc 65535" >> /etc/security/limits.conf

# Install Rust 1.84.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.84.0 \
    && . "$HOME/.cargo/env"

# Add cargo to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /usr/src/atoma-node

COPY . .

# Compile with increased limits
RUN ulimit -n 65535 && \
    if [ "$ENABLE_CC" = "true" ]; then \
    RUST_LOG=${TRACE_LEVEL} cargo build --release --bin atoma-node --features tdx; \
    else \
    RUST_LOG=${TRACE_LEVEL} cargo build --release --bin atoma-node; \
    fi

# Final stage
FROM ubuntu:24.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libsqlite3-0 \
    && if [ "$ENABLE_CC" = "true" ]; then \
    apt-get install -y \
    libtss2-esys-3.0.2-0t64 \
    libtss2-mu-4.0.1-0t64 \
    libtss2-rc0t64 \
    libtss2-sys1t64; \
    fi \
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
