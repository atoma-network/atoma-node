# Builder stage
FROM --platform=$BUILDPLATFORM ubuntu:24.04 AS builder

# Add platform-specific arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TARGETARCH

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    curl \
    libssl-dev \
    libssl3 \
    ca-certificates \
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
    cargo build --release --bin atoma-node

# Final stage
FROM ubuntu:24.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
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
