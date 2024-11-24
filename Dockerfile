# Builder stage
FROM --platform=$BUILDPLATFORM rust:1.76-slim-bullseye AS builder

# Add platform-specific arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TARGETARCH

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies for the target architecture
RUN case "${TARGETARCH}" in \
    "amd64") apt-get update && apt-get install -y \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* ;; \
    "arm64") dpkg --add-architecture arm64 && \
    apt-get update && apt-get install -y \
    gcc-aarch64-linux-gnu \
    libssl-dev:arm64 \
    && rm -rf /var/lib/apt/lists/* ;; \
    *) exit 1 ;; \
    esac

WORKDIR /usr/src/atoma-node

COPY . .

# Set the correct target for cross-compilation
RUN case "${TARGETARCH}" in \
    "amd64") echo "x86_64-unknown-linux-gnu" > /rust_target ;; \
    "arm64") echo "aarch64-unknown-linux-gnu" > /rust_target ;; \
    *) exit 1 ;; \
    esac

# Add target support
RUN rustup target add $(cat /rust_target)

# Compile
RUN RUST_LOG=debug cargo build --release --bin atoma-node --target $(cat /rust_target)

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

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/usr/local/bin/atoma-node", "--config-path", "/app/config.toml"]
