# Builder stage
FROM rust:1.76-slim-bullseye as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/atoma-node

COPY . .

# Build the application
RUN cargo build --release --bin atoma-node

# Final stage
FROM debian:bullseye-slim

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

# Set executable permissions explicitly
RUN chmod +x /usr/local/bin/atoma-node

# Use full path in CMD
CMD ["/usr/local/bin/atoma-node", "--config-path", "/app/config.toml"]
