# Builder stage
FROM rust:1.76 as builder

WORKDIR /usr/src/atoma-node
COPY . .

# Build the application
RUN cargo build --release

# Final stage
FROM ubuntu:22.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Copy the built binary from builder stage
COPY --from=builder /usr/src/atoma-node/target/release/atoma-node /usr/local/bin/atoma-node
# Copy configuration file
COPY --from=builder /usr/src/atoma-node/config.toml ./config.toml

CMD ["atoma-node", "--config-path", "/app/config.toml"]
