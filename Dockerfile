# Builder stage
FROM rust:1.76 as builder
WORKDIR /usr/src/atoma-node
COPY . .

# Build the application
RUN cargo build --release

# Final stage
FROM alpine:3.19

# Install necessary dependencies
RUN apk add --no-cache \
    ca-certificates \
    sqlite \
    libgcc # Required for Rust binaries

WORKDIR /app
# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Copy the built binary from builder stage
COPY --from=builder /usr/src/atoma-node/target/release/atoma-node /usr/local/bin/atoma-node
# Copy configuration file
COPY --from=builder /usr/src/atoma-node/config.toml ./config.toml

CMD ["atoma-node", "--config-path", "/app/config.toml"]
