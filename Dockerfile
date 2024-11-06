# Builder stage
FROM rust:1.76-alpine as builder

# Install build dependencies
RUN apk add --no-cache \
    build-base \
    pkgconfig \
    openssl-dev \
    curl \
    musl-dev \
    perl \
    make \
    linux-headers

WORKDIR /usr/src/atoma-node
COPY . .

# Set environment variables for SSL
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_DIR=/etc/ssl/certs
ENV OPENSSL_LIB_DIR=/usr/lib
ENV OPENSSL_INCLUDE_DIR=/usr/include

# Build the application with static linking
RUN RUSTFLAGS='-C target-feature=+crt-static' cargo build --release

# Final stage
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    sqlite \
    libgcc \
    openssl

WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Copy the built binary from builder stage
COPY --from=builder /usr/src/atoma-node/target/release/atoma-node /usr/local/bin/atoma-node

# Copy configuration file
COPY --from=builder /usr/src/atoma-node/config.toml ./config.toml

# Set executable permissions explicitly
RUN chmod +x /usr/local/bin/atoma-node

# Verify the binary exists and is executable
RUN ls -l /usr/local/bin/atoma-node && \
    ldd /usr/local/bin/atoma-node || true

# Use full path in CMD
CMD ["/usr/local/bin/atoma-node", "--config-path", "/app/config.toml"]
