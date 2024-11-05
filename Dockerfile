# Builder stage
FROM rust:1.76-alpine as builder

# Add build argument for binary selection
ARG BINARY
RUN test -n "$BINARY" || (echo "BINARY is not set" && false)

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

WORKDIR /usr/src/app
COPY . .

# Set environment variables for SSL
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_DIR=/etc/ssl/certs
ENV OPENSSL_LIB_DIR=/usr/lib
ENV OPENSSL_INCLUDE_DIR=/usr/include

# Build the application
RUN cargo build --release

# Final stage
FROM alpine:3.19

# Add build argument for binary selection
ARG BINARY
RUN test -n "$BINARY" || (echo "BINARY is not set" && false)

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    sqlite \
    libgcc \
    openssl

WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Copy the built binary from builder stage using the BINARY argument
COPY --from=builder /usr/src/app/target/release/${BINARY} /usr/local/bin/${BINARY}

# Copy configuration file
COPY --from=builder /usr/src/app/config.toml ./config.toml

# Set executable permissions explicitly
RUN chmod +x /usr/local/bin/${BINARY}

# Use the BINARY argument in the CMD instruction
CMD ["sh", "-c", "exec /usr/local/bin/${BINARY} --config-path /app/config.toml"]