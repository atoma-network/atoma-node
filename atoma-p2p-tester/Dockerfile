# Builder stage
FROM --platform=$BUILDPLATFORM ubuntu:24.04 AS builder

# Add platform-specific arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TARGETARCH

ARG PROFILE

# Install build dependencies
RUN apt-get update && apt-get install -y \
	build-essential \
	pkg-config \
	curl \
	libssl-dev \
	libssl3 \
	ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

# Install Rust 1.84.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.84.0 \
	&& . "$HOME/.cargo/env"

# Add cargo to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory to the workspace root
WORKDIR /usr/src/atoma-node

# Note: This Dockerfile should be built from the workspace root
# with: docker build -f atoma-p2p-tester/Dockerfile -t atoma-p2p-tester .
COPY . .

# Compile atoma-p2p-tester
RUN cargo build --release --package atoma-p2p-tester

# Final stage
FROM ubuntu:24.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
	ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs

# Copy the built binary from builder stage
COPY --from=builder /usr/src/atoma-node/target/release/atoma-p2p-tester /usr/local/bin/atoma-p2p-tester

# Set executable permissions explicitly
RUN chmod +x /usr/local/bin/atoma-p2p-tester

# Copy entrypoint script from the workspace root
COPY --from=builder /usr/src/atoma-node/entrypoint.sh /usr/local/bin/

# Set executable permissions
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["/usr/local/bin/atoma-p2p-tester", "--config-path", "/app/config.toml"]
