#!/usr/bin/env bash
set -eu

mkdir -p coverage

# Clean up previous coverage data
rm -f coverage/*.profraw
rm -f coverage/coverage.profdata

# Install required tools
cargo install rustfilt
rustup component add llvm-tools-preview

# Setup PostgreSQL database for the test
echo "Setting up test database..."
psql postgres -c "CREATE ROLE atoma WITH LOGIN PASSWORD 'atoma'" || true
psql postgres -c "CREATE DATABASE atoma WITH OWNER atoma" || true
psql postgres -c "ALTER ROLE atoma CREATEDB" || true

# Set test environment variables
export DATABASE_URL="postgresql://atoma:atoma@localhost/atoma"
export RUST_BACKTRACE=1

# Run the tests with coverage instrumentation
CARGO_INCREMENTAL=0 RUSTFLAGS="-C instrument-coverage" cargo test --workspace

# Find the test executables
TEST_EXECUTABLES=$(RUSTFLAGS="-C instrument-coverage" cargo test --workspace --no-run --message-format=json \
    | jq -r "select(.profile.test == true) | .filenames[]" \
    | grep -v dSYM)

# Merge coverage files
llvm-profdata merge -sparse coverage/*.profraw -o coverage/coverage.profdata

# Generate the HTML report
llvm-cov show \
    --use-color \
    --ignore-filename-regex='/.cargo/registry' \
    --instr-profile=coverage/coverage.profdata \
    --show-instantiations \
    --show-line-counts-or-regions \
    --Xdemangler=rustfilt \
    --format=html \
    --output-dir=coverage/html \
    $TEST_EXECUTABLES

# Generate the coverage summary
llvm-cov report \
    --use-color \
    --ignore-filename-regex='/.cargo/registry' \
    --instr-profile=coverage/coverage.profdata \
    --summary-only \
    $TEST_EXECUTABLES

echo "Coverage report generated in coverage/html/index.html"