#!/bin/bash

# Run cargo clippy with the specified lint settings
echo "Running cargo clippy..."
cargo clippy --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::style -W clippy::complexity -W clippy::perf -W clippy::suspicious -W clippy::correctness

# Check if clippy succeeded
if [ $? -ne 0 ]; then
    echo "Clippy found issues. Commit aborted."
    exit 1
fi

echo "Clippy checks passed. Proceeding with commit."
exit 0