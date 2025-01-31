#!/bin/bash

# Set the hooks directory
git config core.hooksPath .githooks

# Make the pre-commit hook executable
chmod +x .githooks/pre-commit

echo "Git hooks configured successfully."