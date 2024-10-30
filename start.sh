#!/bin/bash
PROFILES=""

# Check environment variables
source .env

# Add enabled profiles
if [ "${CHAT_COMPLETIONS_ON}" = "true" ]; then
    if [ "${CHAT_COMPLETIONS_BACKEND}" = "vllm" ]; then
        PROFILES="$PROFILES --profile chat_completions_vllm"
    fi
fi
if [ "${EMBEDDINGS_ON}" = "true" ]; then
    if [ "${EMBEDDINGS_BACKEND}" = "tei" ]; then
        PROFILES="$PROFILES --profile embeddings_tei"
    fi
fi
if [ "${IMAGE_GENERATIONS_ON}" = "true" ]; then
    if [ "${IMAGE_GENERATIONS_BACKEND}" = "mistralrs" ]; then
        PROFILES="$PROFILES --profile image_generations_mistral"
    fi
fi

# Start docker compose with the selected profiles
if [ -z "$PROFILES" ]; then
    echo "No services enabled in .env file"
    exit 1
else
    echo "Starting services with profiles:$PROFILES"
    docker compose $PROFILES up
fi
