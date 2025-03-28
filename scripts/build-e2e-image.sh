#!/bin/bash

# This uses the same S3 cache as all test-plans images. Because we use `cargo-chef` in the Dockerfile, we have a layer available with all dependencies built.

CACHE_TO=""

# If we have credentials, write to cache
if [[ -n "${AWS_SECRET_ACCESS_KEY}" ]]; then
  CACHE_TO="--cache-to   type=s3,mode=max,bucket=${AWS_BUCKET_NAME},region=${AWS_REGION},prefix=buildCache,name=${FLAVOUR}-atoma-node-head"
fi

docker buildx build \
  --load \
  $CACHE_TO \
  --cache-from type=s3,mode=max,bucket=${AWS_BUCKET_NAME},region=${AWS_REGION},prefix=buildCache,name=${FLAVOUR}-atoma-node-head \
  -t ${FLAVOUR}-atoma-node-head \
  . \
  -f e2e-tests/Dockerfile.${FLAVOUR}