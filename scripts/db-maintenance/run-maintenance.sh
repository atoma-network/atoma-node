#!/bin/bash
set -e

echo "Starting database maintenance operations..."

# Install required extensions
psql -v ON_ERROR_STOP=1 -d "$POSTGRES_DB" -c "CREATE EXTENSION IF NOT EXISTS pg_partman;"
psql -v ON_ERROR_STOP=1 -d "$POSTGRES_DB" -c "CREATE EXTENSION IF NOT EXISTS aws_s3;"

# Initialize the partitioning schema
psql -v ON_ERROR_STOP=1 -d "$POSTGRES_DB" -f /scripts/init-partitioning.sql

# Run the archival process
psql -v ON_ERROR_STOP=1 -d "$POSTGRES_DB" -f /scripts/archive-partitions.sql

echo "Database maintenance completed successfully."