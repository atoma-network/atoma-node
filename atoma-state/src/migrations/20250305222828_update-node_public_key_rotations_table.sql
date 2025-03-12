-- Rename tdx_quote_bytes to remote_attestation_bytes and add new columns
ALTER TABLE node_public_key_rotations 
    RENAME COLUMN tdx_quote_bytes TO remote_attestation_bytes;

-- Add `device_type` column
ALTER TABLE node_public_key_rotations
    ADD COLUMN device_type INTEGER NOT NULL DEFAULT 0;

-- Drop the existing primary key constraint
ALTER TABLE node_public_key_rotations
    DROP CONSTRAINT node_public_key_rotations_pkey;

-- Add the new composite primary key
ALTER TABLE node_public_key_rotations
    ADD PRIMARY KEY (node_small_id, device_type);
