ALTER TABLE node_public_key_rotations (
    DROP COLUMN tdx_quote_bytes,
    ADD COLUMN tee_quote_bytes BYTEA NOT NULL,
    ADD COLUMN tee_provider BYTEA NOT NULL,
);