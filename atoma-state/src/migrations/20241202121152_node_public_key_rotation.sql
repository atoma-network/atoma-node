-- Create tasks table
CREATE TABLE IF NOT EXISTS node_public_key_rotations (
    node_small_id            BIGINT  PRIMARY KEY,
    public_key_bytes         TEXT    UNIQUE NOT NULL,
    tdx_quote_bytes          TEXT    NOT NULL,
    created_at_epoch         BIGINT  NOT NULL,
);
