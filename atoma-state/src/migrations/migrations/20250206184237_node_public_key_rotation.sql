-- Create tasks table
CREATE TABLE IF NOT EXISTS node_public_key_rotations (
    node_small_id            BIGINT  PRIMARY KEY,
    public_key_bytes         BYTEA    UNIQUE NOT NULL,
    tee_quote_bytes          BYTEA    NOT NULL,
    tee_provider             BYTEA    NOT NULL,
    epoch                    BIGINT  NOT NULL
);