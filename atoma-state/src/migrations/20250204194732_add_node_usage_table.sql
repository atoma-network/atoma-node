CREATE TABLE IF NOT EXISTS nodes (
    node_small_id          BIGINT  PRIMARY KEY,
    node_id                TEXT    NOT NULL,
    node_sui_address        TEXT    NOT NULL
);
