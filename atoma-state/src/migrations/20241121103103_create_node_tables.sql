-- Create tasks table
CREATE TABLE IF NOT EXISTS tasks (
    task_small_id            BIGINT  PRIMARY KEY,
    task_id                  TEXT    UNIQUE NOT NULL,
    role                     BIGINT  NOT NULL,
    model_name               TEXT,
    is_deprecated            BOOLEAN NOT NULL,
    valid_until_epoch        BIGINT,
    deprecated_at_epoch      BIGINT,
    security_level           BIGINT  NOT NULL,
    minimum_reputation_score BIGINT
);

-- Create node_subscriptions table
CREATE TABLE IF NOT EXISTS node_subscriptions (
    task_small_id          BIGINT  NOT NULL,
    node_small_id          BIGINT  NOT NULL,
    price_per_compute_unit BIGINT  NOT NULL,
    max_num_compute_units  BIGINT  NOT NULL,
    valid                  BOOLEAN NOT NULL,
    PRIMARY KEY (task_small_id, node_small_id),
    FOREIGN KEY (task_small_id) 
        REFERENCES tasks (task_small_id)
);

    CREATE INDEX IF NOT EXISTS idx_node_subscriptions_task_small_id_node_small_id 
    ON node_subscriptions (task_small_id, node_small_id);

-- Create stacks table
CREATE TABLE IF NOT EXISTS stacks (
    stack_small_id         BIGINT  PRIMARY KEY,
    owner_address          TEXT    NOT NULL,
    stack_id               TEXT    UNIQUE NOT NULL,
    task_small_id          BIGINT  NOT NULL,
    selected_node_id       BIGINT  NOT NULL,
    num_compute_units      BIGINT  NOT NULL,
    price                  BIGINT  NOT NULL,
    already_computed_units BIGINT  NOT NULL,
    in_settle_period       BOOLEAN NOT NULL,
    total_hash             BYTEA   NOT NULL,
    num_total_messages     BIGINT  NOT NULL,
    FOREIGN KEY (selected_node_id, task_small_id) 
        REFERENCES node_subscriptions (node_small_id, task_small_id)
);

CREATE INDEX IF NOT EXISTS idx_stacks_owner_address 
    ON stacks (owner_address);

CREATE INDEX IF NOT EXISTS idx_stacks_stack_small_id 
    ON stacks (stack_small_id);

-- Create stack_attestation_disputes table
CREATE TABLE IF NOT EXISTS stack_attestation_disputes (
    stack_small_id          BIGINT NOT NULL,
    attestation_commitment  BYTEA  NOT NULL,
    attestation_node_id     BIGINT NOT NULL,
    original_node_id        BIGINT NOT NULL,
    original_commitment     BYTEA  NOT NULL,
    PRIMARY KEY (stack_small_id, attestation_node_id),
    FOREIGN KEY (stack_small_id) 
        REFERENCES stacks (stack_small_id)
);

-- Create stack_settlement_tickets table
CREATE TABLE IF NOT EXISTS stack_settlement_tickets (
    stack_small_id              BIGINT  PRIMARY KEY,
    selected_node_id            BIGINT  NOT NULL,
    num_claimed_compute_units   BIGINT  NOT NULL,
    requested_attestation_nodes TEXT    NOT NULL,
    committed_stack_proofs      BYTEA   NOT NULL,
    stack_merkle_leaves         BYTEA   NOT NULL,
    dispute_settled_at_epoch    BIGINT,
    already_attested_nodes      TEXT    NOT NULL,
    is_in_dispute               BOOLEAN NOT NULL,
    user_refund_amount          BIGINT  NOT NULL,
    is_claimed                  BOOLEAN NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_stack_settlement_tickets_stack_small_id 
    ON stack_settlement_tickets (stack_small_id);