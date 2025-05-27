CREATE TABLE
  IF NOT EXISTS usage_per_model (
    user_address TEXT NOT NULL,
    model TEXT NOT NULL,
    input_amount BIGINT NOT NULL DEFAULT 0,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_amount BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (user_address, model)
  );
