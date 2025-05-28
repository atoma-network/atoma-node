CREATE TABLE
  IF NOT EXISTS usage_per_day (
    user_id BIGINT NOT NULL,
    user_address TEXT NOT NULL,
    date DATE DEFAULT CURRENT_DATE NOT NULL,
    model TEXT NOT NULL,
    input_amount BIGINT NOT NULL DEFAULT 0,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_amount BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    UNIQUE (user_id, user_address, date, model)
  );
