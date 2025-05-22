CREATE TABLE
  IF NOT EXISTS fiat_balance (
    user_address TEXT NOT NULL PRIMARY KEY,
    already_debited_amount BIGINT NOT NULL DEFAULT 0,
    overcharged_unsettled_amount BIGINT NOT NULL DEFAULT 0,
    num_requests BIGINT NOT NULL DEFAULT 0
  );
