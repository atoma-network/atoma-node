-- Add user_id to `fiat_balance` table so that the node can track payments by user, as well.
ALTER TABLE fiat_balances ADD COLUMN user_id BIGINT NOT NULL DEFAULT -1;

-- Drop the existing primary key constraint
ALTER TABLE fiat_balances DROP CONSTRAINT fiat_balance_pkey;

-- Add a new composite primary key
ALTER TABLE fiat_balances ADD PRIMARY KEY (user_id, user_address);
