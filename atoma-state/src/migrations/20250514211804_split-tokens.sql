BEGIN;

ALTER TABLE fiat_balance RENAME COLUMN already_debited_amount TO already_debited_completions_amount;
ALTER TABLE fiat_balance RENAME COLUMN overcharged_unsettled_amount TO overcharged_unsettled_completions_amount;

ALTER TABLE fiat_balance
ADD COLUMN already_debited_input_amount BIGINT NOT NULL DEFAULT 0,
ADD COLUMN overcharged_unsettled_input_amount BIGINT NOT NULL DEFAULT 0;

COMMIT;
