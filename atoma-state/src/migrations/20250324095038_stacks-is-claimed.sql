-- Add columns to the stacks table, for tracking the user refund amount
-- and if the stack has been claimed for confidential compute requests/tasks.
ALTER TABLE stacks
    ADD COLUMN is_claimed BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN user_refund_amount INTEGER,
    ADD COLUMN is_confidential BOOLEAN NOT NULL DEFAULT FALSE;