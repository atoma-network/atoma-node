-- Add migration script here
ALTER TABLE stacks
ADD COLUMN is_claimed BOOLEAN NOT NULL DEFAULT FALSE;
ADD COLUMN user_refund_amount INTEGER;