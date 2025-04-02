-- Potentially optimize the UPDATE check in get_available_stack_with_compute_units
CREATE INDEX IF NOT EXISTS idx_stacks_availability_check 
    ON stacks (stack_small_id, owner_address, in_settle_period, is_claimed, is_locked_for_claim);

DROP INDEX IF EXISTS idx_node_subscriptions_task_small_id_node_small_id;
DROP INDEX IF EXISTS idx_stacks_stack_small_id;