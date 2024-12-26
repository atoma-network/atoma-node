-- Remove foreign key from node_subscriptions
ALTER TABLE node_subscriptions
DROP CONSTRAINT node_subscriptions_task_small_id_fkey;

-- Remove foreign key from stacks
ALTER TABLE stacks
DROP CONSTRAINT stacks_selected_node_id_task_small_id_fkey;

-- Remove foreign key from stack_attestation_disputes
ALTER TABLE stack_attestation_disputes
DROP CONSTRAINT stack_attestation_disputes_stack_small_id_fkey;