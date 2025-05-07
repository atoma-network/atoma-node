-- Create the partman extension if it doesn't exist
CREATE EXTENSION IF NOT EXISTS pg_partman;

-- Create the partman schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS partman;

-- Function to set up partitioning for a table
CREATE
OR REPLACE FUNCTION setup_table_partitioning(
	table_name TEXT,
	partition_column TEXT,
	retention_months INT
) RETURNS void AS $ $ DECLARE original_table TEXT;

partition_start TEXT;

BEGIN -- Check if table exists but is not partitioned
IF EXISTS (
	SELECT
		1
	FROM
		pg_tables
	WHERE
		tablename = table_name
		AND schemaname = 'public'
)
AND NOT EXISTS (
	SELECT
		1
	FROM
		pg_partitioned_table pt
		JOIN pg_class c ON pt.partrelid = c.oid
		JOIN pg_namespace n ON c.relnamespace = n.oid
	WHERE
		n.nspname = 'public'
		AND c.relname = table_name
) THEN -- Rename the existing table
original_table := table_name || '_original';

EXECUTE 'ALTER TABLE ' || quote_ident(table_name) || ' RENAME TO ' || quote_ident(original_table);

-- Create a new partitioned table with the same structure
EXECUTE 'CREATE TABLE ' || quote_ident(table_name) || ' (LIKE ' || quote_ident(original_table) || ') PARTITION BY RANGE (' || quote_ident(partition_column) || ')';

-- Create the partition maintenance function
partition_start := date_trunc(
	'month',
	CURRENT_DATE - (retention_months || ' months') :: interval
) :: text;

PERFORM partman.create_parent(
	'public.' || table_name,
	partition_column,
	'time',
	'monthly',
	p_template_table = > NULL,
	p_retention = > retention_months || ' months',
	p_retention_keep_table = > true,
	p_start_partition = > partition_start,
	p_premake = > 3,
	p_automatic_maintenance = > 'on'
);

-- Copy data from the original table to the partitioned table
EXECUTE 'INSERT INTO ' || quote_ident(table_name) || ' SELECT * FROM ' || quote_ident(original_table);

RAISE NOTICE 'Converted % table to partitioned table',
table_name;

END IF;

END;

$ $ LANGUAGE plpgsql;

-- Set up partitioning for all relevant tables
DO $ $ BEGIN -- Set up partitioning for tasks table
PERFORM setup_table_partitioning('tasks', 'deprecated_at_epoch', 1);

-- Set up partitioning for stacks table
PERFORM setup_table_partitioning('stacks', 'created_at', 1);

-- Set up partitioning for stack_settlement_tickets table
PERFORM setup_table_partitioning('stack_settlement_tickets', 'created_at', 1);

-- Set up partitioning for stack_attestation_disputes table
PERFORM setup_table_partitioning('stack_attestation_disputes', 'created_at', 1);

END $ $;

-- Set up the maintenance job for pg_partman
SELECT
	partman.run_maintenance(p_analyze := true);

-- Configure automatic maintenance for all partitioned tables
UPDATE
	partman.part_config
SET
	automatic_maintenance = 'on'
WHERE
	parent_table IN (
		'public.tasks',
		'public.stacks',
		'public.stack_settlement_tickets',
		'public.stack_attestation_disputes'
	);

-- Create a maintenance function to be called by cron
CREATE
OR REPLACE FUNCTION partman.run_maintenance_proc() RETURNS void LANGUAGE plpgsql AS $ $ BEGIN PERFORM partman.run_maintenance(p_analyze := true);

END;

$ $;