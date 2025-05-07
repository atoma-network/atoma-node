-- Create the partman schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS partman;

-- Check if the users table needs to be converted to a partitioned table
DO $ $ BEGIN -- Check if users table exists but is not partitioned
IF EXISTS (
	SELECT
		1
	FROM
		pg_tables
	WHERE
		tablename = 'users'
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
		AND c.relname = 'users'
) THEN -- Rename the existing table
EXECUTE 'ALTER TABLE users RENAME TO users_original';

-- Create a new partitioned table
EXECUTE '
        CREATE TABLE users (
            id SERIAL,
            -- Copy your schema from the original table
            -- You may need to adjust this according to your actual schema
            -- Use: \d users_original in psql to see the schema
            created_at TIMESTAMP NOT NULL
            -- Add all other columns here
        ) PARTITION BY RANGE (created_at)';

-- Create the partition maintenance function
PERFORM partman.create_parent(
	'public.users',
	'created_at',
	'time',
	'monthly',
	p_template_table = > NULL,
	p_retention = > '36 months',
	p_start_partition = > date_trunc('month', CURRENT_DATE - interval '3 years') :: text
);

-- Copy data from the original table to the partitioned table
EXECUTE 'INSERT INTO users SELECT * FROM users_original';

-- Optional: drop the original table after verifying data
-- EXECUTE 'DROP TABLE users_original';
RAISE NOTICE 'Converted users table to partitioned table';

ELSIF NOT EXISTS (
	SELECT
		1
	FROM
		pg_tables
	WHERE
		tablename = 'users'
		AND schemaname = 'public'
) THEN -- Create a new partitioned table if it doesn't exist
EXECUTE '
        CREATE TABLE users (
            id SERIAL,
            -- Define your schema
            created_at TIMESTAMP NOT NULL
            -- Add all other columns here
        ) PARTITION BY RANGE (created_at)';

-- Create the partition maintenance function
PERFORM partman.create_parent(
	'public.users',
	'created_at',
	'time',
	'monthly',
	p_template_table = > NULL,
	p_retention = > '36 months',
	p_start_partition = > date_trunc('month', CURRENT_DATE - interval '3 years') :: text
);

RAISE NOTICE 'Created new partitioned users table';

END IF;

END $ $;

-- Set up the maintenance job for pg_partman
SELECT
	partman.run_maintenance(p_analyze := true);