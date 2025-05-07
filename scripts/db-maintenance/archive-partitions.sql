-- Create a logging table for archive operations
CREATE TABLE IF NOT EXISTS archive_log (
	id SERIAL PRIMARY KEY,
	operation VARCHAR(50),
	table_name TEXT,
	s3_path TEXT,
	rows_affected BIGINT,
	status VARCHAR(50),
	error_message TEXT,
	executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Configure AWS credentials
-- Note: In production, use IAM roles instead of embedding credentials
DO $ $ BEGIN EXECUTE 'ALTER DATABASE ' || current_database() || ' SET aws_s3.aws_access_key_id TO ''' || current_setting('aws_access_key_id', true) || '''';

EXECUTE 'ALTER DATABASE ' || current_database() || ' SET aws_s3.aws_secret_access_key TO ''' || current_setting('aws_secret_key', true) || '''';

EXECUTE 'ALTER DATABASE ' || current_database() || ' SET aws_s3.aws_region TO ''' || coalesce(current_setting('aws_region', true), 'us-east-1') || '''';

END $ $;

-- Function to archive a partition to S3
CREATE
OR REPLACE FUNCTION archive_partition_to_s3(
	partition_table_name TEXT,
	s3_bucket TEXT,
	s3_prefix TEXT
) RETURNS BIGINT AS $ $ DECLARE s3_path TEXT;

rows_affected BIGINT;

current_time TEXT;

BEGIN -- Generate a timestamp for the archive
current_time := to_char(now(), 'YYYY_MM_DD_HH24_MI_SS');

-- Set the complete S3 path
s3_path := s3_prefix || '/' || partition_table_name || '/' || current_time || '.csv';

-- Count rows in the partition
EXECUTE 'SELECT COUNT(*) FROM ' || quote_ident(partition_table_name) INTO rows_affected;

-- Log the operation start
INSERT INTO
	archive_log (
		operation,
		table_name,
		s3_path,
		rows_affected,
		status
	)
VALUES
	(
		'ARCHIVE_START',
		partition_table_name,
		s3_path,
		rows_affected,
		'IN_PROGRESS'
	);

-- Export the data to S3
PERFORM aws_s3.table_export_to_s3(
	'SELECT * FROM ' || quote_ident(partition_table_name),
	aws_commons.create_s3_uri(
		s3_bucket,
		s3_path,
		current_setting('aws_s3.aws_region', true)
	),
	options := 'FORMAT CSV, HEADER true'
);

-- Log successful completion
INSERT INTO
	archive_log (
		operation,
		table_name,
		s3_path,
		rows_affected,
		status
	)
VALUES
	(
		'ARCHIVE_COMPLETE',
		partition_table_name,
		s3_path,
		rows_affected,
		'SUCCESS'
	);

-- Create a local backup
COPY (
	SELECT
		*
	FROM
		partition_table_name
) TO '/s3-archives/' || partition_table_name || '_' || current_time || '.csv' WITH (FORMAT CSV, HEADER true);

RETURN rows_affected;

EXCEPTION
WHEN OTHERS THEN -- Log error
INSERT INTO
	archive_log (
		operation,
		table_name,
		s3_path,
		rows_affected,
		status,
		error_message
	)
VALUES
	(
		'ARCHIVE_ERROR',
		partition_table_name,
		s3_path,
		rows_affected,
		'ERROR',
		SQLERRM
	);

RAISE;

END;

$ $ LANGUAGE plpgsql;

-- Function to find and archive old data
CREATE
OR REPLACE FUNCTION archive_old_data(
	months_to_keep INT,
	s3_bucket TEXT,
	s3_prefix TEXT
) RETURNS TABLE (table_name TEXT, rows_archived BIGINT) AS $ $ DECLARE archive_date BIGINT;

rec RECORD;

BEGIN -- Calculate the cutoff epoch (assuming epoch is in milliseconds)
archive_date := extract(
	epoch
	from
		(
			CURRENT_DATE - (months_to_keep || ' months') :: interval
		)
) * 1000;

-- Archive old tasks
FOR rec IN
SELECT
	'tasks' as table_name,
	task_small_id
FROM
	tasks
WHERE
	deprecated_at_epoch < archive_date
	AND is_deprecated = true LOOP table_name := rec.table_name;

rows_archived := archive_partition_to_s3(
	'tasks_' || rec.task_small_id,
	s3_bucket,
	s3_prefix
);

RETURN NEXT;

END LOOP;

-- Archive old stacks
FOR rec IN
SELECT
	'stacks' as table_name,
	stack_small_id
FROM
	stacks s
	JOIN tasks t ON s.task_small_id = t.task_small_id
WHERE
	t.deprecated_at_epoch < archive_date
	AND t.is_deprecated = true LOOP table_name := rec.table_name;

rows_archived := archive_partition_to_s3(
	'stacks_' || rec.stack_small_id,
	s3_bucket,
	s3_prefix
);

RETURN NEXT;

END LOOP;

-- Archive related settlement tickets
FOR rec IN
SELECT
	'stack_settlement_tickets' as table_name,
	sst.stack_small_id
FROM
	stack_settlement_tickets sst
	JOIN stacks s ON sst.stack_small_id = s.stack_small_id
	JOIN tasks t ON s.task_small_id = t.task_small_id
WHERE
	t.deprecated_at_epoch < archive_date
	AND t.is_deprecated = true LOOP table_name := rec.table_name;

rows_archived := archive_partition_to_s3(
	'stack_settlement_tickets_' || rec.stack_small_id,
	s3_bucket,
	s3_prefix
);

RETURN NEXT;

END LOOP;

-- Archive related disputes
FOR rec IN
SELECT
	'stack_attestation_disputes' as table_name,
	sad.stack_small_id
FROM
	stack_attestation_disputes sad
	JOIN stacks s ON sad.stack_small_id = s.stack_small_id
	JOIN tasks t ON s.task_small_id = t.task_small_id
WHERE
	t.deprecated_at_epoch < archive_date
	AND t.is_deprecated = true LOOP table_name := rec.table_name;

rows_archived := archive_partition_to_s3(
	'stack_attestation_disputes_' || rec.stack_small_id,
	s3_bucket,
	s3_prefix
);

RETURN NEXT;

END LOOP;

RETURN;

END;

$ $ LANGUAGE plpgsql;

-- Run the archive function
SELECT
	*
FROM
	archive_old_data(
		1,
		-- Keep 1 month of data in PostgreSQL
		'your-s3-bucket-name',
		-- Replace with your S3 bucket
		'database-archives' -- S3 prefix
	);