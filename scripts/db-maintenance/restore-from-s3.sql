-- Function to restore data from S3
CREATE
OR REPLACE FUNCTION restore_from_s3(
	target_table TEXT,
	s3_bucket TEXT,
	s3_path TEXT
) RETURNS BIGINT AS $ $ DECLARE temp_table TEXT;

row_count BIGINT;

BEGIN -- Create a unique temp table name
temp_table := 'temp_import_' || to_char(now(), 'YYYYMMDD_HH24MISS');

-- Create temp table with same structure as target
EXECUTE 'CREATE TEMP TABLE ' || temp_table || ' (LIKE ' || quote_ident(target_table) || ')';

-- Log the operation start
INSERT INTO
	archive_log (operation, table_name, s3_path, status)
VALUES
	(
		'RESTORE_START',
		target_table,
		s3_path,
		'IN_PROGRESS'
	);

-- Import data from S3
PERFORM aws_s3.table_import_from_s3(
	temp_table,
	'',
	-- Use all columns
	'(FORMAT csv, HEADER true)',
	aws_commons.create_s3_uri(
		s3_bucket,
		s3_path,
		current_setting('aws_s3.aws_region', true)
	)
);

-- Count rows imported
EXECUTE 'SELECT COUNT(*) FROM ' || temp_table INTO row_count;

-- Insert into target table
EXECUTE 'INSERT INTO ' || quote_ident(target_table) || ' SELECT * FROM ' || temp_table;

-- Drop temp table
EXECUTE 'DROP TABLE ' || temp_table;

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
		'RESTORE_COMPLETE',
		target_table,
		s3_path,
		row_count,
		'SUCCESS'
	);

RETURN row_count;

EXCEPTION
WHEN OTHERS THEN -- Log error
INSERT INTO
	archive_log (
		operation,
		table_name,
		s3_path,
		status,
		error_message
	)
VALUES
	(
		'RESTORE_ERROR',
		target_table,
		s3_path,
		'ERROR',
		SQLERRM
	);

-- Clean up
EXECUTE 'DROP TABLE IF EXISTS ' || temp_table;

-- Rethrow the exception
RAISE;

END;

$ $ LANGUAGE plpgsql;