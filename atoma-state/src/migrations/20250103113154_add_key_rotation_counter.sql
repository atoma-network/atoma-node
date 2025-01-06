ALTER TABLE node_public_key_rotations
ADD COLUMN key_rotation_counter BIGINT NOT NULL DEFAULT 0;