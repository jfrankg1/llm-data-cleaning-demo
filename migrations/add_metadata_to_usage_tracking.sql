-- Migration: Add metadata column to usage_tracking table
-- Date: 2025-07-23
-- Description: Add missing metadata JSONB column to usage_tracking table for enhanced usage tracking

-- Add metadata column to usage_tracking table
ALTER TABLE usage_tracking 
ADD COLUMN IF NOT EXISTS metadata JSONB;

-- Add index for metadata queries (optional but good for performance)
CREATE INDEX IF NOT EXISTS idx_usage_tracking_metadata ON usage_tracking USING GIN(metadata);

-- Update any existing rows to have empty metadata (NULL is fine, but {} is cleaner)
UPDATE usage_tracking 
SET metadata = '{}'::jsonb 
WHERE metadata IS NULL;