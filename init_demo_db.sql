-- LLM Data Cleaning System - Demo Database Initialization
-- Run this after schema.sql to set up demo data

-- Create demo organization
INSERT INTO organizations (name, description, website) 
VALUES (
    'Demo Laboratory', 
    'Experience the power of AI-driven data cleaning', 
    'https://llm-data-cleaning.streamlit.app'
);

-- Get the organization ID (you'll see this in the output)
-- Save this ID for the next step

-- Create demo user with bcrypt hashed password 'demo123'
-- Replace [ORG_ID] with the actual organization ID from above
/*
INSERT INTO users (email, password_hash, full_name, organization_id, is_admin)
VALUES (
    'demo@example.com', 
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiGSBGzKyKtK', 
    'Demo User', 
    '[ORG_ID]', -- Replace with actual organization ID
    true
);
*/

-- Set usage limits for demo organization
-- Replace [ORG_ID] with the actual organization ID
/*
INSERT INTO usage_limits (organization_id, experiments_monthly, storage_mb, api_calls_daily)
VALUES (
    '[ORG_ID]', -- Replace with actual organization ID
    1000,       -- Generous limits for demo
    10240,      -- 10GB storage
    10000       -- 10k API calls per day
);
*/

-- Create some sample experiments for demo
-- Replace [ORG_ID] and [USER_ID] with actual IDs
/*
INSERT INTO experiments (user_id, organization_id, name, description, status, completed_at)
VALUES 
    ('[USER_ID]', '[ORG_ID]', 'Plate Data Cleanup Demo', 'Cleaned 96-well plate with Unicode characters', 'completed', NOW()),
    ('[USER_ID]', '[ORG_ID]', 'Protocol Standardization', 'Unified multiple protocol formats', 'completed', NOW()),
    ('[USER_ID]', '[ORG_ID]', 'Time-Series Alignment', 'Synchronized equipment monitoring logs', 'completed', NOW());
*/

-- Instructions:
-- 1. Run the first INSERT to create the organization
-- 2. Note the organization ID from the result
-- 3. Uncomment and update the remaining INSERTs with the actual IDs
-- 4. Run the remaining INSERTs