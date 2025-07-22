# Neon PostgreSQL Database Setup

## Step 1: Create Neon Account

1. **Go to Neon**: https://neon.tech
2. **Sign up** for free account (using GitHub is recommended)
3. **Verify email** if required

## Step 2: Create Project

1. **Click "Create Project"**
2. **Project settings**:
   - Project name: `llm-data-cleaning-demo`
   - Database name: `data_cleaning_demo`
   - Region: Choose closest to your users
   - PostgreSQL version: 15 (recommended)

3. **Create project**

## Step 3: Get Connection Details

After project creation, you'll see:
- **Connection string**: `postgresql://user:password@ep-xxx.aws.neon.tech/data_cleaning_demo`
- **Host**: `ep-xxx.aws.neon.tech`
- **Database**: `data_cleaning_demo`
- **Username**: `user`
- **Password**: `password`

**SAVE THESE DETAILS** - you'll need them for Streamlit Cloud secrets.

## Step 4: Initialize Database Schema

1. **Open Neon SQL Editor** (in the dashboard)
2. **Copy the entire contents** of `auth/schema.sql`
3. **Paste and execute** in the SQL Editor
4. **Verify** tables are created (you should see ~10 tables)

## Step 5: Create Demo Data

1. **In Neon SQL Editor**, run this command first:
```sql
INSERT INTO organizations (name, description, website) 
VALUES (
    'Demo Laboratory', 
    'Experience the power of AI-driven data cleaning', 
    'https://llm-data-cleaning.streamlit.app'
);
```

2. **Note the organization ID** from the result (it will look like: `12345678-1234-1234-1234-123456789abc`)

3. **Run this command** (replace `[ORG_ID]` with the actual ID from step 2):
```sql
INSERT INTO users (email, password_hash, full_name, organization_id, is_admin)
VALUES (
    'demo@example.com', 
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiGSBGzKyKtK', 
    'Demo User', 
    '[ORG_ID]', -- Replace with actual organization ID
    true
);
```

4. **Get the user ID** from the result

5. **Set usage limits** (replace `[ORG_ID]` with actual ID):
```sql
INSERT INTO usage_limits (organization_id, experiments_monthly, storage_mb, api_calls_daily)
VALUES (
    '[ORG_ID]', -- Replace with actual organization ID
    1000,       -- Generous limits for demo
    10240,      -- 10GB storage
    10000       -- 10k API calls per day
);
```

6. **Create sample experiments** (replace `[USER_ID]` and `[ORG_ID]` with actual IDs):
```sql
INSERT INTO experiments (user_id, organization_id, name, description, status, completed_at)
VALUES 
    ('[USER_ID]', '[ORG_ID]', 'Plate Data Cleanup Demo', 'Cleaned 96-well plate with Unicode characters', 'completed', NOW()),
    ('[USER_ID]', '[ORG_ID]', 'Protocol Standardization', 'Unified multiple protocol formats', 'completed', NOW()),
    ('[USER_ID]', '[ORG_ID]', 'Time-Series Alignment', 'Synchronized equipment monitoring logs', 'completed', NOW());
```

## Step 6: Test Connection

Run this test query to verify everything works:
```sql
SELECT 
    o.name as organization,
    u.email as user_email,
    COUNT(e.id) as experiment_count
FROM organizations o
JOIN users u ON u.organization_id = o.id
LEFT JOIN experiments e ON e.organization_id = o.id
GROUP BY o.name, u.email;
```

You should see:
- Organization: "Demo Laboratory"
- User: "demo@example.com" 
- Experiment count: 3

## Step 7: Save Connection Info

**IMPORTANT**: Save these details for Streamlit Cloud:
- **DATABASE_URL**: `postgresql://user:password@ep-xxx.aws.neon.tech/data_cleaning_demo`
- **DB_HOST**: `ep-xxx.aws.neon.tech`
- **DB_PORT**: `5432`
- **DB_NAME**: `data_cleaning_demo`
- **DB_USER**: `user`
- **DB_PASSWORD**: `password`

## Demo Login Credentials

Once deployed, users can login with:
- **Email**: `demo@example.com`
- **Password**: `demo123`

## Next Steps
Once database is set up, proceed with Streamlit Cloud deployment using STREAMLIT_CLOUD_SETUP.md