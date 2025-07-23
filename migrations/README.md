# Database Migrations

This folder contains database migration scripts for the LLM Data Cleaning Demo.

## Current Migrations

### add_metadata_to_usage_tracking.sql (2025-07-23)
**Problem**: The `usage_tracking` table was missing a `metadata` column that the application expected.

**Symptoms**: 
- Error: `column "metadata" of relation "usage_tracking" does not exist`
- Failed API call tracking

**Solution**: Adds `metadata JSONB` column to `usage_tracking` table.

## How to Apply Migrations

### Option 1: Manual SQL Execution
```sql
-- Connect to your database and run:
\i migrations/add_metadata_to_usage_tracking.sql
```

### Option 2: Python Script
```bash
# Set your database URL
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"

# Run the migration script
python3 fix_database_schema.py
```

### Option 3: Direct psql
```bash
psql "$DATABASE_URL" -f migrations/add_metadata_to_usage_tracking.sql
```

## For Neon Database (Streamlit Cloud Demo)

1. **Connect to your Neon database**:
   ```bash
   psql "postgresql://user:pass@host.neon.tech:5432/dbname"
   ```

2. **Apply the migration**:
   ```sql
   \i migrations/add_metadata_to_usage_tracking.sql
   ```

3. **Verify the fix**:
   ```sql
   \d usage_tracking
   ```
   You should see the `metadata` column listed.

## Migration Status

- ✅ `add_metadata_to_usage_tracking.sql` - Adds missing metadata column
- 🔄 Ready to apply to production Neon database

## Notes

- All migrations use `IF NOT EXISTS` clauses to prevent errors on re-runs
- Metadata column is added as JSONB for flexible usage tracking data
- Index is created for efficient metadata queries
- Existing rows are updated with empty JSON objects