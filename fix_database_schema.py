#!/usr/bin/env python3
"""
Database Schema Fix Script
Adds missing metadata column to usage_tracking table

This script should be run on the demo database to fix the schema mismatch.
"""

import os
import sys
import psycopg2
from pathlib import Path

def get_database_url():
    """Get database URL from environment"""
    return os.getenv('DATABASE_URL')

def run_migration():
    """Run the database migration"""
    database_url = get_database_url()
    
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        print("Please set DATABASE_URL to your PostgreSQL connection string")
        sys.exit(1)
    
    print("🔧 Connecting to database...")
    
    try:
        # Connect to database
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        print("✅ Connected to database")
        
        # Check if metadata column already exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'usage_tracking' 
            AND column_name = 'metadata'
        """)
        
        if cursor.fetchone():
            print("ℹ️  Metadata column already exists, skipping migration")
            return
        
        print("🚀 Adding metadata column to usage_tracking table...")
        
        # Add metadata column
        cursor.execute("""
            ALTER TABLE usage_tracking 
            ADD COLUMN metadata JSONB
        """)
        
        print("📊 Creating index for metadata column...")
        
        # Add index for metadata
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_tracking_metadata 
            ON usage_tracking USING GIN(metadata)
        """)
        
        print("🔄 Updating existing rows with empty metadata...")
        
        # Update existing rows
        cursor.execute("""
            UPDATE usage_tracking 
            SET metadata = '{}'::jsonb 
            WHERE metadata IS NULL
        """)
        
        # Commit changes
        conn.commit()
        
        print("✅ Database migration completed successfully!")
        print("The usage_tracking table now has a metadata column.")
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()
            print("🔐 Database connection closed")

if __name__ == "__main__":
    print("🏥 Database Schema Fix Script")
    print("=" * 40)
    run_migration()