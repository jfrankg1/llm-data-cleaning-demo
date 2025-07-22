#!/usr/bin/env python3
"""
LLM Data Cleaning System - Deployment Readiness Verification
Checks that all required files are present and properly configured
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists and report status"""
    if os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description}: {dirpath} - MISSING")
        return False

def check_file_content(filepath, search_text, description):
    """Check if a file contains specific text"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if search_text in content:
                print(f"✅ {description}")
                return True
            else:
                print(f"❌ {description} - TEXT NOT FOUND")
                return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run deployment readiness checks"""
    print("🚀 LLM Data Cleaning System - Deployment Readiness Check")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Core application files
    print("\n📱 Core Application Files:")
    all_checks_passed &= check_file_exists("streamlit_app.py", "Main application")
    all_checks_passed &= check_file_exists("streamlit_app_admin.py", "Admin application")
    all_checks_passed &= check_file_exists("requirements.txt", "Dependencies")
    
    # Configuration files
    print("\n⚙️ Configuration Files:")
    all_checks_passed &= check_file_exists(".streamlit/config.toml", "Streamlit config")
    all_checks_passed &= check_file_exists(".env.example", "Environment template")
    all_checks_passed &= check_file_exists(".gitignore", "Git ignore rules")
    
    # Core directories
    print("\n📁 Core Directories:")
    all_checks_passed &= check_directory_exists("auth", "Authentication system")
    all_checks_passed &= check_directory_exists("services", "Business logic services")
    all_checks_passed &= check_directory_exists("src", "Processing engine")
    all_checks_passed &= check_directory_exists("ui", "User interface components")
    all_checks_passed &= check_directory_exists("demo_data", "Demo data files")
    
    # Demo data files
    print("\n🧪 Demo Data Files:")
    all_checks_passed &= check_file_exists("demo_data/raw/data set 6 - Unicode.csv", "Unicode demo file")
    all_checks_passed &= check_file_exists("demo_data/raw/data set 9 - Semicolon.csv", "Semicolon demo file")
    all_checks_passed &= check_file_exists("demo_data/protocols/Experimental Protocol Notes.txt", "Protocol demo file")
    
    # Documentation
    print("\n📚 Documentation:")
    all_checks_passed &= check_file_exists("README.md", "Main README")
    all_checks_passed &= check_file_exists("DEPLOYMENT_GUIDE.md", "Deployment guide")
    all_checks_passed &= check_file_exists("DEMO_SCENARIOS.md", "Demo scenarios")
    all_checks_passed &= check_file_exists("DEPLOYMENT_CHECKLIST.md", "Deployment checklist")
    
    # Database files
    print("\n🗄️ Database Files:")
    all_checks_passed &= check_file_exists("auth/schema.sql", "Database schema")
    all_checks_passed &= check_file_exists("init_demo_db.sql", "Demo data setup")
    
    # Branding checks
    print("\n🎨 Branding Verification:")
    all_checks_passed &= check_file_content("streamlit_app.py", "LLM Data Cleaning System", "Main app title")
    all_checks_passed &= check_file_content("README.md", "LLM Data Cleaning System", "README title")
    all_checks_passed &= check_file_content("streamlit_app.py", "Transform messy lab data", "Value proposition")
    
    # Git repository
    print("\n📦 Git Repository:")
    all_checks_passed &= check_directory_exists(".git", "Git repository")
    
    # Summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED - Ready for deployment!")
        print("\nNext steps:")
        print("1. Follow GITHUB_SETUP.md to create GitHub repository")
        print("2. Follow NEON_SETUP.md to set up database")
        print("3. Follow STREAMLIT_CLOUD_SETUP.md to deploy")
        print("4. Use DEPLOYMENT_CHECKLIST.md to track progress")
        sys.exit(0)
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues before deployment")
        sys.exit(1)

if __name__ == "__main__":
    main()