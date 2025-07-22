# LLM Data Cleaning System - Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub Account**: Required for Streamlit Cloud integration
2. **Streamlit Cloud Account**: Sign up at https://streamlit.io/cloud
3. **PostgreSQL Database**: We recommend Neon.tech (free tier)
4. **Anthropic API Key**: Already included in demo (.env.example)

## Step 1: Database Setup (Neon.tech)

### 1.1 Create Neon Account
1. Go to https://neon.tech
2. Sign up for free account
3. Create new project: "llm-data-cleaning-demo"

### 1.2 Create Database
1. In Neon dashboard, create database: `data_cleaning_demo`
2. Copy connection string (looks like: `postgresql://user:pass@ep-xxx.aws.neon.tech/data_cleaning_demo`)
3. Save this for Step 3

### 1.3 Initialize Schema
1. Open Neon SQL Editor
2. Copy contents of `auth/schema.sql`
3. Run the SQL to create tables

### 1.4 Create Demo Data (Optional)
```sql
-- Create demo organization
INSERT INTO organizations (name, description) 
VALUES ('Demo Laboratory', 'Experience AI-powered data cleaning');

-- Note the organization ID from the result
-- Create demo user (replace [org-id] with actual ID)
INSERT INTO users (email, password_hash, full_name, organization_id, is_admin)
VALUES ('demo@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiGSBGzKyKtK', 'Demo User', '[org-id]', true);
-- Password is: demo123
```

## Step 2: GitHub Repository

### 2.1 Create Repository
1. Go to GitHub
2. Create new repository: `llm-data-cleaning-demo`
3. Make it PUBLIC (required for free Streamlit Cloud)
4. Don't initialize with README

### 2.2 Push Code
```bash
# In the llm-data-cleaning-demo directory
git remote add origin https://github.com/YOUR_USERNAME/llm-data-cleaning-demo.git
git branch -M main
git push -u origin main
```

## Step 3: Streamlit Cloud Deployment

### 3.1 Connect to Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click "New app"
3. Connect your GitHub account if not already connected

### 3.2 Configure Deployment
1. **Repository**: Select `llm-data-cleaning-demo`
2. **Branch**: `main`
3. **Main file path**: `streamlit_app.py`
4. **App URL**: Choose custom URL like `llm-data-cleaning`

### 3.3 Add Secrets
Click "Advanced settings" and add secrets:

```toml
# Database Configuration
DATABASE_URL = "postgresql://user:pass@ep-xxx.aws.neon.tech/data_cleaning_demo"
DB_HOST = "ep-xxx.aws.neon.tech"
DB_PORT = "5432"
DB_NAME = "data_cleaning_demo"
DB_USER = "your_neon_user"
DB_PASSWORD = "your_neon_password"

# API Keys (contact support for demo key)
ANTHROPIC_API_KEY = "sk-ant-api03-your-demo-key-here"

# Security
SECRET_KEY = "demo-deployment-secret-key-change-in-production"
SESSION_TIMEOUT = "3600"
ENVIRONMENT = "demo"

# App Configuration
APP_TITLE = "LLM Data Cleaning System"
APP_SUBTITLE = "AI-Powered Laboratory Data Cleaning"

# Demo Configuration
DEMO_MODE = "true"
DEMO_USER_EMAIL = "demo@example.com"
DEMO_USER_PASSWORD = "demo123"

# Feature Flags
ENABLE_TIME_SERIES = "true"
ENABLE_ADMIN_DASHBOARD = "true"
```

### 3.4 Deploy
1. Click "Deploy!"
2. Wait for deployment (usually 2-5 minutes)
3. Your app will be available at: `https://llm-data-cleaning.streamlit.app`

## Step 4: Verify Deployment

### 4.1 Test Basic Access
1. Visit your app URL
2. Should see login page
3. Check for any error messages

### 4.2 Test Demo Login
1. Use demo credentials:
   - Email: `demo@example.com`
   - Password: `demo123`
2. Should see main dashboard

### 4.3 Test File Processing
1. Upload sample file from `demo_data/raw/`
2. Process and verify results
3. Download cleaned data

## Troubleshooting

### Database Connection Issues
- Verify DATABASE_URL is correct in secrets
- Check Neon dashboard for connection limits
- Ensure schema.sql was run successfully

### Import Errors
- Check logs in Streamlit Cloud dashboard
- Verify all dependencies in requirements.txt
- Ensure no test dependencies included

### Performance Issues
- Monitor Streamlit Cloud resource usage
- Consider upgrading to paid tier if needed
- Optimize database queries

## Monitoring

### Streamlit Cloud Dashboard
- View logs
- Monitor resource usage
- Track visitor analytics

### Database Monitoring
- Check Neon dashboard for:
  - Connection count
  - Query performance
  - Storage usage

## Maintenance

### Updating the App
1. Make changes locally
2. Test thoroughly
3. Push to GitHub
4. Streamlit Cloud auto-deploys

### Database Maintenance
- Regular backups (Neon provides automatic backups)
- Monitor storage usage
- Clean old demo data periodically

## Security Notes

1. **Demo API Key**: The included Anthropic key is for demo only
2. **Production Deployment**: Use your own API keys
3. **Database Security**: Use strong passwords in production
4. **Secret Rotation**: Regularly update secrets

## Support

For deployment issues:
- Streamlit Cloud: https://docs.streamlit.io/streamlit-cloud
- Neon: https://neon.tech/docs
- GitHub Issues: Create issue in your repository

## Next Steps

1. Customize branding and messaging
2. Add more demo scenarios
3. Implement analytics tracking
4. Set up custom domain (optional)
5. Configure email notifications

---

Happy deploying! 🚀