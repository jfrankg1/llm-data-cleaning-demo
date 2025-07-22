# Streamlit Cloud Deployment Instructions

## Prerequisites
- ✅ GitHub repository created and pushed
- ✅ Neon database set up with demo data
- ✅ Connection details saved

## Step 1: Create Streamlit Cloud Account

1. **Go to**: https://streamlit.io/cloud
2. **Click "Sign up"**
3. **Connect with GitHub** (recommended)
4. **Verify email** if required

## Step 2: Deploy New App

1. **Click "New app"** in Streamlit Cloud dashboard
2. **Connect repository**:
   - Repository: `YOUR_USERNAME/llm-data-cleaning-demo`
   - Branch: `main`
   - Main file path: `streamlit_app.py`

3. **App URL** (optional):
   - Custom URL: `llm-data-cleaning` (if available)
   - Final URL will be: `https://llm-data-cleaning.streamlit.app`

## Step 3: Configure Advanced Settings

Click **"Advanced settings"** and add these secrets:

### Database Configuration
```toml
# Database Configuration (use your actual Neon details)
DATABASE_URL = "postgresql://user:password@ep-xxx.aws.neon.tech/data_cleaning_demo"
DB_HOST = "ep-xxx.aws.neon.tech"
DB_PORT = "5432"
DB_NAME = "data_cleaning_demo"
DB_USER = "your_neon_user"
DB_PASSWORD = "your_neon_password"
```

### API Keys
```toml
# API Keys (contact support for demo key)
ANTHROPIC_API_KEY = "sk-ant-api03-your-demo-key-here"
```

### Security Settings
```toml
# Security
SECRET_KEY = "demo-deployment-secret-key-change-in-production"
SESSION_TIMEOUT = "3600"
ENVIRONMENT = "demo"
```

### App Configuration
```toml
# App Configuration
APP_TITLE = "LLM Data Cleaning System"
APP_SUBTITLE = "AI-Powered Laboratory Data Cleaning"
```

### Demo Settings
```toml
# Demo Configuration
DEMO_MODE = "true"
DEMO_USER_EMAIL = "demo@example.com"
DEMO_USER_PASSWORD = "demo123"
```

### Feature Flags
```toml
# Feature Flags
ENABLE_TIME_SERIES = "true"
ENABLE_ADMIN_DASHBOARD = "true"
ENABLE_API_ACCESS = "false"
```

## Step 4: Deploy Application

1. **Click "Deploy!"**
2. **Wait for deployment** (usually 2-5 minutes)
3. **Monitor logs** for any errors
4. **App will be available** at your chosen URL

## Step 5: Test Deployment

### Basic Functionality Test
1. **Visit your app URL**
2. **Should see login page** with LLM Data Cleaning System branding
3. **No error messages** should appear

### Demo Login Test
1. **Use demo credentials**:
   - Email: `demo@example.com`
   - Password: `demo123`
2. **Should see main dashboard**
3. **All tabs should be accessible**

### File Processing Test
1. **Go to "New Analysis" tab**
2. **Upload a file** from demo_data folder
3. **Should process successfully**
4. **Download results** should work

### Admin Dashboard Test
1. **Click admin menu** (if available)
2. **Should show organization statistics**
3. **User management should be accessible**

## Step 6: Troubleshooting

### Common Issues

**Database Connection Errors**
- Verify DATABASE_URL is correct in secrets
- Check Neon dashboard for connection limits
- Ensure schema was properly initialized

**Import Errors**
- Check deployment logs in Streamlit Cloud
- Verify requirements.txt has all dependencies
- Ensure no test dependencies are included

**Performance Issues**
- Monitor resource usage in dashboard
- Consider upgrading to paid tier if needed
- Optimize database queries

**Login Issues**
- Verify demo user was created in database
- Check password hash is correct
- Ensure organization exists

### Getting Help

**Streamlit Cloud Issues**:
- Documentation: https://docs.streamlit.io/streamlit-cloud
- Community: https://discuss.streamlit.io

**Neon Database Issues**:
- Documentation: https://neon.tech/docs
- Support: In Neon dashboard

## Step 7: Share and Monitor

### Share with Team
1. **Send app URL** to stakeholders
2. **Provide demo credentials**
3. **Share DEMO_SCENARIOS.md** with sales team

### Monitor Performance
1. **Check Streamlit Cloud dashboard** for:
   - App uptime
   - Visitor analytics
   - Error logs
   - Resource usage

2. **Check Neon dashboard** for:
   - Database connections
   - Query performance
   - Storage usage

## Success Criteria

✅ App loads without errors
✅ Demo login works
✅ File upload and processing functional
✅ Admin dashboard accessible (if user is admin)
✅ Download features working
✅ All documentation displays properly

## Next Actions After Deployment

1. **Test all demo scenarios** from DEMO_SCENARIOS.md
2. **Share with sales team** for customer demos
3. **Monitor usage** and performance
4. **Collect feedback** from initial users
5. **Plan production enhancements**

---

**🎉 Congratulations!** Your LLM Data Cleaning System demo is now live!