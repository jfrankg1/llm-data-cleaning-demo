-- Organizations table
CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    subscription_tier VARCHAR(50) DEFAULT 'free',
    description TEXT,
    website VARCHAR(255),
    logo_url VARCHAR(255),
    settings JSONB DEFAULT '{}',
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    is_system_admin BOOLEAN DEFAULT false,
    organization_id UUID REFERENCES organizations(id),
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

-- API Keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    key_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    last_used TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    organization_id UUID REFERENCES organizations(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'created',
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    result_file_url TEXT,
    metadata JSONB
);

-- Usage tracking table
CREATE TABLE IF NOT EXISTS usage_tracking (
    id SERIAL PRIMARY KEY,
    organization_id UUID REFERENCES organizations(id),
    user_id UUID REFERENCES users(id),
    metric_type VARCHAR(50), -- 'api_call', 'storage', 'experiment'
    metric_value INTEGER,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Usage limits table
CREATE TABLE IF NOT EXISTS usage_limits (
    organization_id UUID PRIMARY KEY REFERENCES organizations(id),
    experiments_monthly INTEGER DEFAULT 100,
    storage_mb INTEGER DEFAULT 1024,
    api_calls_daily INTEGER DEFAULT 1000
);

-- User sessions table for persistent session management
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT true,
    user_agent TEXT,
    ip_address INET
);

-- System admin audit log table
CREATE TABLE IF NOT EXISTS system_admin_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    admin_user_id UUID REFERENCES users(id),
    action_type VARCHAR(100) NOT NULL, -- 'user_created', 'user_deactivated', 'org_created', etc.
    target_type VARCHAR(50) NOT NULL,  -- 'user', 'organization', 'system'
    target_id UUID,                    -- ID of the affected entity
    action_details JSONB,              -- Additional details about the action
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- System configuration table
CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    updated_by UUID REFERENCES users(id)
);

-- User invitations table
CREATE TABLE IF NOT EXISTS user_invitations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    invited_by UUID REFERENCES users(id),
    role VARCHAR(50) DEFAULT 'member', -- 'admin', 'member'
    invitation_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    accepted_at TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Organization admin audit logs
CREATE TABLE IF NOT EXISTS org_admin_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    admin_user_id UUID REFERENCES users(id),
    action_type VARCHAR(100) NOT NULL,
    target_type VARCHAR(50) NOT NULL, -- 'user', 'organization', 'setting'
    target_id UUID,
    action_details JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Organization notification settings
CREATE TABLE IF NOT EXISTS org_notification_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    notification_type VARCHAR(100) NOT NULL, -- 'usage_warning', 'experiment_failed', etc.
    enabled BOOLEAN DEFAULT true,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(organization_id, notification_type)
);

-- Password history table for preventing password reuse
CREATE TABLE IF NOT EXISTS password_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Account lockout tracking table
CREATE TABLE IF NOT EXISTS account_lockouts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    ip_address INET,
    failed_attempts INTEGER DEFAULT 1,
    last_attempt TIMESTAMP DEFAULT NOW(),
    locked_until TIMESTAMP,
    is_locked BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Password breach check cache (optional performance optimization)
CREATE TABLE IF NOT EXISTS password_breach_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    password_hash_prefix VARCHAR(5) NOT NULL, -- First 5 chars of SHA-1
    is_breached BOOLEAN NOT NULL,
    breach_count INTEGER DEFAULT 0,
    last_checked TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP DEFAULT (NOW() + INTERVAL '7 days'),
    UNIQUE(password_hash_prefix)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_system_admin ON users(is_system_admin);
CREATE INDEX IF NOT EXISTS idx_experiments_user ON experiments(user_id);
CREATE INDEX IF NOT EXISTS idx_experiments_org ON experiments(organization_id);
CREATE INDEX IF NOT EXISTS idx_usage_org_timestamp ON usage_tracking(organization_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_password_history_user ON password_history(user_id);
CREATE INDEX IF NOT EXISTS idx_account_lockouts_user ON account_lockouts(user_id);
CREATE INDEX IF NOT EXISTS idx_account_lockouts_ip ON account_lockouts(ip_address);
CREATE INDEX IF NOT EXISTS idx_password_breach_cache_prefix ON password_breach_cache(password_hash_prefix);
CREATE INDEX IF NOT EXISTS idx_password_breach_cache_expires ON password_breach_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON user_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_admin_logs_admin_user ON system_admin_logs(admin_user_id);
CREATE INDEX IF NOT EXISTS idx_admin_logs_timestamp ON system_admin_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_admin_logs_action_type ON system_admin_logs(action_type);
CREATE INDEX IF NOT EXISTS idx_invitations_org ON user_invitations(organization_id);
CREATE INDEX IF NOT EXISTS idx_invitations_token ON user_invitations(invitation_token);
CREATE INDEX IF NOT EXISTS idx_invitations_email ON user_invitations(email);
CREATE INDEX IF NOT EXISTS idx_org_admin_logs_org ON org_admin_logs(organization_id);
CREATE INDEX IF NOT EXISTS idx_org_admin_logs_timestamp ON org_admin_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_notification_settings_org ON org_notification_settings(organization_id);

-- Security audit log table for cross-tenant access monitoring
CREATE TABLE IF NOT EXISTS security_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    user_email VARCHAR(255),
    user_org_id UUID,
    target_org_id UUID,
    attempt_type VARCHAR(100) NOT NULL,
    success BOOLEAN NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    additional_data JSONB DEFAULT '{}'
);

-- Security audit log indexes
CREATE INDEX IF NOT EXISTS idx_security_audit_user ON security_audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_security_audit_target_org ON security_audit_log(target_org_id);
CREATE INDEX IF NOT EXISTS idx_security_audit_timestamp ON security_audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_audit_attempt_type ON security_audit_log(attempt_type);
CREATE INDEX IF NOT EXISTS idx_security_audit_cross_tenant ON security_audit_log(user_org_id, target_org_id) WHERE user_org_id != target_org_id;

-- Rate limit violations table for DoS protection monitoring
CREATE TABLE IF NOT EXISTS rate_limit_violations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    ip_address INET,
    limit_type VARCHAR(50) NOT NULL,
    limit_value INTEGER NOT NULL,
    current_count INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    additional_data JSONB DEFAULT '{}'
);

-- Rate limit violations indexes
CREATE INDEX IF NOT EXISTS idx_rate_limit_user ON rate_limit_violations(user_id);
CREATE INDEX IF NOT EXISTS idx_rate_limit_ip ON rate_limit_violations(ip_address);
CREATE INDEX IF NOT EXISTS idx_rate_limit_timestamp ON rate_limit_violations(timestamp);
CREATE INDEX IF NOT EXISTS idx_rate_limit_type ON rate_limit_violations(limit_type);