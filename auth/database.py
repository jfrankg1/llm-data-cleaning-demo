"""
Database Access Layer - Refactored to use Repository Pattern
Legacy wrapper functions for backward compatibility with existing code
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv

# Import repository layer
from .repositories import (
    get_user_repository,
    get_organization_repository,
    get_experiment_repository,
    get_session_service,
    get_usage_tracking_service,
    get_system_admin_service
)

load_dotenv()

# Try to use environment configuration, fallback to direct env vars
try:
    from config import get_config
    config = get_config()
    DATABASE_URL = config.get_database_url()
except ImportError:
    # Fallback to direct environment variable access
    DATABASE_URL = os.getenv("DATABASE_URL")

# Set up logging
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a database connection - kept for legacy compatibility."""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

# =============================================================================
# ORGANIZATION FUNCTIONS - Now using OrganizationRepository
# =============================================================================

def create_organization(name: str) -> str:
    """Create a new organization and return its ID."""
    org_repo = get_organization_repository()
    return org_repo.create_organization(name)

def get_organization_usage(org_id: str) -> Dict:
    """Get current usage statistics for an organization."""
    org_repo = get_organization_repository()
    return org_repo.get_organization_usage(org_id)

def track_usage(org_id: str, user_id: str, metric_type: str, value: int = 1):
    """Track usage metrics."""
    usage_service = get_usage_tracking_service()
    return usage_service.track_usage(org_id, user_id, metric_type, value)

def check_usage_limits(org_id: str, metric_type: str) -> bool:
    """Check if organization is within usage limits."""
    org_repo = get_organization_repository()
    return org_repo.check_usage_limits(org_id, metric_type)

def get_usage_warnings(org_id: str) -> Dict:
    """Get usage warnings for metrics approaching limits (80% threshold)."""
    org_repo = get_organization_repository()
    return org_repo.get_usage_warnings(org_id)

def get_organization_users(org_id: str) -> List[Dict]:
    """Get all users in an organization."""
    org_repo = get_organization_repository()
    return org_repo.get_organization_users(org_id)

def get_organization_experiments(org_id: str, limit: int = 100, offset: int = 0) -> List[Dict]:
    """Get all experiments in an organization."""
    org_repo = get_organization_repository()
    return org_repo.get_organization_experiments(org_id, limit, offset)

def get_organization_stats(org_id: str) -> Dict:
    """Get comprehensive statistics for an organization."""
    org_repo = get_organization_repository()
    return org_repo.get_organization_stats(org_id)

# =============================================================================
# USER FUNCTIONS - Now using UserRepository
# =============================================================================

def create_user(email: str, password: str, full_name: str, organization_name: str) -> Dict:
    """Create a new user with organization."""
    user_repo = get_user_repository()
    return user_repo.create_user_with_organization(email, password, full_name, organization_name)

def authenticate_user(email: str, password: str, ip_address: str = None) -> Optional[Dict]:
    """Authenticate a user and return their details if successful."""
    user_repo = get_user_repository()
    
    # Handle legacy return format for account lockout
    result = user_repo.authenticate_user(email, password, ip_address)
    
    # If authentication failed due to account lock, check lockout status
    if result is None:
        user = user_repo.get_user_by_email(email)
        if user and user_repo.is_account_locked(user['id']):
            # Legacy format expected by existing code
            return {
                'error': 'account_locked',
                'message': "Account is locked due to too many failed login attempts."
            }
    
    return result

def get_user_by_id(user_id: str) -> Optional[Dict]:
    """Get user details by ID."""
    user_repo = get_user_repository()
    return user_repo.get_user_by_id(user_id)

def update_user_status(user_id: str, is_active: bool, admin_id: str) -> bool:
    """Activate or deactivate a user. Returns True on success."""
    user_repo = get_user_repository()
    return user_repo.update_user_status(user_id, is_active)

def update_user_admin_status(user_id: str, is_admin: bool, admin_id: str) -> bool:
    """Grant or revoke admin privileges. Returns True on success."""
    user_repo = get_user_repository()
    return user_repo.update_user_admin_status(user_id, is_admin)

def get_user_experiments(user_id: str, limit: int = 50, offset: int = 0) -> List[Dict]:
    """Get user's recent experiments."""
    exp_repo = get_experiment_repository()
    return exp_repo.get_user_experiments(user_id, limit, offset)

def get_user_detailed_usage(org_id: str, user_id: str = None) -> List[Dict]:
    """Get detailed usage information for users in an organization."""
    admin_service = get_system_admin_service()
    return admin_service.get_user_detailed_usage(org_id, user_id)

# =============================================================================
# PASSWORD SECURITY FUNCTIONS - Now using UserRepository
# =============================================================================

def store_password_history(user_id: str, password_hash: str) -> None:
    """Store password hash in history for reuse prevention."""
    user_repo = get_user_repository()
    return user_repo.store_password_history(user_id, password_hash)

def check_password_reuse(user_id: str, password: str) -> bool:
    """Check if password has been used recently."""
    user_repo = get_user_repository()
    return user_repo.check_password_reuse(user_id, password)

def record_failed_login(user_id: str = None, ip_address: str = None) -> None:
    """Record a failed login attempt for account lockout tracking."""
    if user_id:
        user_repo = get_user_repository()
        return user_repo.record_failed_login(user_id, ip_address)

def clear_failed_logins(user_id: str) -> None:
    """Clear failed login attempts after successful login."""
    user_repo = get_user_repository()
    return user_repo.clear_failed_logins(user_id)

def is_account_locked(user_id: str) -> Tuple[bool, Optional[datetime]]:
    """Check if account is currently locked."""
    user_repo = get_user_repository()
    is_locked = user_repo.is_account_locked(user_id)
    # Legacy format expected tuple - simplified for now
    return (is_locked, None)

# =============================================================================
# EXPERIMENT FUNCTIONS - Now using ExperimentRepository
# =============================================================================

def create_experiment(user_id: str, org_id: str, name: str, description: str = None) -> str:
    """Create a new experiment record."""
    exp_repo = get_experiment_repository()
    return exp_repo.create_experiment(user_id, org_id, name, description)

def update_experiment_status(exp_id: str, status: str, result_file_url: str = None, metadata: dict = None):
    """Update experiment status and results."""
    exp_repo = get_experiment_repository()
    return exp_repo.update_experiment_status(exp_id, status, result_file_url, metadata)

def get_experiment_by_id(exp_id: str) -> Optional[Dict]:
    """Get a specific experiment by ID."""
    exp_repo = get_experiment_repository()
    return exp_repo.get_experiment_by_id(exp_id)

def save_experiment_results(exp_id: str, results_data: str, file_name: str) -> str:
    """Save experiment results and return the file path."""
    exp_repo = get_experiment_repository()
    return exp_repo.save_experiment_results(exp_id, results_data, file_name)

# =============================================================================
# SESSION FUNCTIONS - Now using SessionService
# =============================================================================

def create_session(user_id: str, session_token: str, expires_at: datetime, user_agent: str = None, ip_address: str = None) -> bool:
    """Create a new user session."""
    session_service = get_session_service()
    return session_service.create_session(user_id, session_token, expires_at, user_agent, ip_address)

def get_session(session_token: str) -> Optional[Dict]:
    """Get session information by token."""
    session_service = get_session_service()
    return session_service.get_session(session_token)

def update_session_activity(session_token: str) -> bool:
    """Update session's last activity timestamp."""
    session_service = get_session_service()
    return session_service.update_session_activity(session_token)

def invalidate_session(session_token: str) -> bool:
    """Invalidate a specific session."""
    session_service = get_session_service()
    return session_service.invalidate_session(session_token)

def invalidate_user_sessions(user_id: str) -> bool:
    """Invalidate all sessions for a user."""
    session_service = get_session_service()
    return session_service.invalidate_user_sessions(user_id)

def cleanup_expired_sessions() -> int:
    """Clean up expired sessions from the database."""
    session_service = get_session_service()
    return session_service.cleanup_expired_sessions()

def get_user_sessions(user_id: str, active_only: bool = True) -> List[Dict]:
    """Get all sessions for a user."""
    session_service = get_session_service()
    return session_service.get_user_sessions(user_id, active_only)

# =============================================================================
# SYSTEM ADMIN FUNCTIONS - Now using SystemAdminService
# =============================================================================

def is_system_admin(user_id: str) -> bool:
    """Check if user is a system administrator."""
    admin_service = get_system_admin_service()
    return admin_service.is_system_admin(user_id)

def log_system_admin_action(admin_user_id: str, action_type: str, target_type: str, 
                          target_id: str = None, action_details: dict = None) -> bool:
    """Log system admin actions for audit trail."""
    admin_service = get_system_admin_service()
    return admin_service.log_system_admin_action(admin_user_id, action_type, target_type, target_id, action_details)

def get_all_users_system_admin() -> list:
    """Get all users for system admin (across all organizations)."""
    admin_service = get_system_admin_service()
    return admin_service.get_all_users_system_admin()

def update_user_system_admin_status(user_id: str, is_system_admin: bool, admin_user_id: str) -> bool:
    """Update user's system admin status."""
    admin_service = get_system_admin_service()
    return admin_service.update_user_system_admin_status(user_id, is_system_admin, admin_user_id)

def deactivate_organization(org_id: str, admin_user_id: str) -> bool:
    """Deactivate an organization and all its users."""
    admin_service = get_system_admin_service()
    return admin_service.deactivate_organization(org_id, admin_user_id)

def set_system_config(config_key: str, config_value: dict, description: str, admin_user_id: str) -> bool:
    """Set system configuration values."""
    admin_service = get_system_admin_service()
    return admin_service.set_system_config(config_key, config_value, description, admin_user_id)

def get_system_config(config_key: str = None) -> Dict:
    """Get system configuration values."""
    admin_service = get_system_admin_service()
    return admin_service.get_system_config(config_key)

# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =============================================================================

# Some functions that might be referenced but are now handled by repositories
def get_organization_by_id(org_id: str) -> Optional[Dict]:
    """Get organization by ID."""
    org_repo = get_organization_repository()
    return org_repo.get_organization_by_id(org_id)

def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email."""
    user_repo = get_user_repository()
    return user_repo.get_user_by_email(email)