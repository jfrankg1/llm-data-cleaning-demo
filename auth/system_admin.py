"""
System Administrator permissions and decorators for platform-wide administration
"""

import functools
from typing import Dict, Optional, Callable
import streamlit as st
from auth.database import is_system_admin


def is_system_administrator(user_id: str = None) -> bool:
    """
    Check if the current user (or specified user) is a system administrator.
    
    Args:
        user_id: Optional user ID to check. If None, uses current session user.
        
    Returns:
        bool: True if user is a system admin, False otherwise
    """
    if user_id is None:
        # Check current session user
        if not st.session_state.get('authentication_status'):
            return False
        
        # Get system admin status from session or database
        user_id = st.session_state.get('user_id')
        if not user_id:
            return False
    
    return is_system_admin(user_id)


def require_system_admin(func: Callable) -> Callable:
    """
    Decorator to require system admin permissions for a function.
    Shows error message if user is not a system admin.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not is_system_administrator():
            st.error("🚫 Access Denied: This feature requires system administrator privileges.")
            st.info("Only designated system administrators can access this functionality.")
            return None
        return func(*args, **kwargs)
    return wrapper


def system_admin_sidebar_menu():
    """
    Display system admin menu items in the sidebar.
    Should be called within the sidebar context.
    """
    st.markdown("### 🛡️ System Admin")
    
    # System admin navigation
    system_admin_pages = {
        "🏢 Organizations": "system_admin_organizations",
        "👥 All Users": "system_admin_users",
        "📊 Platform Stats": "system_admin_dashboard",
        "🔧 System Config": "system_admin_config",
        "📋 Audit Logs": "system_admin_logs"
    }
    
    for label, page_key in system_admin_pages.items():
        if st.button(label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state['current_page'] = page_key
            st.rerun()
    
    st.divider()


def check_system_admin_permission(action: str) -> bool:
    """
    Check if current user has permission to perform a system admin action.
    
    Args:
        action: The action to check (e.g., 'manage_users', 'deactivate_org', 'view_logs')
        
    Returns:
        bool: True if user has permission, False otherwise
    """
    # For now, all system admins have all permissions
    # Future: implement granular permissions for different types of system admins
    return is_system_administrator()


def get_system_admin_role() -> str:
    """
    Get the system admin role level of the current user.
    
    Returns:
        str: Role level ('super_admin', 'admin', 'monitor', 'none')
    """
    if is_system_administrator():
        # Future: implement different levels of system admin roles
        return 'super_admin'
    else:
        return 'none'


def log_system_admin_access(action: str, details: dict = None):
    """
    Log system admin access for audit purposes.
    
    Args:
        action: The action being performed
        details: Additional details about the action
    """
    from auth.database import log_system_admin_action
    
    user_id = st.session_state.get('user_id')
    if user_id and is_system_administrator():
        # Get IP and user agent from Streamlit context if available
        ip_address = None  # Streamlit doesn't provide easy access to this
        user_agent = None  # Streamlit doesn't provide easy access to this
        
        log_system_admin_action(
            user_id,
            action,
            'system',
            None,
            details,
            ip_address,
            user_agent
        )


def enforce_system_admin_security():
    """
    Enforce security measures for system admin access.
    This function should be called at the beginning of system admin pages.
    """
    # Check if user is authenticated
    if not st.session_state.get('authentication_status'):
        st.error("🔒 Authentication required")
        st.stop()
    
    # Check if user is system admin
    if not is_system_administrator():
        st.error("🚫 System administrator access required")
        st.stop()
    
    # Log access attempt
    log_system_admin_access('system_admin_page_access', {
        'page': st.session_state.get('current_page', 'unknown'),
        'timestamp': st.session_state.get('login_time')
    })


def format_system_admin_action(action_type: str, action_details: dict) -> str:
    """
    Format system admin action for display in logs.
    
    Args:
        action_type: Type of action performed
        action_details: Details about the action
        
    Returns:
        str: Formatted action description
    """
    action_descriptions = {
        'system_admin_granted': '👑 System admin privileges granted',
        'system_admin_revoked': '❌ System admin privileges revoked',
        'organization_deactivated': '🏢 Organization deactivated',
        'organization_activated': '✅ Organization activated',
        'user_deactivated': '👤 User account deactivated',
        'user_activated': '✅ User account activated',
        'config_updated': '⚙️ System configuration updated',
        'system_admin_page_access': '👁️ System admin page accessed'
    }
    
    base_description = action_descriptions.get(action_type, f"🔧 {action_type}")
    
    if action_details:
        if 'user_email' in action_details:
            base_description += f" for {action_details['user_email']}"
        elif 'organization_name' in action_details:
            base_description += f" for {action_details['organization_name']}"
        elif 'config_key' in action_details:
            base_description += f" - {action_details['config_key']}"
    
    return base_description


def get_system_admin_capabilities() -> list:
    """
    Get list of capabilities available to system administrators.
    
    Returns:
        list: List of capability descriptions
    """
    return [
        "👥 Manage all users across all organizations",
        "🏢 View and manage all organizations",
        "📊 Access platform-wide analytics and metrics",
        "⚙️ Configure system-wide settings",
        "📋 View comprehensive audit logs",
        "🚫 Deactivate organizations and users",
        "👑 Grant/revoke system admin privileges",
        "🔧 Access system health monitoring",
        "📈 View usage analytics across all organizations",
        "🛡️ Manage security settings and policies"
    ]