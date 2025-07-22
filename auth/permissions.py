"""
Permission management and decorators for organization admin functionality
"""

import functools
from typing import Dict, Optional, Callable
import streamlit as st


def is_org_admin(user_id: str = None) -> bool:
    """
    Check if the current user (or specified user) is an organization admin.
    
    Args:
        user_id: Optional user ID to check. If None, uses current session user.
        
    Returns:
        bool: True if user is an admin, False otherwise
    """
    if user_id is None:
        # Check current session user
        if not st.session_state.get('authentication_status'):
            return False
        
        # Get admin status from session or database
        return st.session_state.get('is_admin', False)
    else:
        # Check specific user (would need database query)
        # For now, this is a placeholder - implement with database check
        from auth.database import get_user_by_id
        user = get_user_by_id(user_id)
        return user and user.get('is_admin', False)


def is_org_admin_in_organization(user_id: str, org_id: str) -> bool:
    """
    SECURITY FUNCTION: Check if a user is an admin within a specific organization.
    This prevents cross-tenant authorization bypass.
    
    Args:
        user_id: User ID to check
        org_id: Organization ID that user must belong to
        
    Returns:
        bool: True if user is admin AND in the specified organization
    """
    from auth.database import get_user_by_id
    
    user = get_user_by_id(user_id)
    if not user:
        # Log security event: invalid user access attempt
        log_cross_tenant_access_attempt(user_id, org_id, 'invalid_user', False)
        return False
    
    user_org_id = user.get('organization_id')
    is_admin = user.get('is_admin', False)
    
    # Check if this is a cross-tenant access attempt
    if user_org_id != org_id and is_admin:
        # Log security event: cross-tenant admin access attempt
        log_cross_tenant_access_attempt(user_id, org_id, 'cross_tenant_admin_attempt', False)
        return False
    
    # User must be in the specified organization AND be an admin
    access_granted = (user_org_id == org_id and is_admin)
    
    if not access_granted:
        # Log failed access attempt
        log_cross_tenant_access_attempt(user_id, org_id, 'access_denied', False)
    
    return access_granted


def log_cross_tenant_access_attempt(user_id: str, target_org_id: str, attempt_type: str, success: bool):
    """
    Log cross-tenant access attempts for security auditing.
    
    Args:
        user_id: User attempting access
        target_org_id: Organization being accessed
        attempt_type: Type of access attempt
        success: Whether the attempt succeeded
    """
    try:
        from auth.database import get_db_connection
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get user's actual organization
        cur.execute("SELECT organization_id, email FROM users WHERE id = %s", (user_id,))
        user_data = cur.fetchone()
        
        # Log the security event
        cur.execute("""
            INSERT INTO security_audit_log 
            (user_id, user_email, user_org_id, target_org_id, attempt_type, success, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """, (
            user_id,
            user_data['email'] if user_data else 'unknown',
            user_data['organization_id'] if user_data else None,
            target_org_id,
            attempt_type,
            success
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
    except Exception:
        # Don't let logging failures affect the main function
        pass


def require_admin(func: Callable) -> Callable:
    """
    Decorator to require admin permissions for a function.
    Shows error message if user is not an admin.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not is_org_admin():
            st.error("⛔ Access Denied: This feature requires administrator privileges.")
            st.info("Please contact your organization administrator for access.")
            return None
        return func(*args, **kwargs)
    return wrapper


def require_authentication(func: Callable) -> Callable:
    """
    Decorator to require user authentication.
    Redirects to login if not authenticated.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not st.session_state.get('authentication_status'):
            st.error("🔒 Please log in to access this feature.")
            st.stop()
        return func(*args, **kwargs)
    return wrapper


def check_user_permission(user_id: str, org_id: str, permission: str) -> bool:
    """
    Check if a user has a specific permission within an organization.
    
    Args:
        user_id: User ID to check
        org_id: Organization ID
        permission: Permission to check (e.g., 'view_all_experiments', 'manage_users')
        
    Returns:
        bool: True if user has permission, False otherwise
    """
    # SECURITY FIX: Verify user is in the same organization and is admin
    if is_org_admin_in_organization(user_id, org_id):
        return True
    
    # Future: implement role-based permissions
    # Non-admins have limited permissions
    allowed_permissions = ['view_own_experiments', 'create_experiment']
    return permission in allowed_permissions


def filter_by_permission(items: list, user_id: str, permission: str, org_id: str = None) -> list:
    """
    Filter a list of items based on user permissions.
    
    Args:
        items: List of items to filter (e.g., experiments, users)
        user_id: User ID to check permissions for
        permission: Permission required to view items
        org_id: Organization ID to check permissions within
        
    Returns:
        list: Filtered list of items user has permission to view
    """
    # SECURITY FIX: Get user's organization if not provided
    if org_id is None:
        from auth.database import get_user_by_id
        user = get_user_by_id(user_id)
        if not user:
            return []
        org_id = user['organization_id']
    
    if is_org_admin_in_organization(user_id, org_id):
        # Admins can see everything within their organization only
        return [item for item in items if item.get('organization_id') == org_id]
    
    # Non-admins can only see their own items
    if permission == 'view_experiments':
        return [item for item in items if item.get('user_id') == user_id]
    
    return []


def get_user_role(user_id: str) -> str:
    """
    Get the role of a user within their organization.
    
    Args:
        user_id: User ID to check
        
    Returns:
        str: Role name ('admin', 'member', 'viewer')
    """
    if is_org_admin(user_id):
        return 'admin'
    
    # Future: implement more granular roles
    return 'member'


def has_permission_to_modify_user(actor_id: str, target_user_id: str, org_id: str) -> bool:
    """
    Check if a user can modify another user's settings.
    
    Args:
        actor_id: ID of user trying to make changes
        target_user_id: ID of user being modified
        org_id: Organization ID
        
    Returns:
        bool: True if actor can modify target user
    """
    # SECURITY FIX: Verify both users are in same organization
    from auth.database import get_user_by_id
    
    actor = get_user_by_id(actor_id)
    target = get_user_by_id(target_user_id)
    
    if not actor or not target:
        return False
    
    # Both users must be in the specified organization
    if actor['organization_id'] != org_id or target['organization_id'] != org_id:
        return False
    
    # Users can modify their own settings
    if actor_id == target_user_id:
        return True
    
    # Only organization admins can modify other users in the same organization
    return is_org_admin_in_organization(actor_id, org_id)


def admin_sidebar_menu():
    """
    Display admin-only menu items in the sidebar.
    Should be called within the sidebar context.
    """
    st.markdown("### 👨‍💼 Admin Tools")
    
    # Admin navigation
    admin_pages = {
        "👥 User Management": "admin_users",
        "📊 Organization Stats": "admin_stats",
        "🔧 Settings": "admin_settings",
        "📋 Audit Logs": "admin_audit_logs"
    }
    
    for label, page_key in admin_pages.items():
        if st.button(label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state['current_page'] = page_key
            st.rerun()
    
    st.divider()


def enforce_usage_limits(org_id: str, user_id: str, resource: str) -> tuple[bool, str]:
    """
    Check if an action would exceed usage limits.
    Admins may have different limits or exemptions.
    
    Args:
        org_id: Organization ID
        user_id: User ID
        resource: Resource type ('experiment', 'storage', 'api_call')
        
    Returns:
        tuple: (allowed: bool, message: str)
    """
    from auth.database import check_usage_limits
    
    # Check standard limits
    within_limits = check_usage_limits(org_id, resource)
    
    if not within_limits:
        # SECURITY FIX: Verify admin is in the same organization
        if is_org_admin_in_organization(user_id, org_id):
            # Admins might get a warning but can proceed (within their organization only)
            return True, f"⚠️ Warning: {resource} limit exceeded. Admin override applied."
        else:
            return False, f"❌ {resource.title()} limit reached. Please contact your administrator."
    
    return True, ""