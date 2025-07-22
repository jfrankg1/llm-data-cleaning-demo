"""
System Admin Service
Handles all system administration database operations
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)

class SystemAdminService(BaseRepository):
    """Service for system administration database operations"""
    
    def __init__(self):
        """Initialize system admin service"""
        super().__init__()

    def is_system_admin(self, user_id: str) -> bool:
        """
        Check if user is a system administrator
        
        Args:
            user_id: User ID to check
            
        Returns:
            bool: True if user is system admin
        """
        try:
            query = "SELECT is_system_admin FROM users WHERE id = %s"
            result = self.execute_query(query, (user_id,), fetch_one=True)
            
            return result['is_system_admin'] if result else False
            
        except Exception as e:
            self.log_error("is_system_admin", e, {'user_id': user_id})
            return False

    def get_all_users_system_admin(self) -> List[Dict[str, Any]]:
        """
        Get all users for system admin view (across all organizations)
        
        Returns:
            List of all users with organization information
        """
        try:
            query = """
                SELECT 
                    u.id, u.email, u.username, u.is_admin, u.is_system_admin, u.is_active, 
                    u.created_at, u.last_login, o.name as organization_name, u.organization_id
                FROM users u
                JOIN organizations o ON u.organization_id = o.id
                ORDER BY u.created_at DESC
            """
            
            return self.execute_query(query)
            
        except Exception as e:
            self.handle_db_error("get_all_users_system_admin", e)
            return []

    def get_all_organizations_system_admin(self) -> List[Dict[str, Any]]:
        """
        Get all organizations for system admin view
        
        Returns:
            List of all organizations with statistics
        """
        try:
            query = """
                SELECT 
                    o.id, o.name, o.created_at, o.is_active,
                    COUNT(u.id) as user_count,
                    COUNT(CASE WHEN u.is_active THEN 1 END) as active_user_count,
                    COUNT(CASE WHEN u.is_admin THEN 1 END) as admin_count,
                    COUNT(e.id) as experiment_count,
                    MAX(u.last_login) as last_user_activity
                FROM organizations o
                LEFT JOIN users u ON o.id = u.organization_id
                LEFT JOIN experiments e ON o.id = e.organization_id
                GROUP BY o.id, o.name, o.created_at, o.is_active
                ORDER BY o.created_at DESC
            """
            
            return self.execute_query(query)
            
        except Exception as e:
            self.handle_db_error("get_all_organizations_system_admin", e)
            return []

    def update_user_system_admin_status(self, user_id: str, is_system_admin: bool, admin_user_id: str) -> bool:
        """
        Update user's system admin status
        
        Args:
            user_id: User ID to update
            is_system_admin: New system admin status
            admin_user_id: ID of admin performing the action
            
        Returns:
            bool: True if update succeeded
        """
        try:
            # Verify the admin has system admin privileges
            if not self.is_system_admin(admin_user_id):
                logger.warning(f"User {admin_user_id} attempted to modify system admin status without privileges")
                return False
            
            # Update user status
            query = "UPDATE users SET is_system_admin = %s WHERE id = %s"
            self.execute_query(query, (is_system_admin, user_id), fetch_all=False)
            
            # Log the action
            action_type = "grant_system_admin" if is_system_admin else "revoke_system_admin"
            self.log_system_admin_action(admin_user_id, action_type, "user", user_id, {
                'is_system_admin': is_system_admin
            })
            
            status = "granted" if is_system_admin else "revoked"
            logger.info(f"System admin privileges {status} for user {user_id} by admin {admin_user_id}")
            return True
            
        except Exception as e:
            self.handle_db_error("update_user_system_admin_status", e, {
                'user_id': user_id, 'is_system_admin': is_system_admin, 'admin_user_id': admin_user_id
            })
            return False

    def deactivate_organization(self, org_id: str, admin_user_id: str) -> bool:
        """
        Deactivate an organization and all its users
        
        Args:
            org_id: Organization ID to deactivate
            admin_user_id: ID of admin performing the action
            
        Returns:
            bool: True if deactivation succeeded
        """
        try:
            # Verify admin has system admin privileges
            if not self.is_system_admin(admin_user_id):
                logger.warning(f"User {admin_user_id} attempted to deactivate organization without system admin privileges")
                return False
            
            # Deactivate organization and all its users
            operations = [
                ("UPDATE users SET is_active = false WHERE organization_id = %s", (org_id,)),
                ("UPDATE organizations SET is_active = false WHERE id = %s", (org_id,))
            ]
            
            success = self.execute_transaction(operations)
            
            if success:
                # Log the action
                self.log_system_admin_action(admin_user_id, "deactivate_organization", "organization", org_id)
                logger.warning(f"Organization {org_id} deactivated by system admin {admin_user_id}")
            
            return success
            
        except Exception as e:
            self.handle_db_error("deactivate_organization", e, {'org_id': org_id, 'admin_user_id': admin_user_id})
            return False

    def log_system_admin_action(self, admin_user_id: str, action_type: str, target_type: str, 
                               target_id: str = None, action_details: Dict[str, Any] = None) -> bool:
        """
        Log system admin actions for audit trail
        
        Args:
            admin_user_id: ID of admin performing the action
            action_type: Type of action performed
            target_type: Type of target (user, organization, system)
            target_id: Optional ID of the target
            action_details: Optional additional details
            
        Returns:
            bool: True if logging succeeded
        """
        try:
            query = """
                INSERT INTO system_admin_log (admin_user_id, action_type, target_type, target_id, action_details)
                VALUES (%s, %s, %s, %s, %s)
            """
            params = (admin_user_id, action_type, target_type, target_id, action_details)
            
            self.execute_query(query, params, fetch_all=False)
            
            logger.info(f"Logged system admin action: {action_type} on {target_type} {target_id} by {admin_user_id}")
            return True
            
        except Exception as e:
            self.log_error("log_system_admin_action", e, {
                'admin_user_id': admin_user_id, 'action_type': action_type, 'target_type': target_type
            })
            return False

    def get_system_admin_log(self, limit: int = 100, offset: int = 0, 
                            admin_user_id: str = None) -> List[Dict[str, Any]]:
        """
        Get system admin action log
        
        Args:
            limit: Maximum number of log entries to return
            offset: Number of entries to skip
            admin_user_id: Optional filter by admin user
            
        Returns:
            List of log entries
        """
        try:
            if admin_user_id:
                where_clause = "WHERE sal.admin_user_id = %s"
                params = (admin_user_id, limit, offset)
            else:
                where_clause = ""
                params = (limit, offset)
            
            query = f"""
                SELECT 
                    sal.*, 
                    u.email as admin_email, 
                    u.username as admin_username
                FROM system_admin_log sal
                JOIN users u ON sal.admin_user_id = u.id
                {where_clause}
                ORDER BY sal.timestamp DESC
                LIMIT %s OFFSET %s
            """
            
            return self.execute_query(query, params)
            
        except Exception as e:
            self.handle_db_error("get_system_admin_log", e, {'admin_user_id': admin_user_id})
            return []

    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics for admin dashboard
        
        Returns:
            Dict with system-wide statistics
        """
        try:
            stats = {}
            
            # User statistics
            user_query = """
                SELECT 
                    COUNT(*) as total_users,
                    COUNT(CASE WHEN is_active THEN 1 END) as active_users,
                    COUNT(CASE WHEN is_admin THEN 1 END) as admin_users,
                    COUNT(CASE WHEN is_system_admin THEN 1 END) as system_admin_users,
                    COUNT(CASE WHEN created_at >= date_trunc('month', CURRENT_DATE) THEN 1 END) as new_users_this_month
                FROM users
            """
            user_stats = self.execute_query(user_query, fetch_one=True)
            stats['users'] = user_stats if user_stats else {}
            
            # Organization statistics
            org_query = """
                SELECT 
                    COUNT(*) as total_organizations,
                    COUNT(CASE WHEN is_active THEN 1 END) as active_organizations,
                    COUNT(CASE WHEN created_at >= date_trunc('month', CURRENT_DATE) THEN 1 END) as new_organizations_this_month
                FROM organizations
            """
            org_stats = self.execute_query(org_query, fetch_one=True)
            stats['organizations'] = org_stats if org_stats else {}
            
            # Experiment statistics
            exp_query = """
                SELECT 
                    COUNT(*) as total_experiments,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_experiments,
                    COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing_experiments,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_experiments,
                    COUNT(CASE WHEN created_at >= date_trunc('month', CURRENT_DATE) THEN 1 END) as experiments_this_month,
                    COUNT(CASE WHEN created_at >= CURRENT_DATE THEN 1 END) as experiments_today
                FROM experiments
            """
            exp_stats = self.execute_query(exp_query, fetch_one=True)
            stats['experiments'] = exp_stats if exp_stats else {}
            
            # Usage statistics
            usage_query = """
                SELECT 
                    metric_type,
                    SUM(metric_value) as total_value,
                    COUNT(*) as event_count
                FROM usage_tracking 
                WHERE timestamp >= date_trunc('month', CURRENT_DATE)
                GROUP BY metric_type
            """
            usage_results = self.execute_query(usage_query)
            stats['usage_this_month'] = {row['metric_type']: {
                'total_value': row['total_value'], 
                'event_count': row['event_count']
            } for row in usage_results}
            
            return stats
            
        except Exception as e:
            self.handle_db_error("get_system_statistics", e)
            return {}

    def set_system_config(self, config_key: str, config_value: Dict[str, Any], 
                         description: str, admin_user_id: str) -> bool:
        """
        Set system configuration values
        
        Args:
            config_key: Configuration key
            config_value: Configuration value (JSON)
            description: Description of the configuration
            admin_user_id: ID of admin setting the config
            
        Returns:
            bool: True if setting succeeded
        """
        try:
            # Verify admin has system admin privileges
            if not self.is_system_admin(admin_user_id):
                logger.warning(f"User {admin_user_id} attempted to set system config without privileges")
                return False
            
            # Insert or update configuration
            query = """
                INSERT INTO system_config (config_key, config_value, description, updated_by)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (config_key) 
                DO UPDATE SET 
                    config_value = EXCLUDED.config_value,
                    description = EXCLUDED.description,
                    updated_by = EXCLUDED.updated_by,
                    updated_at = CURRENT_TIMESTAMP
            """
            params = (config_key, config_value, description, admin_user_id)
            
            self.execute_query(query, params, fetch_all=False)
            
            # Log the action
            self.log_system_admin_action(admin_user_id, "set_config", "system", config_key, {
                'config_key': config_key,
                'description': description
            })
            
            logger.info(f"System config {config_key} set by admin {admin_user_id}")
            return True
            
        except Exception as e:
            self.handle_db_error("set_system_config", e, {
                'config_key': config_key, 'admin_user_id': admin_user_id
            })
            return False

    def get_system_config(self, config_key: str = None) -> Dict[str, Any]:
        """
        Get system configuration values
        
        Args:
            config_key: Optional specific config key to retrieve
            
        Returns:
            Dict with configuration values
        """
        try:
            if config_key:
                query = "SELECT * FROM system_config WHERE config_key = %s"
                params = (config_key,)
                result = self.execute_query(query, params, fetch_one=True)
                return result if result else {}
            else:
                query = "SELECT * FROM system_config ORDER BY config_key"
                results = self.execute_query(query)
                return {row['config_key']: row for row in results}
            
        except Exception as e:
            self.handle_db_error("get_system_config", e, {'config_key': config_key})
            return {}

    def get_user_detailed_usage(self, org_id: str, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Get detailed usage information for users in an organization
        
        Args:
            org_id: Organization ID
            user_id: Optional specific user ID
            
        Returns:
            List of detailed usage information
        """
        try:
            where_conditions = ["u.organization_id = %s"]
            params = [org_id]
            
            if user_id:
                where_conditions.append("u.id = %s")
                params.append(user_id)
            
            where_clause = " AND ".join(where_conditions)
            
            query = f"""
                SELECT 
                    u.id, u.email, u.username, u.is_admin, u.is_active, u.created_at, u.last_login,
                    COUNT(e.id) as experiment_count,
                    COUNT(CASE WHEN e.status = 'completed' THEN 1 END) as completed_experiments,
                    COUNT(CASE WHEN e.status = 'failed' THEN 1 END) as failed_experiments,
                    MAX(e.created_at) as last_experiment_date,
                    COALESCE(SUM(CASE WHEN ut.metric_type = 'api_call' THEN ut.metric_value END), 0) as api_calls,
                    COALESCE(SUM(CASE WHEN ut.metric_type = 'storage' THEN ut.metric_value END), 0) as storage_used
                FROM users u
                LEFT JOIN experiments e ON u.id = e.user_id
                LEFT JOIN usage_tracking ut ON u.id = ut.user_id
                WHERE {where_clause}
                GROUP BY u.id, u.email, u.username, u.is_admin, u.is_active, u.created_at, u.last_login
                ORDER BY experiment_count DESC
            """
            
            return self.execute_query(query, tuple(params))
            
        except Exception as e:
            self.handle_db_error("get_user_detailed_usage", e, {'org_id': org_id, 'user_id': user_id})
            return []