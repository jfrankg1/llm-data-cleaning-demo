"""
Organization Repository
Handles all organization-related database operations
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)

class OrganizationRepository(BaseRepository):
    """Repository for organization-related database operations"""
    
    def __init__(self):
        """Initialize organization repository"""
        super().__init__()

    def create_organization(self, name: str) -> str:
        """
        Create a new organization
        
        Args:
            name: Organization name
            
        Returns:
            str: Organization ID
            
        Raises:
            Exception: If organization creation fails
        """
        try:
            # Validate required fields
            self.validate_required_fields({'name': name}, ['name'])
            
            # Generate organization ID
            org_id = self.generate_id()
            
            # Create organization and set default usage limits in transaction
            operations = [
                ("INSERT INTO organizations (id, name) VALUES (%s, %s)", (org_id, name)),
                ("INSERT INTO usage_limits (organization_id) VALUES (%s)", (org_id,))
            ]
            
            self.execute_transaction(operations)
            
            logger.info(f"Created organization {org_id} with name {name}")
            return org_id
            
        except Exception as e:
            self.handle_db_error("create_organization", e, {'name': name})

    def get_organization_by_id(self, org_id: str) -> Optional[Dict[str, Any]]:
        """
        Get organization by ID
        
        Args:
            org_id: Organization ID
            
        Returns:
            Organization information dict or None
        """
        try:
            query = "SELECT * FROM organizations WHERE id = %s"
            return self.execute_query(query, (org_id,), fetch_one=True)
            
        except Exception as e:
            self.handle_db_error("get_organization_by_id", e, {'org_id': org_id})
            return None

    def get_organization_users(self, org_id: str) -> List[Dict[str, Any]]:
        """
        Get all users in an organization
        
        Args:
            org_id: Organization ID
            
        Returns:
            List of user information dicts
        """
        try:
            query = """
                SELECT id, email, username, is_active, is_admin, created_at, last_login
                FROM users 
                WHERE organization_id = %s 
                ORDER BY created_at DESC
            """
            return self.execute_query(query, (org_id,))
            
        except Exception as e:
            self.handle_db_error("get_organization_users", e, {'org_id': org_id})
            return []

    def get_organization_experiments(self, org_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get all experiments in an organization
        
        Args:
            org_id: Organization ID
            limit: Maximum number of experiments to return
            offset: Number of experiments to skip
            
        Returns:
            List of experiment information dicts
        """
        try:
            query = """
                SELECT e.*, u.email as user_email, u.username as user_name
                FROM experiments e
                JOIN users u ON e.user_id = u.id
                WHERE e.organization_id = %s 
                ORDER BY e.created_at DESC 
                LIMIT %s OFFSET %s
            """
            return self.execute_query(query, (org_id, limit, offset))
            
        except Exception as e:
            self.handle_db_error("get_organization_experiments", e, {'org_id': org_id})
            return []

    def get_organization_stats(self, org_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for an organization
        
        Args:
            org_id: Organization ID
            
        Returns:
            Dict with organization statistics
        """
        try:
            stats = {}
            
            # User statistics
            user_query = """
                SELECT 
                    COUNT(*) as total_users,
                    COUNT(CASE WHEN is_active THEN 1 END) as active_users,
                    COUNT(CASE WHEN is_admin THEN 1 END) as admin_users
                FROM users 
                WHERE organization_id = %s
            """
            user_stats = self.execute_query(user_query, (org_id,), fetch_one=True)
            stats['users'] = user_stats if user_stats else {}
            
            # Experiment statistics
            exp_query = """
                SELECT 
                    COUNT(*) as total_experiments,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_experiments,
                    COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing_experiments,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_experiments,
                    COUNT(CASE WHEN created_at >= date_trunc('month', CURRENT_DATE) THEN 1 END) as experiments_this_month
                FROM experiments 
                WHERE organization_id = %s
            """
            exp_stats = self.execute_query(exp_query, (org_id,), fetch_one=True)
            stats['experiments'] = exp_stats if exp_stats else {}
            
            # Usage by user
            usage_query = """
                SELECT u.email, u.username, u.full_name, COUNT(e.id) as experiment_count
                FROM users u
                LEFT JOIN experiments e ON u.id = e.user_id
                WHERE u.organization_id = %s
                GROUP BY u.id, u.email, u.username, u.full_name
                ORDER BY experiment_count DESC
                LIMIT 10
            """
            usage_stats = self.execute_query(usage_query, (org_id,))
            stats['top_users'] = usage_stats if usage_stats else []
            
            return stats
            
        except Exception as e:
            self.handle_db_error("get_organization_stats", e, {'org_id': org_id})
            return {}

    def get_organization_usage(self, org_id: str) -> Dict[str, Any]:
        """
        Get current usage statistics for an organization
        
        Args:
            org_id: Organization ID
            
        Returns:
            Dict with usage metrics and limits
        """
        try:
            # Get current month's experiment count
            exp_query = """
                SELECT COUNT(*) as experiments 
                FROM experiments 
                WHERE organization_id = %s 
                AND created_at >= date_trunc('month', CURRENT_DATE)
            """
            experiments = self.execute_query(exp_query, (org_id,), fetch_one=True)
            experiment_count = experiments['experiments'] if experiments else 0
            
            # Get today's API calls
            api_query = """
                SELECT COALESCE(SUM(metric_value), 0) as api_calls 
                FROM usage_tracking 
                WHERE organization_id = %s 
                AND metric_type = 'api_call'
                AND timestamp >= CURRENT_DATE
            """
            api_calls = self.execute_query(api_query, (org_id,), fetch_one=True)
            api_call_count = api_calls['api_calls'] if api_calls else 0
            
            # Get total storage
            storage_query = """
                SELECT COALESCE(SUM(metric_value), 0) as storage_bytes 
                FROM usage_tracking 
                WHERE organization_id = %s 
                AND metric_type = 'storage'
            """
            storage = self.execute_query(storage_query, (org_id,), fetch_one=True)
            storage_bytes = storage['storage_bytes'] if storage else 0
            
            # Get limits
            limits_query = "SELECT * FROM usage_limits WHERE organization_id = %s"
            limits = self.execute_query(limits_query, (org_id,), fetch_one=True)
            
            if not limits:
                # Create default limits if none exist
                self.execute_query("INSERT INTO usage_limits (organization_id) VALUES (%s)", (org_id,), fetch_all=False)
                limits = self.execute_query(limits_query, (org_id,), fetch_one=True)
            
            return {
                'experiments': experiment_count,
                'experiments_limit': limits.get('experiments_monthly', 100) if limits else 100,
                'api_calls': api_call_count,
                'api_calls_limit': limits.get('api_calls_daily', 1000) if limits else 1000,
                'storage_mb': storage_bytes / (1024 * 1024),  # Convert to MB
                'storage_limit_mb': limits.get('storage_mb', 1000) if limits else 1000
            }
            
        except Exception as e:
            self.handle_db_error("get_organization_usage", e, {'org_id': org_id})
            return {
                'experiments': 0, 'experiments_limit': 100,
                'api_calls': 0, 'api_calls_limit': 1000,
                'storage_mb': 0, 'storage_limit_mb': 1000
            }

    def check_usage_limits(self, org_id: str, metric_type: str) -> bool:
        """
        Check if organization is within usage limits
        
        Args:
            org_id: Organization ID
            metric_type: Type of metric to check ('experiment', 'api_call', 'storage')
            
        Returns:
            bool: True if within limits
        """
        try:
            usage = self.get_organization_usage(org_id)
            
            if metric_type == 'experiment':
                return usage['experiments'] < usage['experiments_limit']
            elif metric_type == 'api_call':
                return usage['api_calls'] < usage['api_calls_limit']
            elif metric_type == 'storage':
                return usage['storage_mb'] < usage['storage_limit_mb']
            
            return True
            
        except Exception as e:
            self.log_error("check_usage_limits", e, {'org_id': org_id, 'metric_type': metric_type})
            return True  # Allow if check fails

    def get_usage_warnings(self, org_id: str) -> Dict[str, Any]:
        """
        Get usage warnings for metrics approaching limits (80% threshold)
        
        Args:
            org_id: Organization ID
            
        Returns:
            Dict with warning information
        """
        try:
            usage = self.get_organization_usage(org_id)
            warnings = {
                'has_warnings': False,
                'experiments': None,
                'api_calls': None,
                'storage': None
            }
            
            # Check experiments (80% threshold)
            if usage['experiments_limit'] > 0:
                experiments_pct = (usage['experiments'] / usage['experiments_limit']) * 100
                if experiments_pct >= 80:
                    warnings['experiments'] = {
                        'percentage': experiments_pct,
                        'current': usage['experiments'],
                        'limit': usage['experiments_limit'],
                        'message': f"You have used {experiments_pct:.0f}% of your monthly experiment limit ({usage['experiments']}/{usage['experiments_limit']})"
                    }
                    warnings['has_warnings'] = True
            
            # Check API calls (80% threshold)
            if usage['api_calls_limit'] > 0:
                api_pct = (usage['api_calls'] / usage['api_calls_limit']) * 100
                if api_pct >= 80:
                    warnings['api_calls'] = {
                        'percentage': api_pct,
                        'current': usage['api_calls'],
                        'limit': usage['api_calls_limit'],
                        'message': f"You have used {api_pct:.0f}% of your daily API call limit ({usage['api_calls']}/{usage['api_calls_limit']})"
                    }
                    warnings['has_warnings'] = True
            
            # Check storage (80% threshold)
            if usage['storage_limit_mb'] > 0:
                storage_pct = (usage['storage_mb'] / usage['storage_limit_mb']) * 100
                if storage_pct >= 80:
                    warnings['storage'] = {
                        'percentage': storage_pct,
                        'current': usage['storage_mb'],
                        'limit': usage['storage_limit_mb'],
                        'message': f"You have used {storage_pct:.0f}% of your storage limit ({usage['storage_mb']:.1f}/{usage['storage_limit_mb']} MB)"
                    }
                    warnings['has_warnings'] = True
            
            return warnings
            
        except Exception as e:
            self.handle_db_error("get_usage_warnings", e, {'org_id': org_id})
            return {'has_warnings': False, 'experiments': None, 'api_calls': None, 'storage': None}

    def track_usage(self, org_id: str, user_id: str, metric_type: str, value: int = 1) -> None:
        """
        Track usage metrics for an organization
        
        Args:
            org_id: Organization ID
            user_id: User ID
            metric_type: Type of metric ('experiment', 'api_call', 'storage')
            value: Metric value to add
        """
        try:
            query = """
                INSERT INTO usage_tracking (organization_id, user_id, metric_type, metric_value) 
                VALUES (%s, %s, %s, %s)
            """
            self.execute_query(query, (org_id, user_id, metric_type, value), fetch_all=False)
            
        except Exception as e:
            self.log_error("track_usage", e, {'org_id': org_id, 'user_id': user_id, 'metric_type': metric_type})

    def update_organization_limits(self, org_id: str, experiments_monthly: int = None, 
                                 api_calls_daily: int = None, storage_mb: int = None) -> bool:
        """
        Update organization usage limits
        
        Args:
            org_id: Organization ID
            experiments_monthly: New monthly experiment limit
            api_calls_daily: New daily API call limit
            storage_mb: New storage limit in MB
            
        Returns:
            bool: True if update succeeded
        """
        try:
            # Build dynamic update query
            update_fields = []
            params = []
            
            if experiments_monthly is not None:
                update_fields.append("experiments_monthly = %s")
                params.append(experiments_monthly)
            if api_calls_daily is not None:
                update_fields.append("api_calls_daily = %s")
                params.append(api_calls_daily)
            if storage_mb is not None:
                update_fields.append("storage_mb = %s")
                params.append(storage_mb)
            
            if not update_fields:
                return True  # Nothing to update
            
            params.append(org_id)
            query = f"UPDATE usage_limits SET {', '.join(update_fields)} WHERE organization_id = %s"
            
            self.execute_query(query, tuple(params), fetch_all=False)
            
            logger.info(f"Updated usage limits for organization {org_id}")
            return True
            
        except Exception as e:
            self.handle_db_error("update_organization_limits", e, {'org_id': org_id})
            return False

    def deactivate_organization(self, org_id: str) -> bool:
        """
        Deactivate an organization and all its users
        
        Args:
            org_id: Organization ID
            
        Returns:
            bool: True if deactivation succeeded
        """
        try:
            operations = [
                # Deactivate all users in the organization
                ("UPDATE users SET is_active = false WHERE organization_id = %s", (org_id,)),
                # Mark organization as inactive
                ("UPDATE organizations SET is_active = false WHERE id = %s", (org_id,))
            ]
            
            success = self.execute_transaction(operations)
            
            if success:
                logger.warning(f"Deactivated organization {org_id} and all its users")
            
            return success
            
        except Exception as e:
            self.handle_db_error("deactivate_organization", e, {'org_id': org_id})
            return False