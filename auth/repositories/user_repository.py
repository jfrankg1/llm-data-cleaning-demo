"""
User Repository
Handles all user-related database operations
"""

import bcrypt
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)

class UserRepository(BaseRepository):
    """Repository for user-related database operations"""
    
    def __init__(self):
        """Initialize user repository"""
        super().__init__()

    def create_user(self, email: str, username: str, password: str, organization_id: str, is_admin: bool = False) -> str:
        """
        Create a new user with organization
        
        Args:
            email: User email address
            username: User display name (stored as full_name)
            password: User password (will be hashed)
            organization_id: Organization ID
            is_admin: Whether user should be organization admin
            
        Returns:
            str: User ID
            
        Raises:
            Exception: If user creation fails
        """
        try:
            # Validate required fields
            self.validate_required_fields({
                'email': email,
                'full_name': username,  # Map username to full_name for database
                'password': password,
                'organization_id': organization_id
            }, ['email', 'full_name', 'password', 'organization_id'])
            
            # Check if user already exists
            if self.check_record_exists('users', {'email': email}):
                raise Exception(f"User with email {email} already exists")
            
            # Generate user ID and hash password
            user_id = self.generate_id()
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            current_time = self.get_current_timestamp()
            
            # Insert user (use full_name column name)
            query = """
                INSERT INTO users (id, email, full_name, password_hash, organization_id, is_admin, created_at, last_login)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (user_id, email, username, password_hash, organization_id, is_admin, current_time, current_time)
            
            self.execute_query(query, params, fetch_all=False)
            
            # Store password in history
            self.store_password_history(user_id, password_hash)
            
            logger.info(f"Created user {user_id} with email {email}")
            return user_id
            
        except Exception as e:
            self.handle_db_error("create_user", e, {'email': email, 'organization_id': organization_id})

    def authenticate_user(self, email: str, password: str, ip_address: str = "127.0.0.1") -> Optional[Dict[str, Any]]:
        """
        Authenticate user and return user information
        
        Args:
            email: User email
            password: User password
            ip_address: Client IP address for lockout tracking
            
        Returns:
            Dict with user information if authentication succeeds, None otherwise
        """
        try:
            # Get user by email
            user = self.get_user_by_email(email)
            if not user:
                logger.warning(f"Authentication failed: User {email} not found")
                return None
            
            # Check if account is locked
            if self.is_account_locked(user['id']):
                logger.warning(f"Authentication failed: Account {email} is locked")
                return None
            
            # Verify password
            stored_hash = user['password_hash']
            if not bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                # Record failed login
                self.record_failed_login(user['id'], ip_address)
                logger.warning(f"Authentication failed: Invalid password for {email}")
                return None
            
            # Clear failed login attempts on successful login
            self.clear_failed_logins(user['id'])
            
            # Update last login
            self.update_last_login(user['id'])
            
            # Return user info (excluding password hash)
            user_info = {
                'id': user['id'],
                'email': user['email'],
                'username': user['full_name'],  # Map full_name to username for compatibility
                'full_name': user['full_name'],
                'organization_id': user['organization_id'],
                'is_admin': user['is_admin'],
                'is_system_admin': user.get('is_system_admin', False),
                'created_at': user['created_at'],
                'last_login': user['last_login']
            }
            
            logger.info(f"User {email} authenticated successfully")
            return user_info
            
        except Exception as e:
            self.handle_db_error("authenticate_user", e, {'email': email})
            return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user by ID
        
        Args:
            user_id: User ID
            
        Returns:
            User information dict or None
        """
        try:
            query = """
                SELECT u.*, o.name as organization_name
                FROM users u
                JOIN organizations o ON u.organization_id = o.id
                WHERE u.id = %s
            """
            return self.execute_query(query, (user_id,), fetch_one=True)
            
        except Exception as e:
            self.handle_db_error("get_user_by_id", e, {'user_id': user_id})
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email address
        
        Args:
            email: User email address
            
        Returns:
            User information dict or None
        """
        try:
            query = """
                SELECT u.*, o.name as organization_name
                FROM users u
                JOIN organizations o ON u.organization_id = o.id
                WHERE u.email = %s
            """
            return self.execute_query(query, (email,), fetch_one=True)
            
        except Exception as e:
            self.handle_db_error("get_user_by_email", e, {'email': email})
            return None

    def get_organization_users(self, organization_id: str) -> List[Dict[str, Any]]:
        """
        Get all users in an organization
        
        Args:
            organization_id: Organization ID
            
        Returns:
            List of user information dicts
        """
        try:
            query = """
                SELECT u.id, u.email, u.full_name, u.full_name as username, u.is_admin, u.is_active, u.created_at, u.last_login
                FROM users u
                WHERE u.organization_id = %s
                ORDER BY u.created_at DESC
            """
            return self.execute_query(query, (organization_id,))
            
        except Exception as e:
            self.handle_db_error("get_organization_users", e, {'organization_id': organization_id})
            return []

    def update_user_status(self, user_id: str, is_active: bool) -> bool:
        """
        Update user active status
        
        Args:
            user_id: User ID
            is_active: New active status
            
        Returns:
            bool: True if update succeeded
        """
        try:
            query = "UPDATE users SET is_active = %s WHERE id = %s"
            self.execute_query(query, (is_active, user_id), fetch_all=False)
            
            status = "activated" if is_active else "deactivated"
            logger.info(f"User {user_id} {status}")
            return True
            
        except Exception as e:
            self.handle_db_error("update_user_status", e, {'user_id': user_id, 'is_active': is_active})
            return False

    def update_user_admin_status(self, user_id: str, is_admin: bool) -> bool:
        """
        Update user admin status
        
        Args:
            user_id: User ID
            is_admin: New admin status
            
        Returns:
            bool: True if update succeeded
        """
        try:
            query = "UPDATE users SET is_admin = %s WHERE id = %s"
            self.execute_query(query, (is_admin, user_id), fetch_all=False)
            
            status = "granted" if is_admin else "revoked"
            logger.info(f"Admin privileges {status} for user {user_id}")
            return True
            
        except Exception as e:
            self.handle_db_error("update_user_admin_status", e, {'user_id': user_id, 'is_admin': is_admin})
            return False

    def update_user_system_admin_status(self, user_id: str, is_system_admin: bool) -> bool:
        """
        Update user system admin status
        
        Args:
            user_id: User ID
            is_system_admin: New system admin status
            
        Returns:
            bool: True if update succeeded
        """
        try:
            query = "UPDATE users SET is_system_admin = %s WHERE id = %s"
            self.execute_query(query, (is_system_admin, user_id), fetch_all=False)
            
            status = "granted" if is_system_admin else "revoked"
            logger.info(f"System admin privileges {status} for user {user_id}")
            return True
            
        except Exception as e:
            self.handle_db_error("update_user_system_admin_status", e, {'user_id': user_id, 'is_system_admin': is_system_admin})
            return False

    def get_all_users_system_admin(self) -> List[Dict[str, Any]]:
        """
        Get all users for system admin (across all organizations)
        
        Returns:
            List of all users with organization information
        """
        try:
            query = """
                SELECT u.id, u.email, u.username, u.is_admin, u.is_system_admin, u.is_active, 
                       u.created_at, u.last_login, o.name as organization_name, u.organization_id
                FROM users u
                JOIN organizations o ON u.organization_id = o.id
                ORDER BY u.created_at DESC
            """
            return self.execute_query(query)
            
        except Exception as e:
            self.handle_db_error("get_all_users_system_admin", e)
            return []

    def get_user_detailed_usage(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed usage information for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Dict with user usage details
        """
        try:
            query = """
                SELECT 
                    u.id, u.email, u.username,
                    COUNT(e.id) as experiment_count,
                    COUNT(CASE WHEN e.status = 'completed' THEN 1 END) as completed_experiments,
                    COUNT(CASE WHEN e.status = 'failed' THEN 1 END) as failed_experiments,
                    MAX(e.created_at) as last_experiment_date,
                    COALESCE(uu.api_calls, 0) as api_calls,
                    COALESCE(uu.storage_used, 0) as storage_used
                FROM users u
                LEFT JOIN experiments e ON u.id = e.user_id
                LEFT JOIN user_usage uu ON u.id = uu.user_id
                WHERE u.id = %s
                GROUP BY u.id, u.email, u.username, uu.api_calls, uu.storage_used
            """
            return self.execute_query(query, (user_id,), fetch_one=True)
            
        except Exception as e:
            self.handle_db_error("get_user_detailed_usage", e, {'user_id': user_id})
            return None

    # Password Security Methods
    
    def store_password_history(self, user_id: str, password_hash: str) -> None:
        """
        Store password hash in history for reuse prevention
        
        Args:
            user_id: User ID
            password_hash: Hashed password
        """
        try:
            operations = [
                # Store password hash
                ("INSERT INTO password_history (user_id, password_hash) VALUES (%s, %s)", (user_id, password_hash)),
                # Keep only last 5 passwords
                ("""DELETE FROM password_history 
                   WHERE user_id = %s 
                   AND id NOT IN (
                       SELECT id FROM password_history 
                       WHERE user_id = %s 
                       ORDER BY created_at DESC 
                       LIMIT 5
                   )""", (user_id, user_id))
            ]
            
            self.execute_transaction(operations)
            
        except Exception as e:
            self.handle_db_error("store_password_history", e, {'user_id': user_id})

    def check_password_reuse(self, user_id: str, password: str) -> bool:
        """
        Check if password has been used recently
        
        Args:
            user_id: User ID
            password: Plain text password to check
            
        Returns:
            bool: True if password was used before
        """
        try:
            query = "SELECT password_hash FROM password_history WHERE user_id = %s ORDER BY created_at DESC"
            password_hashes = self.execute_query(query, (user_id,))
            
            for row in password_hashes:
                if bcrypt.checkpw(password.encode('utf-8'), row['password_hash'].encode('utf-8')):
                    return True  # Password was used before
                    
            return False  # Password is new
            
        except Exception as e:
            self.log_error("check_password_reuse", e, {'user_id': user_id})
            return False  # Allow if check fails

    # Account Lockout Methods
    
    def record_failed_login(self, user_id: str, ip_address: str = None) -> None:
        """
        Record a failed login attempt for account lockout tracking
        
        Args:
            user_id: User ID
            ip_address: Client IP address
        """
        try:
            query = """
                INSERT INTO account_lockouts (user_id, ip_address, failed_attempts, last_attempt)
                VALUES (%s, %s, 1, NOW())
                ON CONFLICT (user_id) 
                DO UPDATE SET 
                    failed_attempts = account_lockouts.failed_attempts + 1,
                    last_attempt = NOW(),
                    is_locked = CASE 
                        WHEN account_lockouts.failed_attempts + 1 >= 5 THEN true 
                        ELSE false 
                    END,
                    locked_until = CASE 
                        WHEN account_lockouts.failed_attempts + 1 >= 5 THEN NOW() + INTERVAL '15 minutes' 
                        ELSE NULL 
                    END
            """
            self.execute_query(query, (user_id, ip_address), fetch_all=False)
            
        except Exception as e:
            self.handle_db_error("record_failed_login", e, {'user_id': user_id})

    def clear_failed_logins(self, user_id: str) -> None:
        """
        Clear failed login attempts after successful login
        
        Args:
            user_id: User ID
        """
        try:
            query = """
                UPDATE account_lockouts 
                SET failed_attempts = 0, is_locked = false, locked_until = NULL 
                WHERE user_id = %s
            """
            self.execute_query(query, (user_id,), fetch_all=False)
            
        except Exception as e:
            self.handle_db_error("clear_failed_logins", e, {'user_id': user_id})

    def is_account_locked(self, user_id: str) -> bool:
        """
        Check if account is currently locked
        
        Args:
            user_id: User ID
            
        Returns:
            bool: True if account is locked
        """
        try:
            query = """
                SELECT is_locked, locked_until 
                FROM account_lockouts 
                WHERE user_id = %s
            """
            result = self.execute_query(query, (user_id,), fetch_one=True)
            
            if not result:
                return False
            
            # Check if locked and lockout period hasn't expired
            if result['is_locked'] and result['locked_until']:
                return result['locked_until'] > self.get_current_timestamp()
            
            return False
            
        except Exception as e:
            self.log_error("is_account_locked", e, {'user_id': user_id})
            return False  # Default to not locked if check fails

    def update_last_login(self, user_id: str) -> None:
        """
        Update user's last login timestamp
        
        Args:
            user_id: User ID
        """
        try:
            query = "UPDATE users SET last_login = %s WHERE id = %s"
            self.execute_query(query, (self.get_current_timestamp(), user_id), fetch_all=False)
            
        except Exception as e:
            self.log_error("update_last_login", e, {'user_id': user_id})

    def create_user_with_organization(self, email: str, password: str, username: str, organization_name: str) -> Dict[str, Any]:
        """
        Create a new user and organization (for first-time registration)
        
        Args:
            email: User email address
            password: User password (will be hashed)
            username: User display name
            organization_name: Organization name
            
        Returns:
            Dict with user and organization information
            
        Raises:
            Exception: If user creation fails
        """
        try:
            # Validate required fields
            self.validate_required_fields({
                'email': email,
                'username': username,
                'password': password,
                'organization_name': organization_name
            }, ['email', 'username', 'password', 'organization_name'])
            
            # Check if user already exists
            if self.check_record_exists('users', {'email': email}):
                raise Exception(f"User with email {email} already exists")
            
            # Generate IDs and hash password
            user_id = self.generate_id()
            org_id = self.generate_id()
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            current_time = self.get_current_timestamp()
            
            # Create organization and user in transaction
            operations = [
                # Create organization
                ("INSERT INTO organizations (id, name) VALUES (%s, %s)", (org_id, organization_name)),
                # Set default usage limits
                ("INSERT INTO usage_limits (organization_id) VALUES (%s)", (org_id,)),
                # Create user as admin
                ("INSERT INTO users (id, email, full_name, password_hash, organization_id, is_admin, created_at, last_login) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", 
                 (user_id, email, username, password_hash, org_id, True, current_time, current_time))
            ]
            
            self.execute_transaction(operations)
            
            # Store password in history
            self.store_password_history(user_id, password_hash)
            
            logger.info(f"Created user {user_id} with organization {org_id}")
            
            return {
                'id': user_id,
                'email': email,
                'username': username,
                'organization_id': org_id,
                'organization_name': organization_name,
                'is_admin': True
            }
            
        except Exception as e:
            self.handle_db_error("create_user_with_organization", e, {'email': email, 'organization_name': organization_name})

    def get_user_experiments(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get user's experiments with pagination
        
        Args:
            user_id: User ID
            limit: Maximum number of experiments to return
            offset: Number of experiments to skip
            
        Returns:
            List of experiment information dicts
        """
        try:
            query = """
                SELECT * FROM experiments 
                WHERE user_id = %s 
                ORDER BY created_at DESC 
                LIMIT %s OFFSET %s
            """
            return self.execute_query(query, (user_id, limit, offset))
            
        except Exception as e:
            self.handle_db_error("get_user_experiments", e, {'user_id': user_id})
            return []

    def get_user_usage_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive usage summary for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Dict with usage statistics
        """
        try:
            query = """
                SELECT 
                    u.id, u.email, u.username,
                    COUNT(e.id) as total_experiments,
                    COUNT(CASE WHEN e.status = 'completed' THEN 1 END) as completed_experiments,
                    COUNT(CASE WHEN e.status = 'failed' THEN 1 END) as failed_experiments,
                    COUNT(CASE WHEN e.status = 'processing' THEN 1 END) as processing_experiments,
                    MAX(e.created_at) as last_experiment_date,
                    COUNT(CASE WHEN e.created_at >= date_trunc('month', CURRENT_DATE) THEN 1 END) as experiments_this_month,
                    COALESCE(SUM(CASE WHEN ut.metric_type = 'api_call' THEN ut.metric_value END), 0) as total_api_calls,
                    COALESCE(SUM(CASE WHEN ut.metric_type = 'storage' THEN ut.metric_value END), 0) as storage_used_bytes
                FROM users u
                LEFT JOIN experiments e ON u.id = e.user_id
                LEFT JOIN usage_tracking ut ON u.id = ut.user_id
                WHERE u.id = %s
                GROUP BY u.id, u.email, u.username
            """
            result = self.execute_query(query, (user_id,), fetch_one=True)
            
            if result:
                # Convert storage to MB
                result['storage_used_mb'] = result['storage_used_bytes'] / (1024 * 1024) if result['storage_used_bytes'] else 0
                
            return result
            
        except Exception as e:
            self.handle_db_error("get_user_usage_summary", e, {'user_id': user_id})
            return None