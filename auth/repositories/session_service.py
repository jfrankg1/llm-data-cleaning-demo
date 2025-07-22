"""
Session Service
Handles all session-related database operations
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)

class SessionService(BaseRepository):
    """Service for session-related database operations"""
    
    def __init__(self):
        """Initialize session service"""
        super().__init__()

    def create_session(self, user_id: str, session_token: str, expires_at: datetime, 
                      user_agent: str = None, ip_address: str = None) -> bool:
        """
        Create a new user session
        
        Args:
            user_id: User ID
            session_token: Unique session token
            expires_at: Session expiration datetime
            user_agent: Optional user agent string
            ip_address: Optional client IP address
            
        Returns:
            bool: True if session creation succeeded
        """
        try:
            # Validate required fields
            self.validate_required_fields({
                'user_id': user_id,
                'session_token': session_token,
                'expires_at': expires_at
            }, ['user_id', 'session_token', 'expires_at'])
            
            query = """
                INSERT INTO user_sessions (user_id, session_token, expires_at, user_agent, ip_address)
                VALUES (%s, %s, %s, %s, %s)
            """
            params = (user_id, session_token, expires_at, user_agent, ip_address)
            
            self.execute_query(query, params, fetch_all=False)
            
            logger.info(f"Created session for user {user_id}")
            return True
            
        except Exception as e:
            self.handle_db_error("create_session", e, {'user_id': user_id})
            return False

    def get_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """
        Get session information by token
        
        Args:
            session_token: Session token
            
        Returns:
            Session information dict or None if not found/expired
        """
        try:
            query = """
                SELECT s.*, u.email, u.username, u.organization_id, u.is_admin, u.is_system_admin
                FROM user_sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = %s 
                AND s.expires_at > %s 
                AND s.is_active = true
                AND u.is_active = true
            """
            
            result = self.execute_query(query, (session_token, self.get_current_timestamp()), fetch_one=True)
            
            if result:
                # Update last activity
                self.update_session_activity(session_token)
                
            return result
            
        except Exception as e:
            self.handle_db_error("get_session", e, {'session_token': session_token[:20] + '...'})
            return None

    def update_session_activity(self, session_token: str) -> bool:
        """
        Update session's last activity timestamp
        
        Args:
            session_token: Session token
            
        Returns:
            bool: True if update succeeded
        """
        try:
            query = "UPDATE user_sessions SET last_activity = %s WHERE session_token = %s"
            self.execute_query(query, (self.get_current_timestamp(), session_token), fetch_all=False)
            
            return True
            
        except Exception as e:
            self.log_error("update_session_activity", e, {'session_token': session_token[:20] + '...'})
            return False

    def invalidate_session(self, session_token: str) -> bool:
        """
        Invalidate a specific session
        
        Args:
            session_token: Session token to invalidate
            
        Returns:
            bool: True if invalidation succeeded
        """
        try:
            query = "UPDATE user_sessions SET is_active = false WHERE session_token = %s"
            self.execute_query(query, (session_token,), fetch_all=False)
            
            logger.info(f"Invalidated session {session_token[:20]}...")
            return True
            
        except Exception as e:
            self.handle_db_error("invalidate_session", e, {'session_token': session_token[:20] + '...'})
            return False

    def invalidate_user_sessions(self, user_id: str) -> bool:
        """
        Invalidate all sessions for a user
        
        Args:
            user_id: User ID
            
        Returns:
            bool: True if invalidation succeeded
        """
        try:
            query = "UPDATE user_sessions SET is_active = false WHERE user_id = %s"
            self.execute_query(query, (user_id,), fetch_all=False)
            
            logger.info(f"Invalidated all sessions for user {user_id}")
            return True
            
        except Exception as e:
            self.handle_db_error("invalidate_user_sessions", e, {'user_id': user_id})
            return False

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions from the database
        
        Returns:
            int: Number of sessions cleaned up
        """
        try:
            # First count expired sessions
            count_query = "SELECT COUNT(*) as count FROM user_sessions WHERE expires_at <= %s OR is_active = false"
            count_result = self.execute_query(count_query, (self.get_current_timestamp(),), fetch_one=True)
            expired_count = count_result['count'] if count_result else 0
            
            # Delete expired sessions
            delete_query = "DELETE FROM user_sessions WHERE expires_at <= %s OR is_active = false"
            self.execute_query(delete_query, (self.get_current_timestamp(),), fetch_all=False)
            
            logger.info(f"Cleaned up {expired_count} expired sessions")
            return expired_count
            
        except Exception as e:
            self.handle_db_error("cleanup_expired_sessions", e)
            return 0

    def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all sessions for a user
        
        Args:
            user_id: User ID
            active_only: If True, only return active sessions
            
        Returns:
            List of session information dicts
        """
        try:
            where_conditions = ["user_id = %s"]
            params = [user_id]
            
            if active_only:
                where_conditions.append("is_active = true")
                where_conditions.append("expires_at > %s")
                params.append(self.get_current_timestamp())
            
            where_clause = " AND ".join(where_conditions)
            
            query = f"""
                SELECT session_token, created_at, expires_at, last_activity, 
                       user_agent, ip_address, is_active
                FROM user_sessions 
                WHERE {where_clause}
                ORDER BY last_activity DESC
            """
            
            return self.execute_query(query, tuple(params))
            
        except Exception as e:
            self.handle_db_error("get_user_sessions", e, {'user_id': user_id})
            return []

    def extend_session(self, session_token: str, extension_hours: int = 24) -> bool:
        """
        Extend session expiration time
        
        Args:
            session_token: Session token
            extension_hours: Number of hours to extend the session
            
        Returns:
            bool: True if extension succeeded
        """
        try:
            new_expiry = self.get_current_timestamp() + timedelta(hours=extension_hours)
            
            query = "UPDATE user_sessions SET expires_at = %s WHERE session_token = %s AND is_active = true"
            self.execute_query(query, (new_expiry, session_token), fetch_all=False)
            
            logger.info(f"Extended session {session_token[:20]}... by {extension_hours} hours")
            return True
            
        except Exception as e:
            self.handle_db_error("extend_session", e, {'session_token': session_token[:20] + '...'})
            return False

    def get_session_statistics(self, org_id: str = None) -> Dict[str, Any]:
        """
        Get session statistics for organization or globally
        
        Args:
            org_id: Optional organization ID to filter by
            
        Returns:
            Dict with session statistics
        """
        try:
            # Build query based on filters
            if org_id:
                join_clause = "JOIN users u ON s.user_id = u.id"
                where_clause = "WHERE u.organization_id = %s"
                params = (org_id,)
            else:
                join_clause = ""
                where_clause = ""
                params = ()
            
            query = f"""
                SELECT 
                    COUNT(*) as total_sessions,
                    COUNT(CASE WHEN s.is_active = true AND s.expires_at > NOW() THEN 1 END) as active_sessions,
                    COUNT(CASE WHEN s.created_at >= CURRENT_DATE THEN 1 END) as sessions_today,
                    COUNT(CASE WHEN s.created_at >= date_trunc('week', CURRENT_DATE) THEN 1 END) as sessions_this_week,
                    AVG(EXTRACT(EPOCH FROM (s.last_activity - s.created_at))) as avg_session_duration_seconds
                FROM user_sessions s
                {join_clause}
                {where_clause}
            """
            
            result = self.execute_query(query, params, fetch_one=True)
            
            # Convert duration to minutes
            if result and result['avg_session_duration_seconds']:
                result['avg_session_duration_minutes'] = result['avg_session_duration_seconds'] / 60
            
            return result if result else {}
            
        except Exception as e:
            self.handle_db_error("get_session_statistics", e, {'org_id': org_id})
            return {}

    def is_session_valid(self, session_token: str) -> bool:
        """
        Check if a session token is valid
        
        Args:
            session_token: Session token to validate
            
        Returns:
            bool: True if session is valid and active
        """
        try:
            session = self.get_session(session_token)
            return session is not None
            
        except Exception as e:
            self.log_error("is_session_valid", e, {'session_token': session_token[:20] + '...'})
            return False