"""
Base Repository Class
Provides common database functionality for all repositories
"""

import os
import uuid
import logging
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class BaseRepository:
    """Base repository class with common database functionality"""
    
    def __init__(self):
        """Initialize base repository"""
        self._connection_params = self._get_connection_params()

    def _get_connection_params(self) -> Dict[str, str]:
        """Get database connection parameters from environment"""
        # Check for DATABASE_URL first (for production deployments)
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            # Parse DATABASE_URL if provided
            return {'dsn': database_url}
        
        # Fallback to individual environment variables
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'llm_data_cleaning_production'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }

    def get_db_connection(self):
        """
        Get database connection using environment configuration
        
        Returns:
            psycopg2.connection: Database connection with RealDictCursor
            
        Raises:
            Exception: If connection fails
        """
        try:
            if 'dsn' in self._connection_params:
                # Use DATABASE_URL
                conn = psycopg2.connect(self._connection_params['dsn'], cursor_factory=psycopg2.extras.RealDictCursor)
            else:
                # Use individual parameters
                conn = psycopg2.connect(cursor_factory=psycopg2.extras.RealDictCursor, **self._connection_params)
            
            return conn
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise Exception(f"Database connection failed: {e}")

    @contextmanager
    def get_cursor(self, commit=True):
        """
        Context manager for database operations with automatic cleanup
        
        Args:
            commit: Whether to automatically commit on success
            
        Yields:
            psycopg2.cursor: Database cursor
            
        Example:
            with self.get_cursor() as cur:
                cur.execute("SELECT * FROM users")
                return cur.fetchall()
        """
        conn = None
        cur = None
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            yield cur
            
            if commit:
                conn.commit()
                
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def generate_id(self) -> str:
        """Generate UUID for new records"""
        return str(uuid.uuid4())

    def get_current_timestamp(self) -> datetime:
        """Get current UTC timestamp"""
        return datetime.now(timezone.utc)

    def execute_query(self, query: str, params: Optional[Tuple] = None, fetch_one: bool = False, fetch_all: bool = True) -> Any:
        """
        Execute a database query with error handling
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch_one: Return single result
            fetch_all: Return all results
            
        Returns:
            Query results or None
        """
        try:
            with self.get_cursor() as cur:
                cur.execute(query, params)
                
                if fetch_one:
                    result = cur.fetchone()
                    return dict(result) if result else None
                elif fetch_all:
                    results = cur.fetchall()
                    return [dict(row) for row in results] if results else []
                else:
                    return None  # For INSERT/UPDATE/DELETE without return
                    
        except Exception as e:
            logger.error(f"Query execution failed: {query[:100]}... Error: {e}")
            raise

    def execute_transaction(self, operations: List[Tuple[str, Tuple]]) -> bool:
        """
        Execute multiple operations in a single transaction
        
        Args:
            operations: List of (query, params) tuples
            
        Returns:
            bool: True if all operations succeeded
        """
        try:
            with self.get_cursor(commit=False) as cur:
                for query, params in operations:
                    cur.execute(query, params)
                
                # Manual commit since we set commit=False
                cur.connection.commit()
                return True
                
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            return False

    def log_error(self, operation: str, error: Exception, context: Optional[Dict] = None):
        """
        Standardized error logging for repositories
        
        Args:
            operation: Description of the operation that failed
            error: The exception that occurred
            context: Additional context information
        """
        error_msg = f"Repository operation failed - {operation}: {str(error)}"
        if context:
            error_msg += f" Context: {context}"
        logger.error(error_msg)

    def handle_db_error(self, operation: str, error: Exception, context: Optional[Dict] = None) -> None:
        """
        Handle database errors with logging and re-raising
        
        Args:
            operation: Description of the operation that failed
            error: The exception that occurred
            context: Additional context information
            
        Raises:
            Exception: Re-raises the original exception after logging
        """
        self.log_error(operation, error, context)
        
        # Re-raise with more context
        if isinstance(error, psycopg2.Error):
            raise Exception(f"Database error in {operation}: {str(error)}")
        else:
            raise error

    def validate_required_fields(self, data: Dict[str, Any], required_fields: List[str]) -> None:
        """
        Validate that required fields are present and not empty
        
        Args:
            data: Data dictionary to validate
            required_fields: List of required field names
            
        Raises:
            ValueError: If any required field is missing or empty
        """
        missing_fields = []
        for field in required_fields:
            if field not in data or not data[field]:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    def normalize_result(self, result: Any) -> Any:
        """
        Normalize database results for consistent return types
        
        Args:
            result: Raw database result
            
        Returns:
            Normalized result (dict for single, list for multiple)
        """
        if result is None:
            return None
        
        if isinstance(result, list):
            return [dict(row) if hasattr(row, '_asdict') else row for row in result]
        
        if hasattr(result, '_asdict'):
            return dict(result)
        
        return result

    def check_record_exists(self, table: str, conditions: Dict[str, Any]) -> bool:
        """
        Check if a record exists with given conditions
        
        Args:
            table: Table name
            conditions: Dictionary of field: value conditions
            
        Returns:
            bool: True if record exists
        """
        where_clause = " AND ".join([f"{field} = %s" for field in conditions.keys()])
        query = f"SELECT 1 FROM {table} WHERE {where_clause} LIMIT 1"
        
        result = self.execute_query(query, tuple(conditions.values()), fetch_one=True)
        return result is not None

    def get_record_count(self, table: str, conditions: Optional[Dict[str, Any]] = None) -> int:
        """
        Get count of records matching conditions
        
        Args:
            table: Table name
            conditions: Optional dictionary of field: value conditions
            
        Returns:
            int: Count of matching records
        """
        if conditions:
            where_clause = " AND ".join([f"{field} = %s" for field in conditions.keys()])
            query = f"SELECT COUNT(*) as count FROM {table} WHERE {where_clause}"
            params = tuple(conditions.values())
        else:
            query = f"SELECT COUNT(*) as count FROM {table}"
            params = None
        
        result = self.execute_query(query, params, fetch_one=True)
        return result['count'] if result else 0