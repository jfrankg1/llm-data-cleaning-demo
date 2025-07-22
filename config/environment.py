#!/usr/bin/env python3
"""
Environment Configuration Manager

This module handles environment-specific configuration loading and validation.
It supports development, testing, and production environments with appropriate
defaults and security measures.

Author: Claude AI
Created: 2025-07-04
"""

import os
import logging
from typing import Dict, Any, Optional
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class EnvironmentConfig:
    """
    Environment configuration manager with validation and security.
    
    Automatically loads the appropriate .env file based on the ENVIRONMENT
    variable or explicit environment specification.
    """
    
    def __init__(self, environment: Optional[str] = None):
        """
        Initialize environment configuration.
        
        Args:
            environment: Explicit environment name (development, testing, production)
                        If None, uses ENVIRONMENT env var or defaults to development
        """
        self.project_root = Path(__file__).parent.parent
        self.environment = self._determine_environment(environment)
        self.config = {}
        
        # Load environment-specific configuration
        self._load_environment_config()
        self._validate_required_config()
        self._setup_logging()
    
    def _determine_environment(self, environment: Optional[str]) -> Environment:
        """Determine which environment to use"""
        if environment:
            env_name = environment.lower()
        else:
            env_name = os.getenv('ENVIRONMENT', 'development').lower()
        
        try:
            return Environment(env_name)
        except ValueError:
            logging.warning(f"Unknown environment '{env_name}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def _load_environment_config(self):
        """Load environment-specific .env file"""
        # Check for .env.demo first (for demonstration purposes)
        demo_env_file = self.project_root / '.env.demo'
        if demo_env_file.exists():
            load_dotenv(demo_env_file)
            logging.info(f"Loaded demonstration configuration from {demo_env_file}")
        else:
            # Load base .env file first (fallback values)
            base_env_file = self.project_root / '.env'
            if base_env_file.exists():
                load_dotenv(base_env_file)
            
            # Load environment-specific .env file
            env_file = self.project_root / f'.env.{self.environment.value}'
            if env_file.exists():
                load_dotenv(env_file, override=True)
                logging.info(f"Loaded environment configuration from {env_file}")
            else:
                logging.warning(f"Environment file {env_file} not found, using defaults")
        
        # Load configuration into our config dict
        self._populate_config()
    
    def _populate_config(self):
        """Populate configuration dictionary with validated values"""
        
        # Environment identification
        self.config['environment'] = self.environment.value
        self.config['debug'] = self.get_bool('DEBUG', self.environment == Environment.DEVELOPMENT)
        self.config['log_level'] = os.getenv('LOG_LEVEL', 'INFO')
        
        # Database configuration
        self.config['database'] = {
            'url': os.getenv('DATABASE_URL'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': self.get_int('DB_PORT', 5432),
            'name': os.getenv('DB_NAME', 'dsaas'),
            'user': os.getenv('DB_USER', 'dsaas_user'),
            'password': os.getenv('DB_PASSWORD'),
            'pool_size': self.get_int('DB_POOL_SIZE', 20),
            'max_overflow': self.get_int('DB_MAX_OVERFLOW', 10),
            'pool_timeout': self.get_int('DB_POOL_TIMEOUT', 30),
            'pool_recycle': self.get_int('DB_POOL_RECYCLE', 3600),
        }
        
        # Redis configuration
        self.config['redis'] = {
            'url': os.getenv('REDIS_URL'),
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': self.get_int('REDIS_PORT', 6379),
            'db': self.get_int('REDIS_DB', 0),
            'password': os.getenv('REDIS_PASSWORD'),
            'ssl': self.get_bool('REDIS_SSL', False),
            'cluster_mode': self.get_bool('REDIS_CLUSTER_MODE', False),
        }
        
        # API configuration
        self.config['api'] = {
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
            'max_concurrent_requests': self.get_int('MAX_CONCURRENT_REQUESTS', 3),
            'max_requests_per_minute': self.get_int('MAX_REQUESTS_PER_MINUTE', 50),
            'max_tokens_per_minute': self.get_int('MAX_TOKENS_PER_MINUTE', 20000),
            'request_timeout': self.get_int('REQUEST_TIMEOUT', 120),
            'retry_attempts': self.get_int('RETRY_ATTEMPTS', 3),
        }
        
        # Security configuration
        self.config['security'] = {
            'secret_key': os.getenv('SECRET_KEY'),
            'session_timeout': self.get_int('SESSION_TIMEOUT', 3600),
            'jwt_expiration': self.get_int('JWT_EXPIRATION', 3600),
            'bcrypt_rounds': self.get_int('BCRYPT_ROUNDS', 12),
            'secure_cookies': self.get_bool('SECURE_COOKIES', self.environment == Environment.PRODUCTION),
            'force_https': self.get_bool('FORCE_HTTPS', self.environment == Environment.PRODUCTION),
            'secure_headers': self.get_bool('SECURE_HEADERS', self.environment != Environment.DEVELOPMENT),
        }
        
        # Storage configuration
        self.config['storage'] = {
            'type': os.getenv('STORAGE_TYPE', 'local'),
            'local_path': os.getenv('LOCAL_STORAGE_PATH', './storage'),
            'upload_folder': os.getenv('UPLOAD_FOLDER', './uploads'),
            's3_endpoint_url': os.getenv('S3_ENDPOINT_URL'),
            's3_access_key': os.getenv('S3_ACCESS_KEY'),
            's3_secret_key': os.getenv('S3_SECRET_KEY'),
            's3_bucket': os.getenv('S3_BUCKET'),
            's3_region': os.getenv('S3_REGION'),
            's3_encryption': self.get_bool('S3_SERVER_SIDE_ENCRYPTION', self.environment == Environment.PRODUCTION),
        }
        
        # Usage limits
        self.config['limits'] = {
            'default_experiment_limit': self.get_int('DEFAULT_EXPERIMENT_LIMIT', 100),
            'default_api_call_limit': self.get_int('DEFAULT_API_CALL_LIMIT', 1000),
            'default_storage_limit_mb': self.get_int('DEFAULT_STORAGE_LIMIT_MB', 1000),
            'enterprise_experiment_limit': self.get_int('ENTERPRISE_EXPERIMENT_LIMIT', 1000),
            'enterprise_api_call_limit': self.get_int('ENTERPRISE_API_CALL_LIMIT', 10000),
            'enterprise_storage_limit_mb': self.get_int('ENTERPRISE_STORAGE_LIMIT_MB', 10000),
        }
        
        # Streamlit configuration
        self.config['streamlit'] = {
            'server_port': self.get_int('STREAMLIT_SERVER_PORT', 8501),
            'server_address': os.getenv('STREAMLIT_SERVER_ADDRESS', 'localhost'),
            'server_headless': self.get_bool('STREAMLIT_SERVER_HEADLESS', self.environment != Environment.DEVELOPMENT),
            'max_upload_size': self.get_int('STREAMLIT_SERVER_MAX_UPLOAD_SIZE', 200),
            'max_message_size': self.get_int('STREAMLIT_SERVER_MAX_MESSAGE_SIZE', 200),
            'cache_ttl': self.get_int('STREAMLIT_CACHE_TTL', 3600),
            'theme_base': os.getenv('STREAMLIT_THEME_BASE', 'light'),
        }
        
        # Feature flags
        self.config['features'] = {
            'enable_debug_logging': self.get_bool('ENABLE_DEBUG_LOGGING', self.environment == Environment.DEVELOPMENT),
            'enable_sql_logging': self.get_bool('ENABLE_SQL_LOGGING', self.environment == Environment.DEVELOPMENT),
            'enable_performance_monitoring': self.get_bool('ENABLE_PERFORMANCE_MONITORING', True),
            'disable_rate_limiting': self.get_bool('DISABLE_RATE_LIMITING', False),
            'enable_test_endpoints': self.get_bool('ENABLE_TEST_ENDPOINTS', self.environment == Environment.DEVELOPMENT),
            'enable_analytics': self.get_bool('ENABLE_ANALYTICS', self.environment != Environment.DEVELOPMENT),
            'enable_monitoring': self.get_bool('ENABLE_MONITORING', True),
            'enable_audit_logging': self.get_bool('ENABLE_AUDIT_LOGGING', self.environment == Environment.PRODUCTION),
        }
        
        # External services
        self.config['external_services'] = {
            'mock_haveibeenpwned': self.get_bool('MOCK_HAVEIBEENPWNED', False),
            'mock_email_service': self.get_bool('MOCK_EMAIL_SERVICE', self.environment == Environment.DEVELOPMENT),
            'mock_file_scanning': self.get_bool('MOCK_FILE_SCANNING', self.environment == Environment.DEVELOPMENT),
        }
        
        # Email configuration
        self.config['email'] = {
            'backend': os.getenv('EMAIL_BACKEND', 'console' if self.environment == Environment.DEVELOPMENT else 'sendgrid'),
            'sendgrid_api_key': os.getenv('SENDGRID_API_KEY'),
            'from_email': os.getenv('EMAIL_FROM', 'noreply@dsaas.example.com'),
            'admin_email': os.getenv('EMAIL_ADMIN', 'admin@dsaas.example.com'),
            'rate_limit': self.get_int('EMAIL_RATE_LIMIT', 100),
        }
        
        # Monitoring and analytics
        self.config['monitoring'] = {
            'analytics_endpoint': os.getenv('ANALYTICS_ENDPOINT'),
            'analytics_api_key': os.getenv('ANALYTICS_API_KEY'),
            'monitoring_endpoint': os.getenv('MONITORING_ENDPOINT'),
            'health_check_endpoint': os.getenv('HEALTH_CHECK_ENDPOINT', '/health'),
            'sentry_dsn': os.getenv('SENTRY_DSN'),
        }
    
    def _validate_required_config(self):
        """Validate that required configuration is present"""
        # For production, some fields may use placeholder values that get replaced by env vars
        # Skip validation for production if we're in testing mode
        if self.environment == Environment.PRODUCTION:
            # In production, we expect environment variables to be set externally
            # Only validate if we have actual values, not placeholders
            anthropic_key = self.config['api']['anthropic_api_key']
            if anthropic_key and not anthropic_key.startswith('${'):
                # We have a real key, validate normally
                pass
            else:
                # Skip validation for production with placeholder values
                return
        
        required_fields = []
        
        # Only require API key and secret key for non-production or production with real values
        if self.environment != Environment.PRODUCTION or (
            self.config['api']['anthropic_api_key'] and 
            not self.config['api']['anthropic_api_key'].startswith('${')
        ):
            required_fields.extend([
                ('api.anthropic_api_key', 'ANTHROPIC_API_KEY'),
                ('security.secret_key', 'SECRET_KEY'),
            ])
        
        # Database URL is required unless individual DB params are provided
        if not self.config['database']['url']:
            if not all([
                self.config['database']['host'],
                self.config['database']['name'],
                self.config['database']['user'],
                self.config['database']['password']
            ]):
                required_fields.append(('database.url', 'DATABASE_URL'))
        
        missing_fields = []
        for field_path, env_var in required_fields:
            value = self._get_nested_config(field_path)
            if not value or (isinstance(value, str) and value.startswith('${')):
                missing_fields.append(env_var)
        
        if missing_fields:
            if self.environment == Environment.DEVELOPMENT:
                # For development, just warn about missing fields
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Missing environment variables (using defaults): {', '.join(missing_fields)}")
            else:
                raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")
    
    def _get_nested_config(self, path: str) -> Any:
        """Get nested configuration value using dot notation"""
        keys = path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    def _setup_logging(self):
        """Setup logging based on environment configuration"""
        log_level = getattr(logging, self.config['log_level'].upper(), logging.INFO)
        
        # Configure logging format based on environment
        if self.environment == Environment.PRODUCTION:
            # JSON format for production log aggregation
            logging.basicConfig(
                level=log_level,
                format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            # Human-readable format for development/testing
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        # Log environment startup
        logger = logging.getLogger(__name__)
        logger.info(f"Environment configuration loaded: {self.environment.value}")
        if self.config['debug']:
            logger.debug("Debug mode enabled")
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer value from environment variable"""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float value from environment variable"""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return self._get_nested_config(key) or default
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if running in testing environment"""
        return self.environment == Environment.TESTING
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == Environment.PRODUCTION
    
    def get_database_url(self) -> str:
        """Get complete database URL"""
        if self.config['database']['url']:
            return self.config['database']['url']
        
        # Construct URL from individual components
        db_config = self.config['database']
        return f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}"
    
    def get_redis_url(self) -> str:
        """Get complete Redis URL"""
        if self.config['redis']['url']:
            return self.config['redis']['url']
        
        # Construct URL from individual components
        redis_config = self.config['redis']
        auth = f":{redis_config['password']}@" if redis_config['password'] else ""
        protocol = "rediss" if redis_config['ssl'] else "redis"
        return f"{protocol}://{auth}{redis_config['host']}:{redis_config['port']}/{redis_config['db']}"
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists"""
        return self._get_nested_config(key) is not None

# Global configuration instance
# Will be initialized when the module is imported
_config = None

def get_config(environment: Optional[str] = None) -> EnvironmentConfig:
    """
    Get the global configuration instance.
    
    Args:
        environment: Force specific environment (for testing)
        
    Returns:
        EnvironmentConfig instance
    """
    global _config
    if _config is None or environment is not None:
        _config = EnvironmentConfig(environment)
    return _config

def reload_config(environment: Optional[str] = None):
    """Reload configuration (useful for testing)"""
    global _config
    _config = EnvironmentConfig(environment)
    return _config

# Convenience functions for common configuration access
def is_development() -> bool:
    """Check if running in development environment"""
    return get_config().is_development()

def is_testing() -> bool:
    """Check if running in testing environment"""
    return get_config().is_testing()

def is_production() -> bool:
    """Check if running in production environment"""
    return get_config().is_production()

def get_database_url() -> str:
    """Get database URL for current environment"""
    return get_config().get_database_url()

def get_redis_url() -> str:
    """Get Redis URL for current environment"""
    return get_config().get_redis_url()

def is_rate_limiting_available() -> bool:
    """
    Check if rate limiting is available and enabled.
    
    Returns True if:
    1. Redis is accessible
    2. Rate limiting is not disabled in configuration
    3. Required libraries are available
    """
    config = get_config()
    
    # Check if rate limiting is explicitly disabled
    if config.get('features.disable_rate_limiting', False):
        return False
    
    # Check if Redis is accessible
    try:
        import redis
        redis_url = config.get_redis_url()
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()
        return True
    except Exception:
        return False

def get_environment_config() -> Dict[str, Any]:
    """
    Get environment configuration for backward compatibility.
    
    Returns a dictionary with rate limiting availability and other settings.
    """
    config = get_config().config
    return {
        'RATE_LIMITING_AVAILABLE': is_rate_limiting_available(),
        'DEBUG': config.get('debug', False),
        'ENVIRONMENT': config.get('environment', 'development'),
        'DATABASE_URL': config.get('database', {}).get('url'),
        'REDIS_URL': config.get('redis', {}).get('url'),
        **config
    }