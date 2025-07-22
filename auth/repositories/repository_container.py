"""
Repository Container
Dependency injection container for all repositories and services
"""

from typing import Optional

# Import all repositories and services
from .user_repository import UserRepository
from .organization_repository import OrganizationRepository
from .experiment_repository import ExperimentRepository
from .session_service import SessionService
from .usage_tracking_service import UsageTrackingService
from .system_admin_service import SystemAdminService

# Global container instances (singletons)
_user_repository: Optional[UserRepository] = None
_organization_repository: Optional[OrganizationRepository] = None
_experiment_repository: Optional[ExperimentRepository] = None
_session_service: Optional[SessionService] = None
_usage_tracking_service: Optional[UsageTrackingService] = None
_system_admin_service: Optional[SystemAdminService] = None

def get_user_repository() -> UserRepository:
    """
    Get or create UserRepository singleton instance
    
    Returns:
        UserRepository: User repository instance
    """
    global _user_repository
    if _user_repository is None:
        _user_repository = UserRepository()
    return _user_repository

def get_organization_repository() -> OrganizationRepository:
    """
    Get or create OrganizationRepository singleton instance
    
    Returns:
        OrganizationRepository: Organization repository instance
    """
    global _organization_repository
    if _organization_repository is None:
        _organization_repository = OrganizationRepository()
    return _organization_repository

def get_experiment_repository() -> ExperimentRepository:
    """
    Get or create ExperimentRepository singleton instance
    
    Returns:
        ExperimentRepository: Experiment repository instance
    """
    global _experiment_repository
    if _experiment_repository is None:
        _experiment_repository = ExperimentRepository()
    return _experiment_repository

def get_session_service() -> SessionService:
    """
    Get or create SessionService singleton instance
    
    Returns:
        SessionService: Session service instance
    """
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service

def get_usage_tracking_service() -> UsageTrackingService:
    """
    Get or create UsageTrackingService singleton instance
    
    Returns:
        UsageTrackingService: Usage tracking service instance
    """
    global _usage_tracking_service
    if _usage_tracking_service is None:
        _usage_tracking_service = UsageTrackingService()
    return _usage_tracking_service

def get_system_admin_service() -> SystemAdminService:
    """
    Get or create SystemAdminService singleton instance
    
    Returns:
        SystemAdminService: System admin service instance
    """
    global _system_admin_service
    if _system_admin_service is None:
        _system_admin_service = SystemAdminService()
    return _system_admin_service

def reset_container():
    """
    Reset all singleton instances (useful for testing)
    """
    global _user_repository, _organization_repository, _experiment_repository
    global _session_service, _usage_tracking_service, _system_admin_service
    
    _user_repository = None
    _organization_repository = None
    _experiment_repository = None
    _session_service = None
    _usage_tracking_service = None
    _system_admin_service = None

# Convenience function to get all repositories at once
def get_all_repositories():
    """
    Get all repository instances
    
    Returns:
        Tuple of all repository instances
    """
    return (
        get_user_repository(),
        get_organization_repository(),
        get_experiment_repository(),
        get_session_service(),
        get_usage_tracking_service(),
        get_system_admin_service()
    )