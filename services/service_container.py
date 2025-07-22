"""
Service Container
Dependency injection container for managing service instances
"""

from typing import Dict, Any, Optional
from services.auth_service import AuthService
from services.file_processing_service import FileProcessingService
from services.experiment_service import ExperimentService


class ServiceContainer:
    """Dependency injection container for services"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._initialized = False
    
    def initialize(self):
        """Initialize all services"""
        if self._initialized:
            return
        
        # Register singleton services
        self._singletons['auth_service'] = AuthService()
        self._singletons['file_processing_service'] = FileProcessingService()
        self._singletons['experiment_service'] = ExperimentService()
        
        self._initialized = True
    
    def get_auth_service(self) -> AuthService:
        """Get authentication service instance"""
        if not self._initialized:
            self.initialize()
        return self._singletons['auth_service']
    
    def get_file_processing_service(self) -> FileProcessingService:
        """Get file processing service instance"""
        if not self._initialized:
            self.initialize()
        return self._singletons['file_processing_service']
    
    def get_experiment_service(self) -> ExperimentService:
        """Get experiment service instance"""
        if not self._initialized:
            self.initialize()
        return self._singletons['experiment_service']
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """Get service by name"""
        if not self._initialized:
            self.initialize()
        return self._singletons.get(service_name)
    
    def register_service(self, service_name: str, service_instance: Any):
        """Register a new service instance"""
        self._singletons[service_name] = service_instance
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all services"""
        if not self._initialized:
            self.initialize()
        
        health = {}
        
        # Check auth service
        try:
            auth_service = self.get_auth_service()
            health['auth_service'] = True
        except Exception:
            health['auth_service'] = False
        
        # Check file processing service
        try:
            file_service = self.get_file_processing_service()
            health['file_processing_service'] = True
        except Exception:
            health['file_processing_service'] = False
        
        # Check experiment service
        try:
            exp_service = self.get_experiment_service()
            health['experiment_service'] = True
        except Exception:
            health['experiment_service'] = False
        
        return health


# Global service container instance
service_container = ServiceContainer()

# Convenience functions for easy access
def get_auth_service() -> AuthService:
    """Get authentication service"""
    return service_container.get_auth_service()

def get_file_processing_service() -> FileProcessingService:
    """Get file processing service"""
    return service_container.get_file_processing_service()

def get_experiment_service() -> ExperimentService:
    """Get experiment service"""
    return service_container.get_experiment_service()