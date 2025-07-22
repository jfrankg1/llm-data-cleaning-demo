#!/usr/bin/env python3
"""
Comprehensive Error Handling and Recovery System
Implements Phase 1 TODO item #12 for robust error management across the application
"""

import logging
import traceback
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
import pandas as pd
from functools import wraps

# Set up structured logging
logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Classification of error types for appropriate handling"""
    FILE_IO = "file_io"
    CSV_PARSING = "csv_parsing"
    API_ERROR = "api_error"
    DATA_PROCESSING = "data_processing"
    VALIDATION = "validation"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RESOURCE_LIMIT = "resource_limit"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels for prioritization"""
    CRITICAL = "critical"  # Cannot continue
    HIGH = "high"         # Major functionality affected
    MEDIUM = "medium"     # Partial functionality affected
    LOW = "low"          # Minor issues, can continue
    WARNING = "warning"   # Informational, no impact


class ProcessingError(Exception):
    """Custom exception with enhanced error tracking"""
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.recoverable = recoverable
        self.original_exception = original_exception
        self.timestamp = datetime.now()
        
        # Add traceback if original exception exists
        if original_exception:
            self.details['original_traceback'] = traceback.format_exc()


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies"""
    def can_handle(self, error: ProcessingError) -> bool:
        """Check if this strategy can handle the error"""
        raise NotImplementedError
    
    def recover(self, error: ProcessingError, context: Dict[str, Any]) -> Any:
        """Attempt to recover from the error"""
        raise NotImplementedError


class FileIORecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategies for file I/O errors"""
    
    def can_handle(self, error: ProcessingError) -> bool:
        return error.category == ErrorCategory.FILE_IO
    
    def recover(self, error: ProcessingError, context: Dict[str, Any]) -> Any:
        """Attempt file I/O recovery"""
        file_path = context.get('file_path', '')
        
        # Strategy 1: Try alternative encoding
        if 'encoding' in str(error):
            logger.info(f"Attempting encoding recovery for {file_path}")
            encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except:
                    continue
        
        # Strategy 2: Try binary mode
        if context.get('allow_binary', False):
            logger.info(f"Attempting binary read for {file_path}")
            try:
                with open(file_path, 'rb') as f:
                    return f.read()
            except:
                pass
        
        # Strategy 3: Skip corrupted parts
        if context.get('allow_partial', False):
            logger.info(f"Attempting partial read for {file_path}")
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    return f.read()
            except:
                pass
        
        raise error


class CSVParsingRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategies for CSV parsing errors"""
    
    def can_handle(self, error: ProcessingError) -> bool:
        return error.category == ErrorCategory.CSV_PARSING
    
    def recover(self, error: ProcessingError, context: Dict[str, Any]) -> pd.DataFrame:
        """Attempt CSV parsing recovery"""
        file_path = context.get('file_path', '')
        
        # Strategy 1: Return empty DataFrame with expected structure
        if context.get('allow_empty', True):
            logger.warning(f"Returning empty DataFrame for failed CSV: {file_path}")
            # Create standard 96-well plate structure
            return pd.DataFrame(
                [[None] * 12 for _ in range(8)],
                columns=[f'Col_{i+1}' for i in range(12)],
                index=[f'Row_{chr(65+i)}' for i in range(8)]
            )
        
        raise error


class APIErrorRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategies for API errors"""
    
    def can_handle(self, error: ProcessingError) -> bool:
        return error.category == ErrorCategory.API_ERROR
    
    def recover(self, error: ProcessingError, context: Dict[str, Any]) -> Any:
        """Attempt API error recovery"""
        
        # Strategy 1: Retry with exponential backoff
        if context.get('retry_count', 0) < 3:
            import time
            wait_time = 2 ** context.get('retry_count', 0)
            logger.info(f"Retrying API call after {wait_time} seconds")
            time.sleep(wait_time)
            # Return sentinel to indicate retry
            return {'retry': True, 'wait_time': wait_time}
        
        # Strategy 2: Use cached response if available
        if 'cached_response' in context:
            logger.info("Using cached API response")
            return context['cached_response']
        
        # Strategy 3: Return degraded response
        if context.get('allow_degraded', False):
            logger.warning("Returning degraded API response")
            return {
                'categorization': {'other': context.get('files', [])},
                'degraded': True,
                'error': str(error)
            }
        
        raise error


class RobustErrorHandler:
    """Main error handling and recovery system"""
    
    def __init__(self):
        self.recovery_strategies = [
            FileIORecoveryStrategy(),
            CSVParsingRecoveryStrategy(),
            APIErrorRecoveryStrategy()
        ]
        self.error_history: List[ProcessingError] = []
        self.partial_results: Dict[str, Any] = {}
        
    def classify_error(self, exception: Exception) -> ProcessingError:
        """Classify an exception into a ProcessingError with category and severity"""
        error_message = str(exception)
        error_type = type(exception).__name__
        
        # File I/O errors
        if isinstance(exception, (IOError, OSError, FileNotFoundError)):
            return ProcessingError(
                error_message,
                category=ErrorCategory.FILE_IO,
                severity=ErrorSeverity.HIGH,
                original_exception=exception
            )
        
        # CSV parsing errors
        if 'csv' in error_message.lower() or 'dataframe' in error_message.lower():
            return ProcessingError(
                error_message,
                category=ErrorCategory.CSV_PARSING,
                severity=ErrorSeverity.MEDIUM,
                original_exception=exception
            )
        
        # API errors
        if 'api' in error_message.lower() or 'claude' in error_message.lower():
            return ProcessingError(
                error_message,
                category=ErrorCategory.API_ERROR,
                severity=ErrorSeverity.HIGH,
                original_exception=exception
            )
        
        # Network errors
        if 'timeout' in error_message.lower() or 'connection' in error_message.lower():
            return ProcessingError(
                error_message,
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH,
                original_exception=exception
            )
        
        # Default classification
        return ProcessingError(
            error_message,
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.HIGH,
            original_exception=exception
        )
    
    def handle_error(
        self, 
        error: Union[Exception, ProcessingError], 
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Handle an error with recovery strategies
        
        Args:
            error: The error to handle
            context: Additional context for recovery strategies
            
        Returns:
            Recovery result if successful, None otherwise
        """
        # Ensure we have a ProcessingError
        if not isinstance(error, ProcessingError):
            error = self.classify_error(error)
        
        # Log the error
        self.log_error(error, context)
        
        # Store in history
        self.error_history.append(error)
        
        # Try recovery strategies if error is recoverable
        if error.recoverable:
            for strategy in self.recovery_strategies:
                if strategy.can_handle(error):
                    try:
                        logger.info(f"Attempting recovery with {strategy.__class__.__name__}")
                        result = strategy.recover(error, context or {})
                        logger.info("Recovery successful")
                        return result
                    except Exception as recovery_error:
                        logger.warning(f"Recovery failed: {recovery_error}")
                        continue
        
        # No recovery possible
        return None
    
    def log_error(self, error: ProcessingError, context: Optional[Dict[str, Any]] = None):
        """Log error with appropriate detail level"""
        log_data = {
            'timestamp': error.timestamp.isoformat(),
            'category': error.category.value,
            'severity': error.severity.value,
            'message': str(error),
            'details': error.details,
            'context': context or {}
        }
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error: {json.dumps(log_data, indent=2)}")
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error: {json.dumps(log_data, indent=2)}")
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error: {json.dumps(log_data, indent=2)}")
        else:
            logger.info(f"Low severity error: {json.dumps(log_data, indent=2)}")
    
    def store_partial_result(self, key: str, result: Any):
        """Store partial results for graceful degradation"""
        self.partial_results[key] = result
        logger.info(f"Stored partial result for key: {key}")
    
    def get_partial_results(self) -> Dict[str, Any]:
        """Get all partial results collected during processing"""
        return self.partial_results.copy()
    
    def get_user_friendly_message(self, error: ProcessingError) -> str:
        """Generate user-friendly error message"""
        messages = {
            ErrorCategory.FILE_IO: "There was an issue reading the file. Please check that the file exists and is accessible.",
            ErrorCategory.CSV_PARSING: "The CSV file could not be parsed. Please check the file format and try again.",
            ErrorCategory.API_ERROR: "There was an issue communicating with the AI service. Please try again later.",
            ErrorCategory.DATA_PROCESSING: "An error occurred while processing your data. Some results may be incomplete.",
            ErrorCategory.VALIDATION: "The input data did not meet the required format. Please check your data and try again.",
            ErrorCategory.NETWORK: "A network error occurred. Please check your connection and try again.",
            ErrorCategory.AUTHENTICATION: "Authentication failed. Please check your credentials.",
            ErrorCategory.RESOURCE_LIMIT: "Resource limits have been reached. Please try again later or contact support.",
            ErrorCategory.UNKNOWN: "An unexpected error occurred. Please try again or contact support if the issue persists."
        }
        
        base_message = messages.get(error.category, messages[ErrorCategory.UNKNOWN])
        
        # Add specific details if available
        if error.details.get('file_path'):
            base_message += f" File: {os.path.basename(error.details['file_path'])}"
        
        return base_message
    
    def generate_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        if not self.error_history:
            return {'status': 'success', 'errors': []}
        
        report = {
            'status': 'completed_with_errors',
            'total_errors': len(self.error_history),
            'errors_by_category': {},
            'errors_by_severity': {},
            'recoverable_errors': 0,
            'unrecoverable_errors': 0,
            'partial_results_available': len(self.partial_results) > 0,
            'error_details': []
        }
        
        for error in self.error_history:
            # Count by category
            cat = error.category.value
            report['errors_by_category'][cat] = report['errors_by_category'].get(cat, 0) + 1
            
            # Count by severity
            sev = error.severity.value
            report['errors_by_severity'][sev] = report['errors_by_severity'].get(sev, 0) + 1
            
            # Count recoverable
            if error.recoverable:
                report['recoverable_errors'] += 1
            else:
                report['unrecoverable_errors'] += 1
            
            # Add error detail
            report['error_details'].append({
                'timestamp': error.timestamp.isoformat(),
                'category': cat,
                'severity': sev,
                'message': self.get_user_friendly_message(error),
                'technical_message': str(error),
                'recoverable': error.recoverable
            })
        
        return report


def with_error_handling(
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.HIGH,
    allow_partial: bool = True
):
    """
    Decorator for adding comprehensive error handling to functions
    
    Usage:
        @with_error_handling(category=ErrorCategory.CSV_PARSING)
        def process_csv_file(file_path):
            # function implementation
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = RobustErrorHandler()
            context = {
                'function': func.__name__,
                'args': args,
                'kwargs': kwargs,
                'allow_partial': allow_partial
            }
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Create ProcessingError
                error = ProcessingError(
                    str(e),
                    category=category,
                    severity=severity,
                    details={'function': func.__name__},
                    original_exception=e
                )
                
                # Try recovery
                recovery_result = handler.handle_error(error, context)
                
                if recovery_result is not None:
                    return recovery_result
                elif allow_partial and handler.partial_results:
                    # Return partial results if available
                    logger.warning(f"Returning partial results for {func.__name__}")
                    return handler.get_partial_results()
                else:
                    # Re-raise with enhanced error
                    raise error
        
        return wrapper
    return decorator


# Utility functions for common error scenarios

def safe_file_read(file_path: str, encoding: str = 'utf-8') -> Optional[str]:
    """Safely read a file with multiple fallback strategies"""
    handler = RobustErrorHandler()
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        error = handler.classify_error(e)
        result = handler.handle_error(
            error,
            {'file_path': file_path, 'allow_partial': True, 'allow_binary': False}
        )
        return result


def safe_dataframe_operation(
    operation: Callable,
    df: pd.DataFrame,
    fallback_value: Any = None
) -> Any:
    """Safely perform DataFrame operations with fallback"""
    try:
        return operation(df)
    except Exception as e:
        logger.warning(f"DataFrame operation failed: {e}, returning fallback value")
        return fallback_value


def batch_process_with_recovery(
    items: List[Any],
    process_func: Callable,
    continue_on_error: bool = True
) -> Dict[str, Any]:
    """
    Process multiple items with error recovery and partial results
    
    Returns:
        Dict with 'successful', 'failed', and 'partial' results
    """
    handler = RobustErrorHandler()
    results = {
        'successful': [],
        'failed': [],
        'partial': []
    }
    
    for i, item in enumerate(items):
        try:
            result = process_func(item)
            results['successful'].append((item, result))
        except Exception as e:
            error = handler.classify_error(e)
            handler.handle_error(error, {'item': item, 'index': i})
            
            if continue_on_error:
                results['failed'].append((item, str(error)))
                # Check for partial results
                if handler.partial_results:
                    results['partial'].append((item, handler.get_partial_results()))
            else:
                # Generate report and raise
                report = handler.generate_error_report()
                raise ProcessingError(
                    f"Batch processing failed at item {i}: {str(e)}",
                    category=ErrorCategory.DATA_PROCESSING,
                    severity=ErrorSeverity.HIGH,
                    details={'report': report, 'results_so_far': results}
                )
    
    return results


# Global error handler instance
global_error_handler = RobustErrorHandler()