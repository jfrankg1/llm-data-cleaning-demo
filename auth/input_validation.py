#!/usr/bin/env python3
"""
Input Validation and Sanitization Module

This module provides comprehensive input validation and sanitization
to protect against XSS, SQL injection, and other attacks.

Author: Claude AI
Created: 2025-07-07
"""

import re
import html
import urllib.parse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    sanitized_value: str
    errors: List[str]
    warnings: List[str]

class InputValidator:
    """
    Comprehensive input validation and sanitization system.
    
    Protects against XSS, SQL injection, and malicious input.
    """
    
    def __init__(self):
        """Initialize input validator"""
        # XSS patterns to detect and remove
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>.*?</iframe>',
            r'<object[^>]*>.*?</object>',
            r'<embed[^>]*>',
            r'<link[^>]*>',
            r'<meta[^>]*>',
            r'expression\s*\(',
            r'url\s*\(',
            r'@import',
        ]
        
        # SQL injection patterns
        self.sql_patterns = [
            r"'.*?;.*?--",
            r'".*?;.*?--',
            r"'.*?OR.*?=.*?'",
            r'".*?OR.*?=.*?"',
            r"'.*?\sOR\s.*?=",
            r'".*?\sOR\s.*?=',
            r"\sOR\s+\d+=\d+",
            r"'.*?UNION.*?SELECT",
            r'".*?UNION.*?SELECT',
            r"'.*?DROP\s+TABLE",
            r'".*?DROP\s+TABLE',
            r"'.*?INSERT\s+INTO",
            r'".*?INSERT\s+INTO',
            r"'.*?DELETE\s+FROM",
            r'".*?DELETE\s+FROM',
            r"'.*?UPDATE.*?SET",
            r'".*?UPDATE.*?SET',
            r"--\s*$",
            r"/\*.*?\*/",
            r";\s*--",
        ]
        
        # Dangerous characters for different contexts
        self.dangerous_chars = {
            'html': ['<', '>', '"', "'", '&'],
            'url': ['javascript:', 'data:', 'vbscript:'],
            'sql': ["'", '"', ';', '--', '/*', '*/'],
        }
    
    def sanitize_html(self, text: str) -> str:
        """
        Sanitize HTML content to prevent XSS attacks.
        
        Args:
            text: Input text that may contain HTML
            
        Returns:
            Sanitized text with HTML entities encoded
        """
        if not text:
            return text
            
        # HTML encode dangerous characters
        sanitized = html.escape(text, quote=True)
        
        # Remove dangerous patterns
        for pattern in self.xss_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized
    
    def validate_email(self, email: str) -> ValidationResult:
        """
        Validate email address with XSS protection.
        
        Args:
            email: Email address to validate
            
        Returns:
            ValidationResult with validation status and sanitized email
        """
        errors = []
        warnings = []
        
        if not email:
            return ValidationResult(False, "", ["Email is required"], [])
        
        # Sanitize HTML
        sanitized_email = self.sanitize_html(email.strip())
        
        # Check for XSS patterns
        if self._contains_xss(email):
            errors.append("Email contains invalid characters")
        
        # Basic email format validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, sanitized_email):
            errors.append("Please enter a valid email address")
        
        # Length check
        if len(sanitized_email) > 254:
            errors.append("Email address is too long (max 254 characters)")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, sanitized_email, errors, warnings)
    
    def validate_organization_name(self, name: str) -> ValidationResult:
        """
        Validate organization name with security checks.
        
        Args:
            name: Organization name to validate
            
        Returns:
            ValidationResult with validation status and sanitized name
        """
        errors = []
        warnings = []
        
        if not name:
            return ValidationResult(False, "", ["Organization name is required"], [])
        
        # Sanitize HTML
        sanitized_name = self.sanitize_html(name.strip())
        
        # Check for XSS patterns
        if self._contains_xss(name):
            errors.append("Organization name contains invalid characters")
        
        # Check for SQL injection patterns
        if self._contains_sql_injection(name):
            errors.append("Organization name contains invalid characters")
        
        # Length validation
        if len(sanitized_name) < 2:
            errors.append("Organization name must be at least 2 characters")
        elif len(sanitized_name) > 100:
            errors.append("Organization name must not exceed 100 characters")
        
        # Character validation (allow letters, numbers, spaces, common punctuation)
        allowed_pattern = r'^[a-zA-Z0-9\s\.\-_&,()]+$'
        if not re.match(allowed_pattern, sanitized_name):
            warnings.append("Organization name contains unusual characters")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, sanitized_name, errors, warnings)
    
    def validate_user_name(self, name: str) -> ValidationResult:
        """
        Validate user full name with security checks.
        
        Args:
            name: User full name to validate
            
        Returns:
            ValidationResult with validation status and sanitized name
        """
        errors = []
        warnings = []
        
        if not name:
            return ValidationResult(False, "", ["Full name is required"], [])
        
        # Sanitize HTML
        sanitized_name = self.sanitize_html(name.strip())
        
        # Check for XSS patterns
        if self._contains_xss(name):
            errors.append("Name contains invalid characters")
        
        # Length validation
        if len(sanitized_name) < 1:
            errors.append("Name is required")
        elif len(sanitized_name) > 100:
            errors.append("Name must not exceed 100 characters")
        
        # Character validation (allow letters, spaces, common name characters)
        allowed_pattern = r'^[a-zA-Z\s\.\-\']+$'
        if not re.match(allowed_pattern, sanitized_name):
            warnings.append("Name contains unusual characters")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, sanitized_name, errors, warnings)
    
    def validate_experiment_name(self, name: str) -> ValidationResult:
        """
        Validate experiment name with security checks.
        
        Args:
            name: Experiment name to validate
            
        Returns:
            ValidationResult with validation status and sanitized name
        """
        errors = []
        warnings = []
        
        if not name:
            return ValidationResult(False, "", ["Experiment name is required"], [])
        
        # Sanitize HTML
        sanitized_name = self.sanitize_html(name.strip())
        
        # Check for XSS patterns
        if self._contains_xss(name):
            errors.append("Experiment name contains invalid characters")
        
        # Check for SQL injection patterns
        if self._contains_sql_injection(name):
            errors.append("Experiment name contains invalid characters")
        
        # Length validation
        if len(sanitized_name) < 3:
            errors.append("Experiment name must be at least 3 characters")
        elif len(sanitized_name) > 200:
            errors.append("Experiment name must not exceed 200 characters")
        
        # Allow more flexible characters for experiment names
        # Letters, numbers, spaces, and common punctuation
        allowed_pattern = r'^[a-zA-Z0-9\s\.\-_#@!$%^&*()+=\[\]{}|;:,.<>?]+$'
        if not re.match(allowed_pattern, sanitized_name):
            warnings.append("Experiment name contains unusual characters")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, sanitized_name, errors, warnings)
    
    def validate_search_input(self, search_text: str) -> ValidationResult:
        """
        Validate search input for admin interfaces.
        
        Args:
            search_text: Search text to validate
            
        Returns:
            ValidationResult with validation status and sanitized search text
        """
        errors = []
        warnings = []
        
        if not search_text:
            return ValidationResult(True, "", [], [])
        
        # Sanitize HTML
        sanitized_text = self.sanitize_html(search_text.strip())
        
        # Check for XSS patterns
        if self._contains_xss(search_text):
            errors.append("Search contains invalid characters")
        
        # Check for SQL injection patterns
        if self._contains_sql_injection(search_text):
            errors.append("Search contains invalid characters")
        
        # Length validation
        if len(sanitized_text) > 100:
            errors.append("Search text must not exceed 100 characters")
        
        # Allow only safe search characters (letters, numbers, spaces, basic punctuation)
        allowed_pattern = r'^[a-zA-Z0-9\s\.\-_@]+$'
        if not re.match(allowed_pattern, sanitized_text):
            warnings.append("Search contains unusual characters")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, sanitized_text, errors, warnings)
    
    def validate_url_input(self, url: str) -> ValidationResult:
        """
        Validate URL input with security checks.
        
        Args:
            url: URL to validate
            
        Returns:
            ValidationResult with validation status and sanitized URL
        """
        errors = []
        warnings = []
        
        if not url:
            return ValidationResult(True, "", [], [])
        
        # Sanitize HTML
        sanitized_url = self.sanitize_html(url.strip())
        
        # Check for XSS patterns
        if self._contains_xss(url):
            errors.append("URL contains invalid characters")
        
        # Check for dangerous protocols
        url_lower = url.lower()
        dangerous_protocols = ['javascript:', 'data:', 'vbscript:', 'file:']
        for protocol in dangerous_protocols:
            if url_lower.startswith(protocol):
                errors.append("URL protocol not allowed")
                break
        
        # Length validation
        if len(sanitized_url) > 500:
            errors.append("URL must not exceed 500 characters")
        
        # Basic URL format validation (optional)
        if sanitized_url and not (sanitized_url.startswith('http://') or sanitized_url.startswith('https://')):
            warnings.append("URL should start with http:// or https://")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, sanitized_url, errors, warnings)
    
    def _contains_xss(self, text: str) -> bool:
        """Check if text contains XSS patterns"""
        text_lower = text.lower()
        for pattern in self.xss_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    def _contains_sql_injection(self, text: str) -> bool:
        """Check if text contains SQL injection patterns"""
        for pattern in self.sql_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

# Global validator instance
input_validator = InputValidator()

def validate_and_sanitize_input(input_type: str, value: str) -> ValidationResult:
    """
    Validate and sanitize input based on type.
    
    Args:
        input_type: Type of input ('email', 'organization', 'user_name', 'experiment', 'search', 'url')
        value: Value to validate
        
    Returns:
        ValidationResult with validation status and sanitized value
    """
    if input_type == 'email':
        return input_validator.validate_email(value)
    elif input_type == 'organization':
        return input_validator.validate_organization_name(value)
    elif input_type == 'user_name':
        return input_validator.validate_user_name(value)
    elif input_type == 'experiment':
        return input_validator.validate_experiment_name(value)
    elif input_type == 'search':
        return input_validator.validate_search_input(value)
    elif input_type == 'url':
        return input_validator.validate_url_input(value)
    else:
        # Generic HTML sanitization
        sanitized = input_validator.sanitize_html(value)
        return ValidationResult(True, sanitized, [], [])