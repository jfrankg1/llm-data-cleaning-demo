#!/usr/bin/env python3
"""
Input Validation and Sanitization Module

This module provides comprehensive input validation and sanitization functions
to prevent security vulnerabilities such as injection attacks, XSS, and malware uploads.

Author: Claude AI
Created: 2025-07-04
"""

import re
import os
import hashlib
import bleach
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from urllib.parse import urlparse
import mimetypes

# Try to import magic, fall back to mimetypes if not available
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class FileType(Enum):
    """Allowed file types for upload"""
    CSV = "csv"
    PDF = "pdf"
    TXT = "txt"
    RTF = "rtf"
    DOC = "doc"
    DOCX = "docx"
    TSV = "tsv"

@dataclass
class FileValidation:
    """File validation result"""
    is_valid: bool
    file_type: Optional[FileType]
    mime_type: str
    size_mb: float
    errors: List[str]
    warnings: List[str]
    sanitized_filename: str

class InputValidator:
    """
    Comprehensive input validation and sanitization system.
    
    Provides protection against:
    - SQL injection
    - XSS attacks
    - Path traversal
    - File upload attacks
    - Invalid data formats
    """
    
    def __init__(self):
        """Initialize the input validator"""
        # File upload configuration
        self.max_file_size_mb = 200  # Maximum file size in MB
        self.allowed_extensions = {'.csv', '.pdf', '.txt', '.rtf', '.doc', '.docx', '.tsv'}
        self.allowed_mime_types = {
            'text/csv', 'text/plain', 'text/tab-separated-values',
            'application/pdf', 'application/rtf', 'text/rtf',
            'application/msword', 
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        
        # Input validation patterns
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.safe_string_pattern = re.compile(r'^[a-zA-Z0-9\s\-_.@]+$')
        self.safe_filename_pattern = re.compile(r'^[a-zA-Z0-9\-_.]+$')
        
        # XSS protection configuration (using bleach)
        self.allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'a', 'ul', 'ol', 'li']
        self.allowed_attributes = {'a': ['href', 'title']}
        self.allowed_protocols = ['http', 'https', 'mailto']
    
    # File Validation Methods
    
    def validate_file_upload(self, file_content: bytes, filename: str) -> FileValidation:
        """
        Comprehensive file upload validation.
        
        Args:
            file_content: Raw file content
            filename: Original filename
            
        Returns:
            FileValidation object with results
        """
        errors = []
        warnings = []
        
        # Sanitize filename first
        sanitized_filename = self.sanitize_filename(filename)
        
        # Check file size
        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            errors.append(f"File too large: {size_mb:.1f}MB (max: {self.max_file_size_mb}MB)")
        
        # Check file extension
        file_ext = Path(sanitized_filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            errors.append(f"File type not allowed: {file_ext}")
        
        # Check MIME type using python-magic or fallback to mimetypes
        detected_mime = 'unknown'
        try:
            if MAGIC_AVAILABLE:
                mime = magic.Magic(mime=True)
                detected_mime = mime.from_buffer(file_content)
            else:
                # Fallback to mimetypes based on extension
                detected_mime, _ = mimetypes.guess_type(sanitized_filename)
                if not detected_mime:
                    # Manual mapping for common extensions
                    mime_map = {
                        '.csv': 'text/csv',
                        '.txt': 'text/plain',
                        '.pdf': 'application/pdf',
                        '.doc': 'application/msword',
                        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        '.rtf': 'application/rtf'
                    }
                    detected_mime = mime_map.get(file_ext, 'application/octet-stream')
                warnings.append("File type detection using extension only (libmagic not available)")
            
            # Allow text/plain for CSV files (common misidentification)
            if file_ext == '.csv' and detected_mime == 'text/plain':
                detected_mime = 'text/csv'
            
            if detected_mime not in self.allowed_mime_types:
                errors.append(f"Detected file type not allowed: {detected_mime}")
        except Exception as e:
            logger.warning(f"Could not detect MIME type: {e}")
            warnings.append("Could not verify file type")
            detected_mime = 'unknown'
        
        # Check for malicious content patterns
        if self._contains_malicious_patterns(file_content):
            errors.append("File contains potentially malicious content")
        
        # Determine file type
        file_type = None
        if not errors:
            file_type = self._get_file_type(file_ext)
        
        return FileValidation(
            is_valid=len(errors) == 0,
            file_type=file_type,
            mime_type=detected_mime,
            size_mb=size_mb,
            errors=errors,
            warnings=warnings,
            sanitized_filename=sanitized_filename
        )
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and other attacks.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove any path components
        filename = os.path.basename(filename)
        
        # Remove Unicode control characters
        filename = ''.join(
            char for char in filename 
            if unicodedata.category(char)[0] != 'C'
        )
        
        # Replace spaces and special characters
        filename = re.sub(r'[^\w\s.-]', '_', filename)
        filename = re.sub(r'\s+', '_', filename)
        
        # Remove multiple dots (prevent extension spoofing)
        parts = filename.split('.')
        if len(parts) > 2:
            # Keep only the last extension
            filename = '.'.join(parts[:-1]).replace('.', '_') + '.' + parts[-1]
        
        # Limit length
        max_length = 255
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            filename = name[:max_length - len(ext)] + ext
        
        # Ensure filename is not empty
        if not filename or filename == '.':
            filename = 'unnamed_file'
        
        return filename
    
    def _contains_malicious_patterns(self, content: bytes) -> bool:
        """Check for malicious patterns in file content"""
        # Convert to string for pattern matching
        try:
            text_content = content.decode('utf-8', errors='ignore').lower()
        except:
            return False
        
        # Check for suspicious patterns
        malicious_patterns = [
            r'<script',      # JavaScript
            r'javascript:',  # JavaScript protocol
            r'vbscript:',    # VBScript protocol
            r'onclick=',     # Event handlers
            r'onerror=',
            r'onload=',
            r'eval\(',       # Eval function
            r'exec\(',       # Exec function
            r'__import__',   # Python import
            r'subprocess',   # System commands
            r'os\.system',   # System commands
            r'shell_exec',   # PHP shell
            r'passthru',     # PHP passthru
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, text_content):
                logger.warning(f"Malicious pattern detected: {pattern}")
                return True
        
        return False
    
    def _get_file_type(self, extension: str) -> Optional[FileType]:
        """Map file extension to FileType enum"""
        extension = extension.lower().strip('.')
        try:
            return FileType(extension)
        except ValueError:
            return None
    
    # String Validation Methods
    
    def validate_email(self, email: str) -> Tuple[bool, str]:
        """
        Validate and sanitize email address.
        
        Args:
            email: Email address to validate
            
        Returns:
            Tuple of (is_valid, sanitized_email)
        """
        if not email:
            return False, ""
        
        # Normalize email
        email = email.strip().lower()
        
        # Check pattern
        if not self.email_pattern.match(email):
            return False, email
        
        # Additional checks
        if len(email) > 254:  # RFC 5321
            return False, email
        
        # Check for dangerous characters even in valid emails
        if any(char in email for char in ['<', '>', '"', '\'', ';', '&']):
            return False, email
        
        return True, email
    
    def sanitize_string(self, text: str, max_length: int = 1000, 
                       allow_html: bool = False) -> str:
        """
        Sanitize string input to prevent XSS and injection attacks.
        
        Args:
            text: Text to sanitize
            max_length: Maximum allowed length
            allow_html: Whether to allow safe HTML tags
            
        Returns:
            Sanitized string
        """
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Limit length
        text = text[:max_length]
        
        if allow_html:
            # Use bleach to clean HTML
            text = bleach.clean(
                text,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                protocols=self.allowed_protocols,
                strip=True
            )
        else:
            # Escape all HTML
            text = bleach.clean(text, tags=[], strip=True)
        
        # Additional sanitization for dangerous URL schemes
        dangerous_schemes = ['javascript:', 'vbscript:', 'data:', 'file:']
        for scheme in dangerous_schemes:
            text = text.replace(scheme, scheme.replace(':', '_'))
            text = text.replace(scheme.upper(), scheme.upper().replace(':', '_'))
        
        return text.strip()
    
    def validate_organization_name(self, name: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate organization name.
        
        Args:
            name: Organization name
            
        Returns:
            Tuple of (is_valid, sanitized_name, error_message)
        """
        if not name:
            return False, "", "Organization name is required"
        
        # Check length before sanitization
        if len(name) > 100:
            return False, name[:100], "Organization name must not exceed 100 characters"
        
        # Sanitize
        sanitized = self.sanitize_string(name, max_length=100)
        
        # Check length after sanitization
        if len(sanitized) < 2:
            return False, sanitized, "Organization name must be at least 2 characters"
        
        # Check for suspicious patterns
        suspicious_patterns = ['<script', 'javascript:', 'select', 'drop', 'insert', 'update', 'delete', 'union']
        for pattern in suspicious_patterns:
            if pattern in sanitized.lower():
                return False, sanitized, "Organization name contains invalid characters"
        
        return True, sanitized, None
    
    def validate_experiment_name(self, name: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate experiment name.
        
        Args:
            name: Experiment name
            
        Returns:
            Tuple of (is_valid, sanitized_name, error_message)
        """
        if not name:
            return False, "", "Experiment name is required"
        
        # Sanitize
        sanitized = self.sanitize_string(name, max_length=200)
        
        # Check length
        if len(sanitized) < 3:
            return False, sanitized, "Experiment name must be at least 3 characters"
        
        if len(sanitized) > 200:
            return False, sanitized, "Experiment name must not exceed 200 characters"
        
        return True, sanitized, None
    
    # Numeric Validation Methods
    
    def validate_integer(self, value: Any, min_val: Optional[int] = None, 
                        max_val: Optional[int] = None) -> Tuple[bool, Optional[int]]:
        """
        Validate integer input.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Tuple of (is_valid, validated_value)
        """
        try:
            int_val = int(value)
            
            if min_val is not None and int_val < min_val:
                return False, None
            
            if max_val is not None and int_val > max_val:
                return False, None
            
            return True, int_val
        except (ValueError, TypeError):
            return False, None
    
    # Path Validation Methods
    
    def validate_path(self, path: str, base_path: str) -> Tuple[bool, str]:
        """
        Validate file path to prevent path traversal attacks.
        
        Args:
            path: Path to validate
            base_path: Base directory that paths must be within
            
        Returns:
            Tuple of (is_valid, resolved_path)
        """
        try:
            # Resolve to absolute path
            requested_path = Path(path).resolve()
            base = Path(base_path).resolve()
            
            # Check if path is within base directory
            if base not in requested_path.parents and base != requested_path:
                logger.warning(f"Path traversal attempt: {path}")
                return False, ""
            
            return True, str(requested_path)
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return False, ""
    
    # URL Validation Methods
    
    def validate_url(self, url: str, allowed_schemes: List[str] = None) -> Tuple[bool, str]:
        """
        Validate URL for safety.
        
        Args:
            url: URL to validate
            allowed_schemes: Allowed URL schemes (default: ['http', 'https'])
            
        Returns:
            Tuple of (is_valid, sanitized_url)
        """
        if not url:
            return False, ""
        
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']
        
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in allowed_schemes:
                return False, ""
            
            # Check for dangerous patterns
            dangerous_patterns = ['javascript:', 'data:', 'vbscript:', 'file:']
            for pattern in dangerous_patterns:
                if pattern in url.lower():
                    return False, ""
            
            # Reconstruct URL to ensure it's properly formatted
            sanitized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                sanitized += f"?{parsed.query}"
            
            return True, sanitized
        except Exception:
            return False, ""
    
    # Batch Validation Methods
    
    def validate_form_data(self, data: Dict[str, Any], schema: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Validate form data against a schema.
        
        Args:
            data: Form data to validate
            schema: Validation schema
            
        Returns:
            Dictionary with 'valid', 'errors', and 'sanitized' keys
        """
        errors = {}
        sanitized = {}
        
        for field, rules in schema.items():
            value = data.get(field)
            field_type = rules.get('type', 'string')
            required = rules.get('required', False)
            
            # Check required fields
            if required and not value:
                errors[field] = f"{field} is required"
                continue
            
            if not value:
                sanitized[field] = None
                continue
            
            # Validate based on type
            if field_type == 'email':
                is_valid, clean_value = self.validate_email(value)
                if not is_valid:
                    errors[field] = "Invalid email address"
                else:
                    sanitized[field] = clean_value
            
            elif field_type == 'string':
                max_length = rules.get('max_length', 1000)
                allow_html = rules.get('allow_html', False)
                sanitized[field] = self.sanitize_string(value, max_length, allow_html)
            
            elif field_type == 'integer':
                min_val = rules.get('min')
                max_val = rules.get('max')
                is_valid, int_value = self.validate_integer(value, min_val, max_val)
                if not is_valid:
                    errors[field] = f"Invalid integer value"
                else:
                    sanitized[field] = int_value
            
            elif field_type == 'organization':
                is_valid, clean_value, error = self.validate_organization_name(value)
                if not is_valid:
                    errors[field] = error
                else:
                    sanitized[field] = clean_value
            
            elif field_type == 'experiment':
                is_valid, clean_value, error = self.validate_experiment_name(value)
                if not is_valid:
                    errors[field] = error
                else:
                    sanitized[field] = clean_value
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'sanitized': sanitized
        }

# Global validator instance
validator = InputValidator()

# Convenience functions
def validate_file_upload(file_content: bytes, filename: str) -> FileValidation:
    """Validate file upload using global validator"""
    return validator.validate_file_upload(file_content, filename)

def sanitize_string(text: str, max_length: int = 1000, allow_html: bool = False) -> str:
    """Sanitize string using global validator"""
    return validator.sanitize_string(text, max_length, allow_html)

def validate_email(email: str) -> Tuple[bool, str]:
    """Validate email using global validator"""
    return validator.validate_email(email)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename using global validator"""
    return validator.sanitize_filename(filename)