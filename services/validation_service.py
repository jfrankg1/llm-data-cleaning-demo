"""
Validation Service - File validation and discovery utilities
Extracted from src/dsaas2.py as part of Phase 2 refactoring
"""

import os
import glob
import logging
import unicodedata
from typing import List, Tuple, Optional

# Set up logging
logger = logging.getLogger(__name__)

class ValidationService:
    """Service for file validation, discovery, and filename sanitization"""
    
    def __init__(self):
        """Initialize validation service"""
        # Default supported file extensions
        self.default_extensions = ['csv', 'txt', 'pdf', 'docx', 'doc', 'rtf']
        
        # Common scientific file extensions
        self.scientific_extensions = [
            'csv', 'txt', 'tsv', 'tab',  # Data files
            'pdf', 'docx', 'doc', 'rtf',  # Documents
            'xlsx', 'xls',  # Spreadsheets
            'xml', 'json',  # Structured data
            'png', 'jpg', 'jpeg', 'tiff', 'bmp'  # Images
        ]

    def validate_file(self, file_path: str) -> bool:
        """
        Validate that a file exists and is readable.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            if not os.path.isfile(file_path):
                logger.error(f"File not found: {file_path}")
                return False
                
            # Try to open the file to check if it's readable
            with open(file_path, 'rb') as f:
                f.read(1)  # Try to read one byte
            return True
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return False

    def validate_api_key(self, api_key: Optional[str] = None) -> bool:
        """
        Validate that the API key is properly loaded.
        
        Args:
            api_key: Optional API key to validate. If None, uses environment variable.
            
        Returns:
            bool: True if API key is valid
            
        Raises:
            ValueError: If API key is not found
        """
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please ensure your .env file is properly configured.")
        return bool(api_key)

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename for cross-platform compatibility.
        
        Args:
            filename: Original filename to sanitize
            
        Returns:
            str: Sanitized filename safe for all platforms
        """
        # Normalize unicode characters (e.g., ë -> e)
        filename = unicodedata.normalize('NFD', filename)
        filename = ''.join(char for char in filename if unicodedata.category(char) != 'Mn')
        
        # Replace problematic characters with underscores
        filename = filename.replace(' ', '_')  # Spaces
        filename = filename.replace('/', '_')  # Forward slashes
        filename = filename.replace('\\', '_')  # Backslashes
        filename = filename.replace(':', '_')  # Colons
        filename = filename.replace('*', '_')  # Asterisks
        filename = filename.replace('?', '_')  # Question marks
        filename = filename.replace('"', '_')  # Double quotes
        filename = filename.replace('<', '_')  # Less than
        filename = filename.replace('>', '_')  # Greater than
        filename = filename.replace('|', '_')  # Pipes
        
        # Remove any double underscores and trailing underscores
        while '__' in filename:
            filename = filename.replace('__', '_')
        filename = filename.strip('_')
        
        return filename

    def discover_files(
        self, 
        directory: str, 
        extensions: Optional[List[str]] = None,
        include_scientific: bool = True,
        recursive: bool = False
    ) -> List[Tuple[str, str]]:
        """
        Discover files in a directory with specified extensions.
        
        Args:
            directory: Directory to search for files
            extensions: List of file extensions to include (e.g., ['csv', 'pdf', 'txt'])
                       If None, uses default extensions
            include_scientific: If True, includes common scientific file types
            recursive: If True, searches subdirectories recursively
            
        Returns:
            List of tuples containing (file_path, extension)
        """
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return []
        
        if extensions is None:
            extensions = self.scientific_extensions if include_scientific else self.default_extensions
        
        files = []
        
        for ext in extensions:
            if recursive:
                pattern = os.path.join(directory, '**', f"*.{ext}")
                file_paths = glob.glob(pattern, recursive=True)
            else:
                pattern = os.path.join(directory, f"*.{ext}")
                file_paths = glob.glob(pattern)
            
            for file_path in file_paths:
                if self.validate_file(file_path):
                    files.append((file_path, ext))
                    logger.info(f"Found file: {os.path.basename(file_path)}")
                else:
                    logger.warning(f"Skipping invalid file: {file_path}")
        
        # Sort files by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
        
        return files

    def validate_directory(self, directory: str, create_if_missing: bool = False) -> bool:
        """
        Validate that a directory exists and is accessible.
        
        Args:
            directory: Directory path to validate
            create_if_missing: If True, creates the directory if it doesn't exist
            
        Returns:
            bool: True if directory is valid or was created successfully
        """
        try:
            if os.path.isdir(directory):
                # Check if directory is writable
                test_file = os.path.join(directory, '.write_test')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    return True
                except Exception:
                    logger.error(f"Directory is not writable: {directory}")
                    return False
            
            elif create_if_missing:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
                return True
            
            else:
                logger.error(f"Directory not found: {directory}")
                return False
                
        except Exception as e:
            logger.error(f"Error validating directory {directory}: {str(e)}")
            return False

    def validate_file_extension(self, file_path: str, allowed_extensions: Optional[List[str]] = None) -> bool:
        """
        Validate that a file has an allowed extension.
        
        Args:
            file_path: Path to the file
            allowed_extensions: List of allowed extensions (without dots)
                               If None, uses default extensions
            
        Returns:
            bool: True if file extension is allowed
        """
        if allowed_extensions is None:
            allowed_extensions = self.default_extensions
        
        file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        return file_ext in [ext.lower() for ext in allowed_extensions]

    def get_file_info(self, file_path: str) -> dict:
        """
        Get comprehensive information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            dict: File information including size, modification time, extension, etc.
        """
        try:
            if not self.validate_file(file_path):
                return {'error': 'File is not valid or accessible'}
            
            stat = os.stat(file_path)
            file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            
            return {
                'path': file_path,
                'filename': os.path.basename(file_path),
                'extension': file_ext,
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified_time': stat.st_mtime,
                'is_readable': True,
                'is_supported_type': self.validate_file_extension(file_path),
                'sanitized_filename': self.sanitize_filename(os.path.basename(file_path))
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return {'error': str(e)}

    def batch_validate_files(self, file_paths: List[str]) -> dict:
        """
        Validate multiple files and return summary.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            dict: Validation summary with valid/invalid files and statistics
        """
        valid_files = []
        invalid_files = []
        total_size = 0
        supported_types = set()
        
        for file_path in file_paths:
            if self.validate_file(file_path):
                file_info = self.get_file_info(file_path)
                if 'error' not in file_info:
                    valid_files.append(file_info)
                    total_size += file_info['size_bytes']
                    supported_types.add(file_info['extension'])
                else:
                    invalid_files.append({'path': file_path, 'error': file_info['error']})
            else:
                invalid_files.append({'path': file_path, 'error': 'File validation failed'})
        
        return {
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'total_valid': len(valid_files),
            'total_invalid': len(invalid_files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'file_types_found': list(supported_types),
            'validation_success_rate': len(valid_files) / len(file_paths) if file_paths else 0
        }


# Convenience functions for backward compatibility and global access
_validation_service_instance = None

def get_validation_service() -> ValidationService:
    """Get singleton instance of validation service"""
    global _validation_service_instance
    if _validation_service_instance is None:
        _validation_service_instance = ValidationService()
    return _validation_service_instance

def validate_file(file_path: str) -> bool:
    """Convenience function for backward compatibility"""
    service = get_validation_service()
    return service.validate_file(file_path)

def validate_api_key(api_key: Optional[str] = None) -> bool:
    """Convenience function for backward compatibility"""
    service = get_validation_service()
    return service.validate_api_key(api_key)

def sanitize_filename(filename: str) -> str:
    """Convenience function for backward compatibility"""
    service = get_validation_service()
    return service.sanitize_filename(filename)

def discover_files(
    directory: str, 
    extensions: Optional[List[str]] = None
) -> List[Tuple[str, str]]:
    """Convenience function for backward compatibility"""
    service = get_validation_service()
    return service.discover_files(directory, extensions)