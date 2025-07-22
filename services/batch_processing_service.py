"""
Batch Processing Service - Optimized concurrent file processing
Extracted and enhanced from src/dsaas2_batch.py as part of Phase 2 refactoring
"""

import os
import time
import threading
import asyncio
import logging
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import pandas as pd

# Import services
from .claude_api_service import get_claude_service
from .data_processing_service import get_data_processing_service
from .validation_service import get_validation_service

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing operations"""
    max_concurrent_requests: int = 3
    max_requests_per_minute: int = 50
    max_tokens_per_minute: int = 20000
    request_timeout: int = 120
    retry_attempts: int = 3
    retry_delay: float = 1.0
    batch_size: int = 5
    progress_callback: Optional[Callable] = None
    
    @property
    def timeout(self) -> int:
        """Alias for request_timeout for backward compatibility"""
        return self.request_timeout

class RateLimiter:
    """Thread-safe rate limiter for API requests"""
    
    def __init__(self, max_requests_per_minute: int, max_tokens_per_minute: int):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.requests_count = 0
        self.tokens_count = 0
        self.window_start = time.time()
        self.lock = threading.Lock()
    
    def can_make_request(self, estimated_tokens: int = 1000) -> bool:
        """Check if we can make a request within rate limits"""
        with self.lock:
            current_time = time.time()
            # Reset window if a minute has passed
            if current_time - self.window_start >= 60:
                self.requests_count = 0
                self.tokens_count = 0
                self.window_start = current_time
            
            # Check if we can make the request
            return (self.requests_count < self.max_requests_per_minute and 
                    self.tokens_count + estimated_tokens < self.max_tokens_per_minute)
    
    def record_request(self, tokens_used: int = 1000):
        """Record a successful request"""
        with self.lock:
            self.requests_count += 1
            self.tokens_count += tokens_used
    
    def get_wait_time(self) -> float:
        """Get time to wait before next request"""
        with self.lock:
            current_time = time.time()
            time_in_window = current_time - self.window_start
            return max(0, 60 - time_in_window)

class BatchProcessingService:
    """Service for batch processing files with rate limiting and optimization"""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize batch processing service
        
        Args:
            config: Optional batch configuration. Uses defaults if None.
        """
        self.config = config or BatchConfig()
        self.rate_limiter = RateLimiter(
            self.config.max_requests_per_minute,
            self.config.max_tokens_per_minute
        )
        
        # Initialize service dependencies
        self.claude_service = get_claude_service()
        self.data_service = get_data_processing_service()
        self.validation_service = get_validation_service()

    def process_files_batch(
        self,
        files: List[Tuple[str, str]],
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Process multiple files in batches with rate limiting.
        
        Args:
            files: List of (file_path, category) tuples
            organization_id: Organization ID for usage tracking
            user_id: User ID for usage tracking
            progress_callback: Optional callback for progress updates (current, total, status)
            
        Returns:
            dict: Processing results with success/failure counts and data
        """
        total_files = len(files)
        processed_files = 0
        successful_results = {}
        failed_files = []
        
        logger.info(f"Starting batch processing of {total_files} files")
        
        # Process files in batches
        for batch_start in range(0, total_files, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, total_files)
            batch_files = files[batch_start:batch_end]
            
            # Process current batch
            batch_results = self._process_batch_concurrent(
                batch_files, organization_id, user_id
            )
            
            # Collect results
            for file_path, result in batch_results.items():
                if isinstance(result, Exception):
                    failed_files.append({
                        'file_path': file_path,
                        'error': str(result)
                    })
                else:
                    successful_results[file_path] = result
                
                processed_files += 1
                
                # Report progress
                if progress_callback:
                    status = f"Processed {os.path.basename(file_path)}"
                    progress_callback(processed_files, total_files, status)
        
        return {
            'total_files': total_files,
            'successful_count': len(successful_results),
            'failed_count': len(failed_files),
            'success_rate': len(successful_results) / total_files if total_files > 0 else 0,
            'results': successful_results,
            'failures': failed_files
        }

    def categorize_files_batch(
        self,
        files: List[str],
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Categorize multiple files using Claude API.
        
        Args:
            files: List of file paths to categorize
            organization_id: Organization ID for usage tracking
            user_id: User ID for usage tracking
            
        Returns:
            dict: Mapping of file paths to categories
        """
        # Prepare file content for Claude
        file_content_list = []
        for file_path in files:
            try:
                # Read file content (simplified version)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()[:2000]  # Limit content for categorization
                
                file_content_list.append({
                    'filename': os.path.basename(file_path),
                    'content': content,
                    'path': file_path
                })
            except Exception as e:
                logger.warning(f"Could not read file {file_path} for categorization: {e}")
                continue
        
        if not file_content_list:
            return {}
        
        # Format for Claude
        formatted_content = ""
        for i, file_info in enumerate(file_content_list):
            formatted_content += f"File {i+1}: {file_info['filename']}\n"
            formatted_content += f"Content: {file_info['content']}\n\n"
        
        # Get categorization from Claude
        try:
            response = self.claude_service.analyze_with_claude(
                content=formatted_content,
                analysis_type='categorize',
                organization_id=organization_id,
                user_id=user_id
            )
            
            # Parse response with robust JSON extraction
            import json
            import re
            
            # Try to extract JSON from response
            try:
                # First try direct JSON parsing
                categorization = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from text response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    categorization = json.loads(json_match.group())
                else:
                    # Fallback: parse text response manually
                    categorization = self._parse_categorization_response(response, file_content_list)
            
            # Map back to full file paths
            result = {}
            for file_info in file_content_list:
                filename = file_info['filename']
                if filename in categorization:
                    result[file_info['path']] = categorization[filename]
                else:
                    result[file_info['path']] = 'other'
            
            return result
            
        except Exception as e:
            logger.error(f"Batch categorization failed: {e}")
            # Return default categorization based on file extension
            return self._default_categorization(file_content_list)

    def categorize_files_extended(
        self,
        files: List[str],
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Categorize multiple files using extended categorization that includes log detection.
        
        Args:
            files: List of file paths to categorize
            organization_id: Organization ID for usage tracking
            user_id: User ID for usage tracking
            
        Returns:
            dict: Mapping of file paths to categories (data, map, protocol, log, other)
        """
        # Prepare file content for Claude
        file_content_list = []
        for file_path in files:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()[:10000]  # Limit content for categorization
                
                file_content_list.append({
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'content': content
                })
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                continue
        
        # Format content for Claude
        formatted_content = ""
        for i, file_info in enumerate(file_content_list):
            formatted_content += f"File {i+1}: {file_info['filename']}\n"
            formatted_content += f"Content: {file_info['content']}\n\n"
        
        # Get extended categorization from Claude
        try:
            response = self.claude_service.analyze_with_claude(
                content=formatted_content,
                analysis_type='categorize_extended',
                organization_id=organization_id,
                user_id=user_id
            )
            
            # Parse response with robust JSON extraction
            import json
            import re
            
            # Try to extract JSON from response
            try:
                # First try direct JSON parsing
                categorization = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from <answer> tags
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    categorization = json.loads(answer_match.group(1))
                else:
                    # Try to extract JSON from text response
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        categorization = json.loads(json_match.group())
                    else:
                        # Fallback: parse text response manually
                        categorization = self._parse_categorization_response(response, file_content_list)
            
            # Map back to full file paths
            result = {}
            for file_info in file_content_list:
                filename = file_info['filename']
                if filename in categorization:
                    result[file_info['path']] = categorization[filename]
                else:
                    result[file_info['path']] = 'other'
            
            return result
            
        except Exception as e:
            logger.error(f"Extended batch categorization failed: {e}")
            # Return default categorization based on file extension
            return self._default_categorization(file_content_list)

    def analyze_files_batch(
        self,
        file_categories: Dict[str, str],
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze files based on their categories and return processed data.
        
        Args:
            file_categories: Dictionary mapping file paths to categories
            organization_id: Organization ID for usage tracking
            user_id: User ID for usage tracking
            
        Returns:
            dict: Processed DataFrames organized by category
        """
        results = {
            'data': [],
            'map': [],
            'protocol': []
        }
        
        # Group files by category
        categorized_files = {}
        for file_path, category in file_categories.items():
            if category not in categorized_files:
                categorized_files[category] = []
            categorized_files[category].append(file_path)
        
        # Process each category
        for category, file_paths in categorized_files.items():
            if category in ['data', 'map', 'protocol']:
                for file_path in file_paths:
                    try:
                        # Analyze file with Claude
                        analysis_response = self.claude_service.analyze_with_claude(
                            content=self._read_file_for_analysis(file_path),
                            analysis_type=category,
                            organization_id=organization_id,
                            user_id=user_id
                        )
                        
                        # Parse Claude's response
                        import json
                        analysis = json.loads(analysis_response)
                        
                        # Process the analysis into DataFrame
                        df = self.data_service.process_file(file_path, analysis, category)
                        results[category].append(df)
                        
                    except Exception as e:
                        logger.error(f"Failed to analyze {file_path}: {e}")
                        continue
        
        # Combine DataFrames within each category
        combined_results = {}
        for category, dataframes in results.items():
            if dataframes:
                combined_results[category] = pd.concat(dataframes, ignore_index=True)
        
        return combined_results

    def _process_batch_concurrent(
        self,
        batch_files: List[Tuple[str, str]],
        organization_id: Optional[str],
        user_id: Optional[str]
    ) -> Dict[str, Union[pd.DataFrame, Exception]]:
        """Process a batch of files concurrently with rate limiting"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests) as executor:
            # Submit all tasks
            future_to_file = {}
            for file_path, category in batch_files:
                future = executor.submit(
                    self._process_single_file_with_rate_limit,
                    file_path, category, organization_id, user_id
                )
                future_to_file[future] = file_path
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=self.config.timeout)
                    results[file_path] = result
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    results[file_path] = e
        
        return results
    
    def _parse_categorization_response(self, response: str, file_content_list: List[dict]) -> Dict[str, str]:
        """Parse categorization response manually if JSON fails"""
        categorization = {}
        
        # Try to find file categorizations in text
        for file_info in file_content_list:
            filename = file_info['filename']
            # Look for patterns like "filename: category" or "filename" -> "category"
            patterns = [
                rf'"{filename}"\s*:\s*"([^"]+)"',
                rf"'{filename}'\s*:\s*'([^']+)'",
                rf'{filename}\s*:\s*([a-zA-Z]+)',
                rf'{filename}\s*->\s*([a-zA-Z]+)'
            ]
            
            found = False
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    category = match.group(1).lower()
                    if category in ['data', 'map', 'protocol', 'log', 'other']:
                        categorization[filename] = category
                        found = True
                        break
            
            if not found:
                # Default based on file extension
                if filename.endswith('.csv'):
                    categorization[filename] = 'data'
                elif filename.endswith(('.txt', '.pdf', '.doc', '.docx')):
                    categorization[filename] = 'protocol'
                else:
                    categorization[filename] = 'other'
        
        return categorization
    
    def _default_categorization(self, file_content_list: List[dict]) -> Dict[str, str]:
        """Provide default categorization based on file extensions and content hints"""
        result = {}
        
        for file_info in file_content_list:
            filename = file_info['filename']
            file_path = file_info['path']
            content = file_info.get('content', '').lower()
            
            # Check for log file indicators in filename
            log_indicators = ['log', 'sensor', 'monitor', 'equipment', 'instrument', 'event', 'activity']
            is_log_file = any(indicator in filename.lower() for indicator in log_indicators)
            
            # Check for log file indicators in content
            content_log_indicators = ['timestamp', 'datetime', 'time', 'temperature', 'pressure', 'status', 'event']
            has_log_content = any(indicator in content for indicator in content_log_indicators)
            
            # Simple heuristic based on file extension and content
            if filename.endswith('.csv'):
                if is_log_file or has_log_content:
                    result[file_path] = 'log'
                else:
                    result[file_path] = 'data'
            elif filename.endswith(('.txt', '.pdf', '.doc', '.docx', '.rtf')):
                if is_log_file:
                    result[file_path] = 'log'
                else:
                    result[file_path] = 'protocol'
            else:
                result[file_path] = 'other'
        
        return result

    def _process_single_file_with_rate_limit(
        self,
        file_path: str,
        category: str,
        organization_id: Optional[str],
        user_id: Optional[str]
    ) -> pd.DataFrame:
        """Process a single file with rate limiting"""
        # Wait for rate limit if necessary
        while not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.get_wait_time()
            if wait_time > 0:
                time.sleep(min(wait_time, 1.0))  # Sleep in small increments
        
        try:
            # Analyze file with Claude
            analysis_response = self.claude_service.analyze_with_claude(
                content=self._read_file_for_analysis(file_path),
                analysis_type=category,
                organization_id=organization_id,
                user_id=user_id
            )
            
            # Record the API request
            self.rate_limiter.record_request(1000)  # Estimate 1000 tokens
            
            # Parse and process the analysis
            import json
            analysis = json.loads(analysis_response)
            
            return self.data_service.process_file(file_path, analysis, category)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            raise

    def _read_file_for_analysis(self, file_path: str) -> str:
        """Read file content for Claude analysis"""
        try:
            # Use validation service to check file
            if not self.validation_service.validate_file(file_path):
                raise ValueError(f"File validation failed: {file_path}")
            
            # Read file content (simplified - could use file processing service)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get current batch processing statistics"""
        return {
            'rate_limiter': {
                'requests_this_minute': self.rate_limiter.requests_count,
                'tokens_this_minute': self.rate_limiter.tokens_count,
                'max_requests_per_minute': self.rate_limiter.max_requests_per_minute,
                'max_tokens_per_minute': self.rate_limiter.max_tokens_per_minute,
                'window_start': self.rate_limiter.window_start
            },
            'config': {
                'max_concurrent_requests': self.config.max_concurrent_requests,
                'batch_size': self.config.batch_size,
                'retry_attempts': self.config.retry_attempts
            }
        }

    def update_config(self, **kwargs) -> None:
        """Update batch processing configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Update rate limiter if relevant settings changed
        if 'max_requests_per_minute' in kwargs or 'max_tokens_per_minute' in kwargs:
            self.rate_limiter = RateLimiter(
                self.config.max_requests_per_minute,
                self.config.max_tokens_per_minute
            )


# Convenience functions for backward compatibility and global access
_batch_service_instance = None

def get_batch_processing_service(config: Optional[BatchConfig] = None) -> BatchProcessingService:
    """Get singleton instance of batch processing service"""
    global _batch_service_instance
    if _batch_service_instance is None:
        _batch_service_instance = BatchProcessingService(config)
    return _batch_service_instance

def batch_send_to_claude(
    files: List[Tuple[str, str]],
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function for backward compatibility"""
    service = get_batch_processing_service()
    return service.process_files_batch(files, organization_id, user_id)

def Claude_categorizer(
    files: List[str],
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, str]:
    """Convenience function for backward compatibility"""
    service = get_batch_processing_service()
    return service.categorize_files_batch(files, organization_id, user_id)

def batch_analyze_files(
    file_categories: Dict[str, str],
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """Convenience function for backward compatibility"""
    service = get_batch_processing_service()
    return service.analyze_files_batch(file_categories, organization_id, user_id)

# Global configuration for backward compatibility
batch_config = BatchConfig()

# Global rate limiter for backward compatibility
rate_limiter = RateLimiter(
    batch_config.max_requests_per_minute,
    batch_config.max_tokens_per_minute
)