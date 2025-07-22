"""
Time Series Alignment Module
AI-powered alignment and unification of time-based and event-based sensor logs
"""

import os
import json
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime

# Import existing components
from .csv_parser_enhanced import SmartCSVParser
from .unicode_processor import UnicodeProcessor, read_scientific_file
from .error_handler import (
    RobustErrorHandler, ProcessingError, ErrorCategory, ErrorSeverity,
    with_error_handling
)
from .filename_utils import FilenameSanitizer

# Set up logging
logger = logging.getLogger(__name__)

class TimeSeriesAlignmentProcessor:
    """Main processor for time-series alignment and unification"""
    
    def __init__(self):
        """Initialize the time-series alignment processor"""
        self.csv_parser = SmartCSVParser()
        self.unicode_processor = UnicodeProcessor()
        self.error_handler = RobustErrorHandler()
        self.filename_sanitizer = FilenameSanitizer()
        
        # Load the prompt template
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load the time-series alignment prompt from external file"""
        try:
            prompt_path = Path(__file__).parent.parent / "prompts" / "time_series_alignment.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load time-series alignment prompt: {e}")
            # Fallback to embedded prompt if file loading fails
            return self._get_fallback_prompt()
    
    def _get_fallback_prompt(self) -> str:
        """Fallback prompt if external file loading fails"""
        return """
        You are analyzing log files from IoT devices. Analyze the CSV files and determine:
        1. Whether each file is event-based or time-based
        2. If timestamp data is present
        3. The structure and indices of the data
        
        Output results in JSON format with 0-based indices.
        """
    
    @with_error_handling(category=ErrorCategory.DATA_PROCESSING, severity=ErrorSeverity.HIGH)
    def process_log_files(
        self,
        file_paths: List[str],
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process multiple log files for time-series alignment
        
        Args:
            file_paths: List of file paths to process
            organization_id: Organization ID for tracking
            user_id: User ID for tracking
            
        Returns:
            dict: Processing results with alignment analysis
        """
        results = {
            'files_processed': len(file_paths),
            'successful_files': [],
            'failed_files': [],
            'alignment_analysis': {},
            'unified_data': None,
            'processing_summary': {
                'total_files': len(file_paths),
                'processed_successfully': 0,
                'processing_errors': 0,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Process each file using hybrid approach
        processed_files = []
        for file_path in file_paths:
            try:
                # Try smart CSV processing first
                csv_result = self._process_standard_csv(file_path)
                
                if csv_result['success']:
                    # Smart CSV processing succeeded
                    processed_files.append(csv_result)
                    results['successful_files'].append(file_path)
                    results['processing_summary']['processed_successfully'] += 1
                    logger.info(f"Processed {file_path} using smart CSV method")
                else:
                    # Fall back to Claude analysis
                    logger.info(f"Falling back to Claude analysis for {file_path}")
                    file_content = self._read_and_parse_file(file_path)
                    processed_files.append({
                        'success': True,
                        'path': file_path,
                        'filename': os.path.basename(file_path),
                        'content': file_content,
                        'processing_method': 'claude_analysis',
                        'requires_claude': True
                    })
                    results['successful_files'].append(file_path)
                    results['processing_summary']['processed_successfully'] += 1
                
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                results['failed_files'].append({
                    'path': file_path,
                    'error': str(e)
                })
                results['processing_summary']['processing_errors'] += 1
        
        # Process and unify the data based on processing method
        if processed_files:
            try:
                # Separate files by processing method
                smart_csv_files = [f for f in processed_files if f.get('processing_method') == 'smart_csv']
                claude_files = [f for f in processed_files if f.get('requires_claude')]
                
                # Handle Claude analysis if needed
                if claude_files:
                    file_contents = [{'path': f['path'], 'filename': f['filename'], 'content': f['content']} 
                                   for f in claude_files]
                    analysis_results = self._analyze_file_structures(
                        file_contents, organization_id, user_id
                    )
                    results['alignment_analysis'] = analysis_results
                    
                    # Process Claude-analyzed files
                    for i, claude_file in enumerate(claude_files):
                        # Convert Claude analysis to dataframe
                        file_analysis = self._extract_file_analysis(analysis_results, claude_file['filename'])
                        if file_analysis:
                            logger.info(f"Converting Claude analysis to DataFrame for {claude_file['filename']}")
                            df = self._parse_file_based_on_analysis(claude_file['path'], file_analysis)
                            if df is not None:
                                # Add source file column
                                df['source_file'] = claude_file['filename']
                                
                                # Add to processed files with DataFrame
                                claude_file['dataframe'] = df
                                claude_file['processing_method'] = 'claude_analysis'
                                logger.info(f"Successfully converted Claude analysis for {claude_file['filename']} - {len(df)} rows, {len(df.columns)} columns")
                            else:
                                logger.warning(f"Failed to convert Claude analysis to DataFrame for {claude_file['filename']}")
                        else:
                            logger.warning(f"No valid analysis found for {claude_file['filename']}")
                
                # Create unified dataset using new logic
                if processed_files:
                    unified_data = self._create_unified_dataset(processed_files)
                    results['unified_data'] = unified_data
                    
                    # Add processing method summary
                    results['processing_methods'] = {
                        'smart_csv_count': len(smart_csv_files),
                        'claude_analysis_count': len(claude_files),
                        'total_files': len(processed_files)
                    }
                
            except Exception as e:
                logger.error(f"Failed to process and unify data: {e}")
                results['analysis_error'] = str(e)
        
        return results
    
    def _detect_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect timestamp column using intelligent heuristics"""
        # Check common timestamp column names first
        timestamp_candidates = [
            'timestamp', 'time', 'datetime', 'date_time', 'created_at', 
            'updated_at', 'event_time', 'log_time', 'recorded_at'
        ]
        
        for col in df.columns:
            if col.lower().strip() in timestamp_candidates:
                return col
        
        # Check for timestamp-like patterns in column names
        timestamp_patterns = ['time', 'date', 'stamp']
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(pattern in col_lower for pattern in timestamp_patterns):
                try:
                    # Verify it's actually datetime data
                    sample_data = df[col].dropna().head(5)
                    if len(sample_data) > 0:
                        pd.to_datetime(sample_data)
                        return col
                except:
                    continue
        
        # Check first few columns for timestamp-like data patterns
        for col in df.columns[:3]:  # Usually timestamp is in first few columns
            try:
                sample_data = df[col].dropna().head(5)
                if len(sample_data) > 0:
                    parsed_dates = pd.to_datetime(sample_data, errors='coerce')
                    if not parsed_dates.isna().all():  # At least some valid dates
                        return col
            except:
                continue
        
        return None
    
    def _is_standard_structure(self, df: pd.DataFrame, timestamp_col: Optional[str]) -> bool:
        """Validate if the DataFrame has a standard structure for fast processing"""
        # Additional checks for files that should use Claude fallback
        if self._requires_claude_analysis(df):
            return False
            
        return (
            timestamp_col is not None and           # Has identifiable timestamp column
            len(df.columns) >= 2 and               # Has data columns beyond timestamp
            len(df) > 0 and                        # Has data rows
            not df.columns.duplicated().any()      # No duplicate headers
        )
    
    def _requires_claude_analysis(self, df: pd.DataFrame) -> bool:
        """Check if file requires Claude analysis based on complexity markers"""
        # Check for multiple timestamp columns
        timestamp_candidates = ['time', 'timestamp', 'datetime', 'date']
        timestamp_columns = [col for col in df.columns 
                           if any(candidate in col.lower() for candidate in timestamp_candidates)]
        
        if len(timestamp_columns) > 1:
            logger.info("Multiple timestamp columns detected - requires Claude analysis")
            return True
        
        # Check for duplicate column names
        if df.columns.duplicated().any():
            logger.info("Duplicate column names detected - requires Claude analysis")
            return True
            
        # Check for embedded comments in data (look for rows starting with #)
        try:
            # Check if any cell in first few rows contains comment markers
            for i in range(min(5, len(df))):
                for col in df.columns:
                    cell_value = str(df.iloc[i, df.columns.get_loc(col)])
                    if cell_value.strip().startswith('#'):
                        logger.info("Embedded comments detected - requires Claude analysis")
                        return True
        except Exception:
            pass
        
        # Check for data quality issues that indicate complexity
        if self._has_data_quality_issues(df):
            return True
        
        # Check for specific complexity indicators that Smart CSV can't handle well
        return self._has_advanced_timestamp_complexity(df)
    
    def _has_data_quality_issues(self, df: pd.DataFrame) -> bool:
        """Check for data quality issues that indicate file complexity"""
        try:
            # Check for excessive missing data (>50% missing in any column)
            missing_threshold = 0.5
            for col in df.columns:
                if df[col].isna().sum() / len(df) > missing_threshold:
                    logger.info(f"Excessive missing data in column '{col}' - requires Claude analysis")
                    return True
            
            # Check for highly mixed data types in columns
            for col in df.columns:
                if df[col].dtype == 'object':  # String columns
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 5:
                        # Check for mixed numeric and text patterns
                        numeric_count = sum(1 for val in sample_values if str(val).replace('.', '').replace('-', '').isdigit())
                        text_count = len(sample_values) - numeric_count
                        
                        # If we have significant mixing of numeric and text, it's complex
                        if numeric_count > 0 and text_count > 0 and min(numeric_count, text_count) >= 2:
                            logger.info(f"Mixed data types in column '{col}' - requires Claude analysis")
                            return True
            
            # Check for unusual column naming patterns that suggest complexity
            suspicious_column_patterns = [
                'unnamed:', 'column_', 'field_', 'var_', 'x.', 'x_'
            ]
            suspicious_columns = [col for col in df.columns 
                                if any(pattern in str(col).lower() for pattern in suspicious_column_patterns)]
            
            if len(suspicious_columns) > 2:
                logger.info(f"Suspicious column names detected: {suspicious_columns} - requires Claude analysis")
                return True
            
            # Check for extremely wide files (>100 columns) - these are often complex
            if df.shape[1] > 100:
                logger.info(f"Extremely wide format with {df.shape[1]} columns - requires Claude analysis")
                return True
            
            # Check for files with very few data rows relative to columns (might be transposed or malformed)
            # Only flag if we have an extreme ratio AND very few rows (likely transposed data)
            if len(df) <= 2 and df.shape[1] > 10:
                logger.info(f"Likely transposed or malformed: {df.shape[1]} columns vs {len(df)} rows - requires Claude analysis")
                return True
                
            # Check for irregular row lengths (sign of malformed CSV)
            if hasattr(df, 'index') and len(df) > 5:
                # Sample a few rows to check for completeness
                sample_rows = df.head(5)
                for idx, row in sample_rows.iterrows():
                    non_null_count = row.notna().sum()
                    expected_count = len(df.columns)
                    
                    # If a row has significantly fewer values than columns, it's irregular
                    if non_null_count < expected_count * 0.5:
                        logger.info(f"Irregular row structure detected at row {idx} - requires Claude analysis")
                        return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error in data quality check: {e}")
            # If we can't analyze data quality, assume it's complex
            return True
    
    def _has_advanced_timestamp_complexity(self, df: pd.DataFrame) -> bool:
        """Check for advanced timestamp complexity requiring Claude analysis"""
        try:
            # Check for embedded timestamps in text fields
            for col in df.columns:
                if df[col].dtype == 'object':  # String columns
                    sample_values = df[col].dropna().head(3).astype(str)
                    for value in sample_values:
                        # Look for timestamps embedded in text descriptions
                        if ('T' in value and 'Z' in value) or ('at 20' in value and ':' in value):
                            logger.info("Embedded timestamps in text detected - requires Claude analysis")
                            return True
            
            # Check for epoch/numeric timestamp columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                # Check if column name suggests it's a timestamp
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['epoch', 'millis', 'unix', 'tick']):
                    logger.info(f"Epoch timestamp column '{col}' detected - requires Claude analysis")
                    return True
                
                # Check if numeric values look like epoch timestamps (around 10-13 digits)
                sample_values = df[col].dropna().head(3)
                for value in sample_values:
                    if isinstance(value, (int, float)) and 1000000000 <= value <= 9999999999999:
                        logger.info(f"Numeric timestamp values in '{col}' detected - requires Claude analysis")
                        return True
            
            # Check for non-standard datetime formats that might confuse pandas
            for col in df.columns:
                col_lower = col.lower()
                if any(time_keyword in col_lower for time_keyword in ['time', 'date', 'stamp']):
                    # Check sample values for non-standard formats
                    sample_values = df[col].dropna().head(3).astype(str)
                    for value in sample_values:
                        # Look for specific non-standard datetime patterns
                        if any(pattern in value for pattern in ['-MST', '-PST', '-EST', '-GMT']):
                            logger.info(f"Non-standard datetime format with timezone in '{col}' detected - requires Claude analysis")
                            return True
                        
                        # Check for unusual datetime patterns (custom formats that pandas can't parse)
                        if len(value) > 10 and any(c.isdigit() for c in value):
                            try:
                                # Try parsing with pandas - if it fails, needs Claude
                                pd.to_datetime(value)
                            except:
                                logger.info(f"Non-standard datetime format in '{col}' detected - requires Claude analysis")
                                return True
            
            # Check for encoding/special character issues that indicate complexity
            for col in df.columns:
                # Check column names for non-ASCII characters
                if any(ord(c) > 127 for c in col):
                    logger.info("Non-ASCII column names detected - requires Claude analysis")
                    return True
                
                # Check data values for special characters/units
                if df[col].dtype == 'object':
                    sample_values = df[col].dropna().head(3).astype(str)
                    for value in sample_values:
                        # Look for embedded units or special characters
                        if any(char in value for char in ['°', '§', 'µ', '±']) or \
                           any(unit in value.upper() for unit in ['HPA', 'L/MIN', 'KG/M3']):
                            logger.info("Embedded units or special characters detected - requires Claude analysis")
                            return True
                                
        except Exception as e:
            logger.debug(f"Error in advanced timestamp complexity check: {e}")
            
        return False
    
    def _has_duplicate_headers(self, file_path: str) -> bool:
        """Check if the CSV file has duplicate headers by reading the first line"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
            
            # Split by comma to get header names
            headers = [h.strip() for h in first_line.split(',')]
            
            # Check for duplicates
            return len(headers) != len(set(headers))
        except Exception:
            return False
    
    def _process_standard_csv(self, file_path: str) -> Dict[str, Any]:
        """Process standard CSV files using fast pandas-based approach"""
        try:
            # Check for duplicate headers in the raw file first
            if self._has_duplicate_headers(file_path):
                raise ValueError("Duplicate headers detected, fallback to Claude analysis needed")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Detect timestamp column
            timestamp_col = self._detect_timestamp_column(df)
            
            if not self._is_standard_structure(df, timestamp_col):
                raise ValueError("File structure is not standard, fallback to Claude analysis needed")
            
            # Standardize timestamp column name
            if timestamp_col != 'timestamp':
                df = df.rename(columns={timestamp_col: 'timestamp'})
            
            # Add source file tracking
            filename = os.path.basename(file_path)
            df['source_file'] = filename
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Successfully processed {filename} using fast CSV processing")
            
            return {
                'success': True,
                'dataframe': df,
                'processing_method': 'smart_csv',
                'original_timestamp_column': timestamp_col,
                'filename': filename
            }
            
        except Exception as e:
            logger.warning(f"Standard CSV processing failed for {file_path}: {e}")
            return {'success': False, 'error': f"File structure is not standard, fallback to Claude analysis needed: {str(e)}"}
    
    def _read_and_parse_file(self, file_path: str) -> str:
        """Read and parse a single log file"""
        try:
            # Use the scientific file reader for consistent encoding handling
            content = read_scientific_file(file_path)
            
            # Parse as CSV to validate structure
            df, structure = self.csv_parser.parse_file(file_path)
            
            # Return the raw content for Claude analysis
            return content
            
        except Exception as e:
            raise ProcessingError(
                f"Failed to read file {file_path}: {str(e)}",
                category=ErrorCategory.FILE_IO,
                severity=ErrorSeverity.HIGH,
                details={'file_path': file_path}
            )
    
    def _analyze_file_structures(
        self,
        file_contents: List[Dict[str, Any]],
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze the structure of log files using Claude AI
        
        Args:
            file_contents: List of file content dictionaries
            organization_id: Organization ID for tracking
            user_id: User ID for tracking
            
        Returns:
            dict: Analysis results for each file
        """
        # Import Claude service
        from services.claude_api_service import get_claude_service
        claude_service = get_claude_service()
        
        # Prepare content for Claude analysis
        formatted_content = self._format_files_for_analysis(file_contents)
        
        # Prepare the prompt with content substitution
        if '{{LOG_FILES}}' in self.prompt_template:
            prompt = self.prompt_template.replace('{{LOG_FILES}}', formatted_content)
        else:
            # If no placeholder, append content to prompt
            prompt = self.prompt_template + "\n\n" + formatted_content
        
        try:
            # Send to Claude for analysis
            response = claude_service.send_to_claude(
                content=formatted_content,
                prompt=prompt,
                model="claude-4-sonnet-20250514",
                max_tokens=8000,
                temperature=0.0,
                organization_id=organization_id,
                user_id=user_id
            )
            
            # Parse the JSON response
            analysis_results = self._parse_analysis_response(response)
            
            return analysis_results
            
        except Exception as e:
            error_msg = f"Claude AI analysis failed: {str(e)}"
            logger.error(error_msg)
            raise ProcessingError(
                f"Time-series analysis cannot proceed without Claude AI analysis. {error_msg}",
                category=ErrorCategory.API_ERROR,
                severity=ErrorSeverity.HIGH,
                details={'file_count': len(file_contents), 'error': str(e)}
            )
    
    def _format_files_for_analysis(self, file_contents: List[Dict[str, Any]]) -> str:
        """Format file contents for Claude analysis"""
        formatted_content = ""
        
        for i, file_info in enumerate(file_contents):
            formatted_content += f"File {i+1}: {file_info['filename']}\n"
            formatted_content += f"Content:\n{file_info['content']}\n"
            formatted_content += "=" * 50 + "\n\n"
        
        return formatted_content
    
    def _extract_file_analysis(self, analysis_results: Dict[str, Any], filename: str) -> Optional[Dict[str, Any]]:
        """Extract analysis results for a specific file"""
        try:
            # Handle different analysis result formats
            if isinstance(analysis_results, dict):
                # Check if the results contain file-specific analysis
                if filename in analysis_results:
                    return analysis_results[filename]
                
                # Check for single file analysis (direct format)
                if any(key in analysis_results for key in ['file_type', 'timestamp_present', 'timestamp_column_index']):
                    return analysis_results
                
                # Look for analysis in nested structure
                for key, value in analysis_results.items():
                    if isinstance(value, dict) and 'file_type' in value:
                        return value
            
            elif isinstance(analysis_results, list) and len(analysis_results) > 0:
                # Return first analysis result for now
                return analysis_results[0]
            
            logger.warning(f"Could not extract analysis for file {filename} from results: {analysis_results}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract file analysis for {filename}: {e}")
            return None
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse Claude's JSON response"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'<json_output>(.*?)</json_output>', response, re.DOTALL)
            if json_match:
                json_text = json_match.group(1).strip()
            else:
                # Try to find JSON in the response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_text = json_match.group()
                else:
                    raise ValueError("No JSON found in response")
            
            # Parse the JSON
            analysis_results = json.loads(json_text)
            
            return analysis_results
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise ProcessingError(
                f"Invalid JSON response from Claude: {str(e)}",
                category=ErrorCategory.DATA_PROCESSING,
                severity=ErrorSeverity.HIGH,
                details={'response_snippet': response[:500]}
            )
    
    def _unify_log_data(
        self,
        file_contents: List[Dict[str, Any]],
        analysis_results: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """
        Unify multiple log files into a single aligned dataset
        
        Args:
            file_contents: List of file content dictionaries
            analysis_results: Analysis results from Claude
            
        Returns:
            pd.DataFrame: Unified dataset or None if unification fails
        """
        try:
            unified_dfs = []
            
            # Debug: log the structure of analysis_results
            logger.info(f"Analysis results type: {type(analysis_results)}")
            logger.info(f"Analysis results: {analysis_results}")
            
            for file_info in file_contents:
                filename = file_info['filename']
                
                # Get analysis for this file
                file_analysis = None
                if isinstance(analysis_results, list):
                    # Find matching analysis result by filename or index
                    for i, result in enumerate(analysis_results):
                        if (result.get('filename') == filename or 
                            result.get('file_name') == filename):
                            file_analysis = result
                            break
                    
                    # If not found by filename, try by index (for array-based results)
                    if not file_analysis and len(analysis_results) > 0:
                        file_index = next((i for i, f in enumerate(file_contents) if f['filename'] == filename), None)
                        if file_index is not None and file_index < len(analysis_results):
                            file_analysis = analysis_results[file_index]
                            
                elif isinstance(analysis_results, dict):
                    # Try direct filename lookup
                    if filename in analysis_results:
                        file_analysis = analysis_results[filename]
                    else:
                        # Try without extension or with different variations
                        base_name = filename.replace('.csv', '')
                        for key in analysis_results.keys():
                            if base_name in str(key) or filename in str(key):
                                file_analysis = analysis_results[key]
                                break
                
                if not file_analysis:
                    error_msg = f"Claude AI analysis failed for file {filename}. Unable to determine file structure."
                    logger.error(error_msg)
                    raise ProcessingError(
                        f"Time-series analysis failed: {error_msg}",
                        category=ErrorCategory.API_ERROR,
                        severity=ErrorSeverity.HIGH,
                        details={'filename': filename, 'analysis_results': analysis_results}
                    )
                
                # Parse the file based on analysis
                df = self._parse_file_based_on_analysis(file_info['path'], file_analysis)
                
                if df is not None:
                    # Add source file column
                    df['source_file'] = filename
                    unified_dfs.append(df)
            
            # Combine all DataFrames
            if unified_dfs:
                unified_df = pd.concat(unified_dfs, ignore_index=True, sort=False)
                
                # Sort by timestamp if available
                timestamp_cols = [col for col in unified_df.columns if isinstance(col, str) and ('timestamp' in col.lower() or 'time' in col.lower())]
                if timestamp_cols:
                    try:
                        unified_df[timestamp_cols[0]] = pd.to_datetime(unified_df[timestamp_cols[0]])
                        unified_df = unified_df.sort_values(timestamp_cols[0])
                    except Exception as e:
                        logger.warning(f"Failed to sort by timestamp: {e}")
                
                return unified_df
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to unify log data: {e}")
            return None
    
    def _create_unified_dataset(self, processed_files: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create unified dataset from processed files using enhanced schema unification"""
        try:
            # Collect all DataFrames
            dataframes = []
            all_columns = set(['timestamp'])
            
            for file_data in processed_files:
                if file_data.get('processing_method') == 'smart_csv':
                    # Already have DataFrame from smart CSV processing
                    df = file_data['dataframe'].copy()
                    dataframes.append(df)
                    
                    # Collect unique column names (excluding timestamp and source_file)
                    cols = [col for col in df.columns if col not in ['timestamp', 'source_file']]
                    all_columns.update(cols)
                
                elif file_data.get('processing_method') == 'claude_analysis':
                    # Process Claude-analyzed DataFrame
                    if 'dataframe' in file_data:
                        df = file_data['dataframe'].copy()
                        dataframes.append(df)
                        
                        # Collect unique column names (excluding timestamp and source_file)
                        cols = [col for col in df.columns if col not in ['timestamp', 'source_file']]
                        all_columns.update(cols)
                        logger.info(f"Added Claude-analyzed file {file_data.get('filename')} to unified dataset")
                    else:
                        logger.warning(f"Claude-analyzed file {file_data.get('filename')} has no DataFrame - skipping")
                
                elif file_data.get('requires_claude'):
                    # Handle files that require Claude but haven't been processed yet
                    logger.warning(f"File {file_data.get('filename')} requires Claude analysis but wasn't processed - skipping")
                    continue
            
            if not dataframes:
                logger.warning("No DataFrames available for unification")
                return pd.DataFrame()
            
            # Create unified schema: timestamp (if exists) + all unique columns + source_file
            has_timestamp = 'timestamp' in all_columns
            if has_timestamp:
                unified_columns = ['timestamp'] + sorted(all_columns - {'timestamp'}) + ['source_file']
            else:
                unified_columns = sorted(all_columns) + ['source_file']
                logger.info("No timestamp column found - creating event-based unified schema")
            
            # Standardize each DataFrame to unified schema
            standardized_dfs = []
            for df in dataframes:
                # Add missing columns with appropriate NaN values
                for col in unified_columns:
                    if col not in df.columns:
                        # Use pandas NA for missing values
                        df[col] = pd.NA
                
                # Reorder columns to match unified schema
                df = df[unified_columns]
                standardized_dfs.append(df)
            
            # Concatenate all DataFrames (with explicit sort=False to avoid FutureWarning)
            unified_df = pd.concat(standardized_dfs, ignore_index=True, sort=False)
            
            # Ensure timestamp is datetime and sort (if timestamp column exists)
            if 'timestamp' in unified_df.columns:
                try:
                    unified_df['timestamp'] = pd.to_datetime(unified_df['timestamp'], errors='coerce')
                    # Sort by timestamp if we have valid timestamps
                    valid_timestamps = unified_df['timestamp'].notna()
                    if valid_timestamps.any():
                        unified_df = unified_df.sort_values('timestamp').reset_index(drop=True)
                        logger.info("Sorted unified dataset by timestamp")
                    else:
                        logger.warning("No valid timestamps found - dataset not sorted by time")
                except Exception as e:
                    logger.warning(f"Failed to process timestamps in unified dataset: {e}")
            else:
                logger.info("No timestamp column in unified dataset - event-based data")
            
            # Clean up data types
            unified_df = self._optimize_data_types(unified_df)
            
            logger.info(f"Created unified dataset with {len(unified_df)} rows and {len(unified_df.columns)} columns")
            logger.info(f"Columns: {list(unified_df.columns)}")
            
            return unified_df
            
        except Exception as e:
            logger.error(f"Failed to create unified dataset: {e}")
            raise ProcessingError(
                f"Failed to create unified time-series dataset: {str(e)}",
                category=ErrorCategory.DATA_PROCESSING,
                severity=ErrorSeverity.HIGH,
                details={'processed_files_count': len(processed_files)}
            )
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for better performance and consistency"""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            if col in ['timestamp', 'source_file']:
                continue  # Skip special columns
                
            # Try to convert to numeric if possible
            try:
                # First, try to convert to numeric
                numeric_series = pd.to_numeric(optimized_df[col], errors='coerce')
                
                # If most values are numeric, use numeric type
                non_null_count = optimized_df[col].notna().sum()
                numeric_count = numeric_series.notna().sum()
                
                if non_null_count > 0 and (numeric_count / non_null_count) > 0.7:
                    optimized_df[col] = numeric_series
                    
            except Exception:
                # Keep as object/string type
                pass
        
        return optimized_df
    
    def _parse_file_based_on_analysis(
        self,
        file_path: str,
        analysis: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """
        Parse a file based on Claude's structural analysis
        
        Args:
            file_path: Path to the file
            analysis: Analysis results from Claude
            
        Returns:
            pd.DataFrame: Parsed data or None if parsing fails
        """
        try:
            logger.info(f"Parsing file {file_path} with analysis: {analysis}")
            
            # Read the file directly first
            import pandas as pd
            try:
                # For complex files, we need to be more flexible with CSV reading
                # Try different approaches to handle problematic CSV formats
                df = None
                
                # For complex files, go straight to manual parsing
                # since pandas CSV reading often fails on complex scientific formats
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                    
                    # Parse lines as CSV manually
                    import csv
                    from io import StringIO
                    
                    csv_data = []
                    for line in lines:
                        try:
                            # Skip comment lines that start with #
                            if line.strip().startswith('#'):
                                csv_data.append([line.strip()])  # Keep as single column for structure
                                continue
                            
                            # Parse each line as CSV
                            reader = csv.reader(StringIO(line))
                            row = next(reader)
                            csv_data.append(row)
                        except Exception as line_e:
                            # For problematic lines, try to split by comma as fallback
                            try:
                                row = line.split(',')
                                csv_data.append(row)
                            except:
                                # Skip completely problematic lines
                                continue
                    
                    if csv_data:
                        # Find the maximum number of columns
                        max_cols = max(len(row) for row in csv_data)
                        # Pad shorter rows with empty strings
                        for row in csv_data:
                            while len(row) < max_cols:
                                row.append('')
                        
                        df = pd.DataFrame(csv_data)
                        logger.info(f"Read file manually: {df.shape[0]} rows, {df.shape[1]} columns")
                    else:
                        raise ValueError("No valid CSV data found")
                except Exception as manual_e:
                    logger.error(f"Manual parsing failed: {manual_e}")
                    return None
                
                if df is None or df.empty:
                    logger.error("Failed to read any data from file")
                    return None
                    
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                return None
            
            # Apply structural information from analysis
            data_start_row = analysis.get('data_start_row_index', 0)
            variable_names_row = analysis.get('variable_names_row_index', 0)
            
            # Extract column headers
            if variable_names_row < len(df):
                headers = df.iloc[variable_names_row].tolist()
                logger.info(f"Extracted headers from row {variable_names_row}: {headers}")
            else:
                # Generate default headers
                headers = [f'col_{i}' for i in range(df.shape[1])]
                logger.warning(f"Generated default headers: {headers}")
            
            # Extract data starting from the specified row
            if data_start_row < len(df):
                data_df = df.iloc[data_start_row:].copy()
                data_df.columns = headers
                data_df = data_df.reset_index(drop=True)
                logger.info(f"Extracted data from row {data_start_row}: {data_df.shape[0]} rows")
            else:
                logger.error(f"Data start row {data_start_row} is beyond file length {len(df)}")
                return None
            
            # Handle timestamp column
            timestamp_col_index = analysis.get('timestamp_column_index', 0)
            if timestamp_col_index is not None and timestamp_col_index < len(headers):
                timestamp_col_name = headers[timestamp_col_index]
                
                # Rename timestamp column to standard name
                if timestamp_col_name != 'timestamp':
                    data_df = data_df.rename(columns={timestamp_col_name: 'timestamp'})
                    logger.info(f"Renamed column '{timestamp_col_name}' to 'timestamp'")
                
                # Convert to datetime with error handling
                try:
                    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'], errors='coerce')
                    # Filter out rows with invalid timestamps
                    valid_timestamp_mask = data_df['timestamp'].notna()
                    if valid_timestamp_mask.sum() == 0:
                        logger.error("No valid timestamps found after conversion")
                        return None
                    elif valid_timestamp_mask.sum() < len(data_df):
                        logger.warning(f"Removed {len(data_df) - valid_timestamp_mask.sum()} rows with invalid timestamps")
                        data_df = data_df[valid_timestamp_mask].reset_index(drop=True)
                    
                    logger.info("Successfully converted timestamp column to datetime")
                except Exception as e:
                    logger.warning(f"Failed to convert timestamp column: {e}")
                    # If timestamp conversion fails, treat as event-based data
                    if 'timestamp' in data_df.columns:
                        data_df = data_df.drop(columns=['timestamp'])
                    logger.info("Treating as event-based data without timestamps")
            else:
                logger.warning(f"No valid timestamp column found at index {timestamp_col_index}")
            
            logger.info(f"Final parsed DataFrame: {data_df.shape[0]} rows, {data_df.shape[1]} columns")
            logger.info(f"Columns: {list(data_df.columns)}")
            
            return data_df
            
        except Exception as e:
            logger.error(f"Failed to parse file {file_path} based on analysis: {e}")
            return None

# Service factory function
def get_time_series_alignment_processor() -> TimeSeriesAlignmentProcessor:
    """Factory function to get time-series alignment processor instance"""
    return TimeSeriesAlignmentProcessor()