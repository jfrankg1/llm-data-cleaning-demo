"""
Data Processing Service - Plate data extraction and processing
Extracted from src/dsaas2.py as part of Phase 2 refactoring
"""

import pandas as pd
import re
import logging
from typing import Dict, Any, Optional, Union
from io import StringIO

# Import enhanced CSV parser
try:
    from src.csv_parser_enhanced import SmartCSVParser
    CSV_PARSER_AVAILABLE = True
except ImportError:
    CSV_PARSER_AVAILABLE = False

# Import error handling
try:
    from src.error_handler import (
        RobustErrorHandler, ProcessingError, ErrorCategory, ErrorSeverity
    )
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

class DataProcessingService:
    """Service for plate data processing and extraction"""
    
    def __init__(self):
        """Initialize data processing service"""
        if ERROR_HANDLING_AVAILABLE:
            self.error_handler = RobustErrorHandler()
        else:
            self.error_handler = None
        
        if CSV_PARSER_AVAILABLE:
            self.csv_parser = SmartCSVParser()
        else:
            self.csv_parser = None

    def extract_plate_block(
        self,
        csv_filepath: str,
        analysis: Dict[str, Any],
        block_type: str = 'data'
    ) -> pd.DataFrame:
        """
        Generic function to extract data from plate blocks.
        
        Args:
            csv_filepath: Path to the CSV file
            analysis: Claude's analysis containing indices
            block_type: Type of block to extract ('data' or 'map')
            
        Returns:
            DataFrame with columns [Plate ID, Well ID, raw data/mapping]
            
        Raises:
            ProcessingError: If CSV parsing fails or data extraction fails
        """
        # Parse CSV file with fallback strategy
        df = self._parse_csv_with_fallback(csv_filepath)
        
        # Get block indices from Claude's analysis
        indices = self._get_block_indices(analysis, block_type)
        start_row, end_row, start_col, end_col = indices
        
        # Apply corrections for common Claude detection issues
        end_row = self._correct_end_row(df, start_row, end_row, start_col, end_col, block_type)
        
        # Extract the data block
        block = df.iloc[start_row:end_row+1, start_col:end_col+1]
        
        # Get plate ID from analysis
        plate_id = analysis['metadata'].get('plate_id', {}).get('value', 'Unknown')
        
        # Convert block to well-based format
        return self._format_block_data(block, plate_id, block_type)

    def process_file(
        self, 
        file_path: str, 
        analysis: Dict[str, Any],
        file_type: str = 'auto'
    ) -> pd.DataFrame:
        """
        Process a file based on Claude's analysis.
        
        Args:
            file_path: Path to the file to process
            analysis: Claude's analysis results
            file_type: Type of file ('data', 'map', 'protocol', or 'auto')
            
        Returns:
            DataFrame with processed data
            
        Raises:
            ValueError: If file type cannot be determined or is unsupported
        """
        try:
            # Auto-detect file type from analysis if not specified
            if file_type == 'auto':
                file_type = self._detect_file_type(analysis)
            
            # Process based on file type
            if file_type == 'protocol':
                return self._process_protocol_file(analysis)
            elif file_type in ['data', 'map']:
                df = self.extract_plate_block(file_path, analysis, file_type)
                # Ensure Plate ID is string type
                df['Plate ID'] = df['Plate ID'].astype(str)
                return df
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            if self.error_handler:
                # Create detailed error
                processing_error = ProcessingError(
                    f"File processing failed for {file_path}",
                    category=ErrorCategory.DATA_PROCESSING,
                    severity=ErrorSeverity.HIGH,
                    details={
                        'file_path': file_path,
                        'file_type': file_type,
                        'analysis_keys': list(analysis.keys()) if analysis else None
                    },
                    original_exception=e
                )
                raise processing_error
            else:
                raise

    def standardize_plate_id(self, plate_id_value: Any) -> str:
        """
        Standardize plate ID values to pad single digits with zero.
        
        Args:
            plate_id_value: The original plate ID value
            
        Returns:
            str: The standardized plate ID with zero-padded numbers
        """
        if pd.isna(plate_id_value):
            return str(plate_id_value)
        
        plate_id_str = str(plate_id_value).strip()
        
        # Check if it's a letter followed by a number (e.g., A1, B10)
        match = re.match(r'^([A-Za-z])(\d+)$', plate_id_str)
        if match:
            letter = match.group(1)
            number = match.group(2)
            # Pad single digit numbers with zero
            if len(number) == 1:
                return f"{letter}0{number}"
            else:
                return plate_id_str
        
        # Return as is for other formats
        return plate_id_str

    def combine_results(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine results from different file types into a single DataFrame.
        
        Args:
            results: Dictionary containing DataFrames for each file type
                    (protocol, data, map)
                    
        Returns:
            pd.DataFrame: Combined DataFrame with all data merged
            
        Raises:
            ValueError: If no data is provided to combine
        """
        if not results:
            raise ValueError("No data to combine")
        
        # Standardize Plate ID values across all DataFrames
        for category, df in results.items():
            if 'Plate ID' in df.columns:
                results[category]['Plate ID'] = df['Plate ID'].apply(self.standardize_plate_id).astype(str)
        
        # Start with protocol data if available (as it typically has the most complete metadata)
        if 'protocol' in results:
            combined_df = results['protocol'].copy()
            merge_order = ['map', 'data']
        else:
            # Otherwise start with the first available dataframe
            combined_df = None
            for category in ['data', 'map']:
                if category in results:
                    combined_df = results[category].copy()
                    merge_order = ['protocol', 'map', 'data']
                    merge_order.remove(category)
                    break
        
        if combined_df is None:
            raise ValueError("No valid data found in results")
        
        # Merge remaining dataframes
        for category in merge_order:
            if category in results:
                combined_df = self._merge_dataframes(combined_df, results[category])
        
        return combined_df

    def validate_plate_data(self, df: pd.DataFrame, expected_wells: int = 96) -> Dict[str, Any]:
        """
        Validate plate data structure and completeness.
        
        Args:
            df: DataFrame to validate
            expected_wells: Expected number of wells (96 or 384)
            
        Returns:
            dict: Validation results with status and metrics
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check required columns
        required_columns = ['Plate ID', 'Well ID']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation['is_valid'] = False
            validation['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check well ID format and completeness
        if 'Well ID' in df.columns:
            well_ids = df['Well ID'].dropna().unique()
            validation['metrics']['unique_wells'] = len(well_ids)
            
            # Validate well ID format (e.g., A1, B2, H12)
            invalid_wells = []
            for well_id in well_ids:
                if not re.match(r'^[A-H]\d{1,2}$', str(well_id)):
                    invalid_wells.append(well_id)
            
            if invalid_wells:
                validation['warnings'].append(f"Invalid well ID format: {invalid_wells[:5]}")
            
            # Check for expected number of wells
            if len(well_ids) != expected_wells:
                validation['warnings'].append(
                    f"Expected {expected_wells} wells, found {len(well_ids)}"
                )
        
        # Check for data completeness
        if len(df) > 0:
            data_columns = [col for col in df.columns if col not in ['Plate ID', 'Well ID']]
            for col in data_columns:
                missing_data = df[col].isna().sum()
                validation['metrics'][f'{col}_missing'] = missing_data
                if missing_data > 0:
                    missing_percent = (missing_data / len(df)) * 100
                    if missing_percent > 50:
                        validation['warnings'].append(
                            f"Column '{col}' has {missing_percent:.1f}% missing data"
                        )
        
        return validation
    
    def _detect_delimiter_from_file(self, csv_filepath: str) -> str:
        """Detect delimiter from file content"""
        try:
            with open(csv_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first few lines to detect delimiter
                sample_lines = []
                for _ in range(10):
                    line = f.readline()
                    if line:
                        sample_lines.append(line.strip())
                    else:
                        break
                
                if not sample_lines:
                    return ','  # Default to comma
                
                # Count occurrences of common delimiters
                delimiter_counts = {',': 0, ';': 0, '\t': 0, '|': 0}
                
                for line in sample_lines:
                    for delim in delimiter_counts:
                        delimiter_counts[delim] += line.count(delim)
                
                # Choose delimiter with highest count
                # Special handling for semicolon vs comma
                if delimiter_counts[';'] > 0 and delimiter_counts[','] > 0:
                    # If semicolons are more frequent, it's likely European format
                    if delimiter_counts[';'] > delimiter_counts[',']:
                        logger.info(f"Detected European CSV format (semicolon delimiter) for {csv_filepath}")
                        return ';'
                
                # Return delimiter with highest count
                max_delimiter = max(delimiter_counts.keys(), key=lambda k: delimiter_counts[k])
                
                # If no delimiter found, default to comma
                if delimiter_counts[max_delimiter] == 0:
                    return ','
                
                logger.info(f"Detected delimiter '{max_delimiter}' for {csv_filepath}")
                return max_delimiter
                
        except Exception as e:
            logger.warning(f"Error detecting delimiter for {csv_filepath}: {str(e)}. Using comma as default.")
            return ','

    def _parse_csv_with_fallback(self, csv_filepath: str) -> pd.DataFrame:
        """Parse CSV with fallback strategies"""
        try:
            # First detect delimiter to handle semicolon-delimited files
            delimiter = self._detect_delimiter_from_file(csv_filepath)
            
            # Use basic pandas parsing for extract_plate_block to ensure Claude's indices work correctly
            # SmartCSVParser removes headers which breaks Claude's index expectations
            df = pd.read_csv(csv_filepath, header=None, delimiter=delimiter)
            logger.info(f"Parsed {csv_filepath} with basic pandas using delimiter '{delimiter}', shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.warning(f"Basic pandas parsing failed for {csv_filepath}: {str(e)}, trying SmartCSVParser")
            
            # Fallback to SmartCSVParser if available
            if self.csv_parser:
                try:
                    df, metadata = self.csv_parser.parse_file(csv_filepath)
                    # Convert to header=None format for compatibility
                    df = pd.DataFrame(df.values)
                    logger.info(f"SmartCSVParser fallback successful, shape: {df.shape}")
                    return df
                    
                except Exception as fallback_error:
                    logger.error(f"SmartCSVParser also failed for {csv_filepath}: {str(fallback_error)}")
                    
                    if self.error_handler:
                        # Create detailed error and try recovery
                        csv_error = ProcessingError(
                            f"CSV parsing failed for {csv_filepath}",
                            category=ErrorCategory.CSV_PARSING,
                            severity=ErrorSeverity.MEDIUM,
                            details={
                                'file_path': csv_filepath,
                                'basic_error': str(e),
                                'enhanced_error': str(fallback_error)
                            }
                        )
                        
                        # Try recovery strategy
                        recovery_result = self.error_handler.handle_error(
                            csv_error,
                            {'file_path': csv_filepath, 'allow_empty': True}
                        )
                        
                        if recovery_result is not None:
                            logger.info(f"Using recovery DataFrame for {csv_filepath}")
                            return recovery_result
                        else:
                            raise csv_error
                    else:
                        raise fallback_error
            else:
                # No enhanced parser available, re-raise original error
                raise e

    def _get_block_indices(self, analysis: Dict[str, Any], block_type: str) -> tuple:
        """Extract block indices from Claude's analysis"""
        if block_type == 'map':
            indices = analysis['raw_mapping_indices']
        else:
            indices = analysis[f'raw_{block_type}_indices']
        
        start_row = indices['start_row']
        end_row = indices['end_row']
        start_col = indices['start_col']
        end_col = indices['end_col']
        
        return start_row, end_row, start_col, end_col

    def _correct_end_row(
        self, 
        df: pd.DataFrame, 
        start_row: int, 
        end_row: int, 
        start_col: int, 
        end_col: int, 
        block_type: str
    ) -> int:
        """Apply corrections for common Claude detection issues"""
        # Fix for mapping file: Claude sometimes returns end_row: 7 but it should be 8
        # This happens when Claude misses the last row of an 8x12 plate
        if block_type == 'map' and end_row == 7 and start_row == 1:
            # Check if there's actually a row 8 with data
            if len(df) > 8 and not df.iloc[8, start_col:end_col+1].isna().all():
                logger.info("Adjusting mapping end_row from 7 to 8 to capture all 96 wells")
                end_row = 8
        
        # Fix for inconsistent end_row detection for data files
        # Claude sometimes returns end_row: 7 for 8-row plates
        if end_row == 7 and start_row == 1:
            # Check if there's actually a row 8 with data
            if len(df) > 8 and not df.iloc[8, start_col:end_col+1].isna().all():
                logger.info("Adjusting end_row from 7 to 8 to capture all 96 wells")
                end_row = 8
        
        return end_row

    def _format_block_data(self, block: pd.DataFrame, plate_id: str, block_type: str) -> pd.DataFrame:
        """Convert block data to well-based format"""
        reformatted_data = []
        
        for row_idx in range(block.shape[0]):
            for col_idx in range(block.shape[1]):
                # Create well ID - row_idx directly maps to rows A-H
                row_letter = chr(65 + row_idx)  # A=65, B=66, etc.
                col_number = col_idx + 1
                well_id = f"{row_letter}{col_number}"
                
                # Get value
                value = block.iloc[row_idx, col_idx]
                if pd.isna(value) or value == '':
                    value = None
                else:
                    # Ensure consistent data types - convert all values to strings
                    # This prevents mixed types (numpy.float64, str, etc.) that cause Arrow serialization errors
                    value = str(value)
                
                # Create column name
                column_name = 'raw mapping' if block_type == 'map' else f'raw {block_type}'
                
                reformatted_data.append({
                    'Plate ID': plate_id,
                    'Well ID': well_id,
                    column_name: value
                })
        
        return pd.DataFrame(reformatted_data)

    def _detect_file_type(self, analysis: Dict[str, Any]) -> str:
        """Auto-detect file type from Claude's analysis"""
        if 'raw_data_indices' in analysis:
            return 'data'
        elif 'raw_mapping_indices' in analysis:
            return 'map'
        elif 'protocol_data' in analysis or 'experimental_conditions' in analysis:
            return 'protocol'
        else:
            raise ValueError("Cannot determine file type from analysis")

    def _process_protocol_file(self, analysis: Union[Dict[str, Any], str]) -> pd.DataFrame:
        """Process protocol file analysis into DataFrame with dynamic field processing"""
        
        # Handle string responses from Claude (pure JSON or text with embedded JSON)
        if isinstance(analysis, str):
            import json
            import re
            
            # Try to parse as pure JSON first (new format)
            try:
                analysis = json.loads(analysis.strip())
                logger.info("Successfully parsed pure JSON response")
            except json.JSONDecodeError:
                # Fallback: Try to find JSON in markdown code blocks (legacy format)
                json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', analysis, re.DOTALL)
                if json_block_match:
                    try:
                        analysis = json.loads(json_block_match.group(1))
                        logger.info("Successfully extracted JSON from markdown code block")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON from code block: {e}")
                
                # If still string, try to find any JSON in the text
                if isinstance(analysis, str):
                    json_match = re.search(r'\{.*\}', analysis, re.DOTALL)
                    if json_match:
                        try:
                            analysis = json.loads(json_match.group())
                            logger.info("Successfully extracted JSON from text")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse extracted JSON: {e}")
                            # Create a fallback simple response
                            analysis = {"error": "Could not parse Claude response", "raw_response": analysis[:500]}
        
        # Now process as JSON format with dynamic field handling
        if isinstance(analysis, dict):
            protocol_data = []
            
            # Format 1: Plate-nested structure - Check for different plate key formats
            plate_like_keys = []
            for key in analysis.keys():
                if isinstance(key, str):
                    # Original format: "plate_1", "plate_4", etc.
                    if key.startswith(('plate_', 'Plate_', 'PLATE_')):
                        plate_like_keys.append(key)
                    # New format: direct plate IDs like "1", "4", etc. (single digits or short strings)
                    elif len(key) <= 3 and (key.isdigit() or key.isalnum()):
                        # Check if this key contains well-like data
                        plate_data = analysis[key]
                        if isinstance(plate_data, dict):
                            # Check if it contains well IDs
                            well_like_keys = [k for k in plate_data.keys() 
                                            if isinstance(k, str) and len(k) <= 3 and 
                                            len(k) >= 2 and k[0] in 'ABCDEFGH']
                            if len(well_like_keys) > 10:  # Likely a plate with wells
                                plate_like_keys.append(key)
            
            if plate_like_keys:
                for plate_key in plate_like_keys:
                    plate_data = analysis[plate_key]
                    if isinstance(plate_data, dict):
                        # Extract plate ID from key
                        if plate_key.startswith(('plate_', 'Plate_', 'PLATE_')):
                            # Original format: "plate_1" -> "1"
                            plate_id = plate_key.replace('plate_', '').replace('Plate_', '').replace('PLATE_', '')
                        else:
                            # New format: direct plate ID
                            plate_id = plate_key
                        
                        for well_id, conditions in plate_data.items():
                            if isinstance(well_id, str) and len(well_id) <= 3 and well_id[0] in 'ABCDEFGH':
                                # Dynamic field processing: extract all condition fields
                                row_data = {
                                    'Plate ID': plate_id,
                                    'Well ID': well_id
                                }
                                
                                # Add all condition fields dynamically with prefixes
                                if isinstance(conditions, dict):
                                    for field_name, field_value in conditions.items():
                                        # Use original field names from Claude with protocol_ prefix
                                        column_name = f"protocol_{field_name}"
                                        row_data[column_name] = str(field_value)
                                else:
                                    # If conditions is not a dict, store as single field
                                    row_data['protocol_conditions'] = str(conditions)
                                
                                protocol_data.append(row_data)
            
            # Format 2: Well-by-well mapping in root level
            elif all(key.startswith(('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')) and len(key) <= 3 for key in analysis.keys() if isinstance(key, str)):
                # This looks like well IDs (A1, A2, etc.)
                for well_id, conditions in analysis.items():
                    if isinstance(well_id, str) and len(well_id) <= 3:
                        row_data = {
                            'Plate ID': 'Unknown',  # Will be updated if found
                            'Well ID': well_id
                        }
                        
                        # Add all condition fields dynamically
                        if isinstance(conditions, dict):
                            for field_name, field_value in conditions.items():
                                column_name = f"protocol_{field_name}"
                                row_data[column_name] = str(field_value)
                        else:
                            row_data['protocol_conditions'] = str(conditions)
                            
                        protocol_data.append(row_data)
            
            # Format 3: Structured format with metadata and well data
            elif 'metadata' in analysis or 'wells' in analysis or 'experimental_conditions' in analysis:
                plate_id = 'Unknown'
                
                # Extract plate ID if available
                if 'metadata' in analysis and 'plate_id' in analysis['metadata']:
                    plate_id_info = analysis['metadata']['plate_id']
                    if isinstance(plate_id_info, dict) and 'value' in plate_id_info:
                        plate_id = plate_id_info['value']
                    else:
                        plate_id = str(plate_id_info)
                
                # Process well data
                well_data = analysis.get('wells', analysis.get('experimental_conditions', {}))
                for well_id, conditions in well_data.items():
                    row_data = {
                        'Plate ID': plate_id,
                        'Well ID': well_id
                    }
                    
                    # Add all condition fields dynamically
                    if isinstance(conditions, dict):
                        for field_name, field_value in conditions.items():
                            column_name = f"protocol_{field_name}"
                            row_data[column_name] = str(field_value)
                    else:
                        row_data['protocol_conditions'] = str(conditions)
                        
                    protocol_data.append(row_data)
            
            # If we collected protocol data from JSON format, process and validate it
            if protocol_data:
                df = pd.DataFrame(protocol_data)
                
                # Comprehensive validation
                validation_result = self._validate_protocol_dataframe(df)
                if not validation_result['is_valid']:
                    logger.warning(f"Protocol DataFrame validation failed: {validation_result['errors']}")
                    # Continue with warnings but don't fail completely
                    for warning in validation_result['warnings']:
                        logger.warning(f"Protocol validation warning: {warning}")
                
                # Ensure all columns are string type to prevent Arrow serialization errors
                for col in df.columns:
                    df[col] = df[col].astype(str)
                
                logger.info(f"Successfully processed protocol data: {len(df)} wells, {len(df.columns)} columns")
                return df
        
        # If no protocol data was found, log error and return empty DataFrame
        logger.error("No recognizable protocol data found in Claude response")
        logger.debug(f"Analysis keys: {list(analysis.keys()) if isinstance(analysis, dict) else 'Not a dict'}")
        
        # Return empty DataFrame with basic structure
        return pd.DataFrame(columns=['Plate ID', 'Well ID', 'protocol_conditions'])
    
    def _validate_protocol_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate protocol DataFrame structure and content"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check required columns
        required_columns = ['Plate ID', 'Well ID']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation['is_valid'] = False
            validation['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check data completeness
        if len(df) == 0:
            validation['warnings'].append("DataFrame is empty")
            validation['metrics']['total_wells'] = 0
            return validation
        
        validation['metrics']['total_wells'] = len(df)
        validation['metrics']['unique_plates'] = df['Plate ID'].nunique() if 'Plate ID' in df.columns else 0
        validation['metrics']['columns'] = list(df.columns)
        
        # Validate well ID format if present
        if 'Well ID' in df.columns:
            well_ids = df['Well ID'].dropna().unique()
            invalid_wells = []
            for well_id in well_ids:
                if not re.match(r'^[A-H]\d{1,2}$', str(well_id)):
                    invalid_wells.append(well_id)
            
            if invalid_wells:
                validation['warnings'].append(f"Invalid well ID format: {invalid_wells[:5]}")
        
        # Check for protocol condition fields
        protocol_columns = [col for col in df.columns if col.startswith('protocol_')]
        validation['metrics']['protocol_fields'] = len(protocol_columns)
        
        if len(protocol_columns) == 0:
            validation['warnings'].append("No protocol condition fields found")
        
        return validation

    def _merge_dataframes(self, left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
        """Merge two DataFrames on Plate ID and Well ID"""
        # Get overlapping columns (excluding merge keys)
        merge_keys = ['Plate ID', 'Well ID']
        overlapping_cols = set(left_df.columns) & set(right_df.columns) - set(merge_keys)
        
        # For overlapping columns, keep only those that don't already exist in left_df
        cols_to_merge = merge_keys + [
            col for col in right_df.columns 
            if col not in merge_keys and col not in overlapping_cols
        ]
        
        return pd.merge(
            left_df,
            right_df[cols_to_merge],
            on=merge_keys,
            how='outer'
        )


# Convenience functions for backward compatibility and global access
_data_processing_service_instance = None

def get_data_processing_service() -> DataProcessingService:
    """Get singleton instance of data processing service"""
    global _data_processing_service_instance
    if _data_processing_service_instance is None:
        _data_processing_service_instance = DataProcessingService()
    return _data_processing_service_instance

def extract_plate_block(
    csv_filepath: str,
    analysis: Dict[str, Any],
    block_type: str = 'data'
) -> pd.DataFrame:
    """Convenience function for backward compatibility"""
    service = get_data_processing_service()
    return service.extract_plate_block(csv_filepath, analysis, block_type)

def process_file(
    file_path: str, 
    analysis: Dict[str, Any],
    file_type: str = 'auto'
) -> pd.DataFrame:
    """Convenience function for backward compatibility"""
    service = get_data_processing_service()
    return service.process_file(file_path, analysis, file_type)

def standardize_plate_id(plate_id_value: Any) -> str:
    """Convenience function for backward compatibility"""
    service = get_data_processing_service()
    return service.standardize_plate_id(plate_id_value)

def combine_results(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convenience function for backward compatibility"""
    service = get_data_processing_service()
    return service.combine_results(results)