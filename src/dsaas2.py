# Core functionality extracted from streamlit_dsaas2.py
# This is a standalone version without streamlit dependencies

import os
import base64
import json
import pandas as pd
import re
import chardet
from striprtf.striprtf import rtf_to_text
import docx
from typing import Dict, Any, List, Tuple, Optional
from PyPDF2 import PdfReader
from io import StringIO, BytesIO
from anthropic import Anthropic
from pathlib import Path
from dotenv import load_dotenv
import argparse
import sys
import logging
from datetime import datetime
import glob
from .csv_parser_enhanced import SmartCSVParser
from .filename_utils import FilenameSanitizer
from .error_handler import (
    RobustErrorHandler, ProcessingError, ErrorCategory, ErrorSeverity,
    with_error_handling, safe_file_read, batch_process_with_recovery
)
from .unicode_processor import UnicodeProcessor, process_scientific_text, read_scientific_file

# Load environment variables
load_dotenv()

# Initialize global error handler and Unicode processor
error_handler = RobustErrorHandler()
unicode_processor = UnicodeProcessor()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dsaas2.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import extracted services
from services.claude_api_service import (
    get_claude_service, send_to_claude, analyze_with_claude, validate_api_key as claude_validate_api_key
)
from services.validation_service import (
    get_validation_service, validate_file, validate_api_key, sanitize_filename, discover_files
)
from services.data_processing_service import (
    get_data_processing_service, extract_plate_block, process_file, 
    standardize_plate_id, combine_results
)
from services.batch_processing_service import (
    get_batch_processing_service, batch_send_to_claude, Claude_categorizer, 
    batch_analyze_files, BatchConfig
)

# Initialize services
claude_service = get_claude_service()
validation_service = get_validation_service()
data_service = get_data_processing_service()
batch_service = get_batch_processing_service()

# Legacy constants for backward compatibility (now provided by services)
SYSTEM_PROMPT = claude_service.SYSTEM_PROMPT

# Prompts are now encapsulated in claude_api_service.py
# Legacy constants for backward compatibility
MAP_PROMPT = claude_service._get_map_prompt()
DATA_PROMPT = claude_service._get_data_prompt()
PROTOCOL_PROMPT = claude_service._get_protocol_prompt()
CATEGORIZER_PROMPT = claude_service._get_categorizer_prompt()

# Original MAP_PROMPT for reference:
_ORIGINAL_MAP_PROMPT = """
You are an expert data analyst specializing in interpreting CSV files from 96-well plate experiments. Your task is to analyze the content of a CSV file and extract specific information from it. 
Your analysis should be thorough, precise, and conservative. Only report information that you are highly confident about.

The CSV file you are to analyze will be uploaded through your API in a separate content block.

Your primary objectives are:

1. Identify and extract metadata from the CSV file.
2. Locate the raw mapping block within the CSV file and determine its position-based indices.
3. Examine the file for sample-identifying information.

Please follow these steps in your analysis:

1. Examine the CSV content for metadata:
- Look for clearly labeled information such as Plate ID, experiment date, or other relevant details.
- Only identify metadata items if you are absolutely certain they are present and correctly labeled.
- For the Plate ID specifically, determine its exact location using 0-based indices (row and column).
- IMPORTANT: Extract ONLY the core identifier from Plate ID values (e.g., if you see "Plate ID 1" extract "1", if you see "Sample Plate A" extract "A").

2. Search for the raw mapping block:
- This should be a contiguous block of data (often numerical or alphanumeric) representing well positions.
- The raw mapping block is typically a rectangular grid of data without header rows or descriptive text mixed in.
- It often follows a pattern where each row corresponds to a row on the plate (e.g., rows A-H) and each column corresponds to a column on the plate (e.g., columns 1-12).

3. Determine the exact indices of the raw mapping block:
- start_row: The row index (0-based) where the raw mapping block starts
- end_row: The row index (0-based) where the raw mapping block ends
- start_col: The column index (0-based) where raw mapping starts
- end_col: The column index (0-based) where raw mapping ends

4. If you cannot confidently identify the raw mapping block, set all indices to null.

5. Examine the file for sample-identifying information:
- Look for numerical, alphanumeric, or other formats that might represent sample IDs.
- Sample IDs can be any alphanumeric string (e.g., "S001", "AB123", "CTRL-POS").
- Identify any positive and negative controls.
- Look for patterns in sample identification across the file.

After your examination, provide your findings in the following JSON format:

{
    "metadata": {
        "plate_id": {
            "value": "value",
            "row": X,
            "column": Y
        },
        "experiment_date": "value",
        ... (other metadata)
    },
    "raw_mapping_indices": {
        "start_row": X,
        "end_row": Y,
        "start_col": A,
        "end_col": B
    },
    "sample_identification": {
        "format": "description of format",
        "examples": ["example1", "example2", ...],
        "controls": {
            "positive": ["example1", "example2", ...],
            "negative": ["example1", "example2", ...]
        }
    }
}

Remember:
- Only include metadata items that you are certain about.
- The indices in raw_mapping_indices and for the plate_id location must be 0-based (first row/column is index 0, second is index 1, etc.).
- Set any raw_mapping_indices to null if you cannot confidently identify the raw mapping block.
- Include sample identification information only if you are confident about its accuracy.
- Sample IDs can be alphanumeric strings, not just numerical values.
- Provide ONLY the JSON object in your final output, with no additional text.

Please proceed, returning ONLY the JSON output.
"""

DATA_PROMPT = """
You are an expert data analyst specializing in interpreting CSV files from 96-well plate experiments. Your task is to analyze the content of a CSV file and extract specific information from it. The CSV content will be provided separately in the API call, so you should refer to it as if it were present, but do not expect it to be directly embedded in this prompt.

Your analysis should be thorough, precise, and conservative. Only report information that you are highly confident about.

Your primary objectives are:
1. Identify and extract raw experimental data from the CSV file.
2. Locate the raw experimental data block within the CSV file and determine its zero-based position indices.
3. Examine the file for sample-identifying information.

Please follow these steps in your analysis:

1. Examine the CSV content for metadata:
   - Look for clearly labeled information such as Plate ID, experiment date, or other relevant details.
   - Only identify metadata items if you are absolutely certain they are present and correctly labeled.
   - For each metadata item, including the Plate ID, determine its exact location using 0-based indices (row and column).
   - IMPORTANT: Extract ONLY the core identifier from Plate ID values (e.g., if you see "Plate ID 1" extract "1", if you see "Sample Plate A" extract "A").

2. Search for the raw experimental data block:
   - This should be a contiguous block of data (often numerical or alphanumeric) representing well positions.
   - The raw experimental data block is typically a rectangular grid of data without header rows or descriptive text mixed in.
   - It often follows a pattern where each row corresponds to a row on the plate (e.g., rows A-H) and each column corresponds to a column on the plate (e.g., columns 1-12).

3. Determine the exact indices of the raw experimental data block:
   - start_row: The row index (0-based) where the raw experimental data block starts
   - end_row: The row index (0-based) where the raw experimental data block ends
   - start_col: The column index (0-based) where raw experimental data starts
   - end_col: The column index (0-based) where raw experimental data ends

4. If you cannot confidently identify the raw experimental data block, set all indices to null.

5. Examine the file for sample-identifying information:
   - Look for numerical, alphanumeric, or other formats that might represent sample IDs.
   - Sample IDs can be any alphanumeric string (e.g., "S001", "AB123", "CTRL-POS").
   - Identify any positive and negative controls.
   - Look for patterns in sample identification across the file.

Before providing your final output, perform your analysis inside <csv_analysis> tags. Follow these steps:

1. Describe the overall structure of the CSV file.
2. Quote and list all potential metadata items you find, noting their row and column indices.
3. Quote the first few rows and columns of what you believe to be the raw experimental data block.
4. List out the indices for the raw experimental data block.
5. List all potential sample IDs you find, categorizing them as regular samples, positive controls, or negative controls if possible.
6. Reason about any uncertainties or ambiguities in your analysis.

Pay special attention to:
- Ensuring you provide row and column information for each metadata element.
- Double-checking the selection of the first row of raw experimental data, ensuring it starts from index 1 (second row) if appropriate.

After your analysis, provide your findings in the following JSON format:

{
    "metadata": {
        "plate_id": {
            "value": "value",
            "row": X,
            "column": Y
        },
        "experiment_date": {
            "value": "value",
            "row": X,
            "column": Y
        },
        ... (other metadata)
    },
    "raw_data_indices": {
        "start_row": X,
        "end_row": Y,
        "start_col": A,
        "end_col": B
    },
    "sample_identification": {
        "format": "description of format",
        "examples": ["example1", "example2", ...],
        "controls": {
            "positive": ["example1", "example2", ...],
            "negative": ["example1", "example2", ...]
        }
    }
}

Remember:
- Only include metadata items that you are certain about.
- The indices in raw_data_indices and for the metadata locations must be 0-based (first row/column is index 0, second is index 1, etc.).
- Set any raw_data_indices to null if you cannot confidently identify the raw experimental data block.
- The raw experimental data block will not contain metadata such as Plate ID, experimental temperatures, or others.
- Include sample identification information only if you are confident about its accuracy.
- Sample IDs can be alphanumeric strings, not just numerical values.
- Provide ONLY the JSON object in your final output, with no additional text or duplication of your analysis from the csv_analysis block.

Please proceed with your analysis and then provide the JSON output.
"""

# PROTOCOL_PROMPT removed - now centralized in claude_api_service.py
# Use claude_service._get_protocol_prompt() for the actual prompt

# Core functions

# send_to_claude function is now provided by claude_api_service
# Keeping wrapper for backward compatibility
def send_to_claude(
    content: str,
    system_prompt: str = None,
    prompt: str = None,
    model: str = "claude-4-sonnet-20250514",
    max_tokens: int = 5000,
    temperature: float = 0.0,
    organization_id: str = None,
    user_id: str = None
) -> str:
    """Legacy wrapper for send_to_claude - delegates to claude_api_service"""
    return claude_service.send_to_claude(
        content=content,
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        organization_id=organization_id,
        user_id=user_id
    )

@with_error_handling(category=ErrorCategory.FILE_IO, allow_partial=True)
def process_file_for_claude(file_path: str) -> dict:
    """
    Process a file for Claude API, extracting content and metadata.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        dict: Dictionary containing:
            - content: The file content (text or base64 encoded)
            - mime_type: The MIME type of the file
            - filename: The base filename
            - is_image: Whether the file is an image
            - is_binary: Whether the content is binary (base64 encoded)
            
    Raises:
        ProcessingError: If there is an error processing the file
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    is_binary = False
    content = ""
    
    try:
        if file_extension == '.pdf':
            # For PDF files, extract text using PyPDF2
            try:
                reader = PdfReader(file_path)
                content = "\n".join(page.extract_text() for page in reader.pages)
            except Exception as e:
                logger.warning(f"PDF extraction failed for {file_path}: {str(e)}")
                # Try to recover with partial content
                error_handler.store_partial_result(f"pdf_error_{file_path}", str(e))
                content = f"[PDF extraction failed: {str(e)}]"
            
        elif file_extension == '.txt':
            # Use Unicode processor for robust text file reading
            try:
                content, encoding_used = unicode_processor.read_file_with_encoding(file_path)
                logger.info(f"Read text file {file_path} with encoding: {encoding_used}")
                
                # Process Unicode content for scientific data
                processing_result = unicode_processor.process_text_comprehensive(
                    content,
                    normalize=True,
                    convert_scientific=True,
                    convert_numeric=True,
                    clean_control=True
                )
                
                content = processing_result['processed_text']
                
                # Log Unicode processing results
                if processing_result['has_changes']:
                    logger.info(f"Unicode processing applied to {file_path}: {', '.join(processing_result['processing_steps'])}")
                    if processing_result['unicode_chars_found']:
                        logger.info(f"Converted Unicode characters: {processing_result['unicode_chars_found'][:10]}")  # Log first 10
                
            except Exception as e:
                logger.warning(f"Unicode processing failed for {file_path}: {str(e)}, using fallback")
                # Fallback to original method
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    encoding = detected['encoding']
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
        
        elif file_extension == '.rtf':
            # For RTF files, strip RTF formatting with Unicode handling
            try:
                rtf_content, encoding_used = unicode_processor.read_file_with_encoding(file_path)
                content = rtf_to_text(rtf_content)
                
                # Apply Unicode processing to extracted text
                processing_result = unicode_processor.process_text_comprehensive(
                    content,
                    normalize=True,
                    convert_scientific=True,
                    convert_numeric=True,
                    clean_control=True
                )
                content = processing_result['processed_text']
                
                if processing_result['has_changes']:
                    logger.info(f"Unicode processing applied to RTF {file_path}")
                    
            except Exception as e:
                logger.warning(f"Unicode processing failed for RTF {file_path}: {str(e)}, using fallback")
                with open(file_path, 'r', encoding='utf-8') as f:
                    rtf_content = f.read()
                    content = rtf_to_text(rtf_content)
        
        elif file_extension in ['.doc', '.docx']:
            # For Word documents with Unicode handling
            try:
                doc = docx.Document(file_path)
                content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                
                # Apply Unicode processing to extracted text
                processing_result = unicode_processor.process_text_comprehensive(
                    content,
                    normalize=True,
                    convert_scientific=True,
                    convert_numeric=True,
                    clean_control=True
                )
                content = processing_result['processed_text']
                
                if processing_result['has_changes']:
                    logger.info(f"Unicode processing applied to Word doc {file_path}")
                    
            except Exception as e:
                logger.warning(f"Unicode processing failed for Word doc {file_path}: {str(e)}")
                # Re-raise the original exception as this is likely a document structure issue
                raise
            
        elif file_extension == '.csv':
            # For CSV files, read with Unicode handling
            try:
                content, encoding_used = unicode_processor.read_file_with_encoding(file_path)
                logger.info(f"Read CSV file {file_path} with encoding: {encoding_used}")
                
                # Apply minimal Unicode processing for CSV (preserve structure)
                processing_result = unicode_processor.process_text_comprehensive(
                    content,
                    normalize=True,
                    convert_scientific=False,  # Don't convert scientific chars in CSV (preserve data)
                    convert_numeric=False,     # Don't convert numeric Unicode in CSV (preserve data)
                    clean_control=True         # Clean control chars that might break parsing
                )
                content = processing_result['processed_text']
                
                if processing_result['has_changes']:
                    logger.info(f"Unicode cleanup applied to CSV {file_path}")
                    
            except Exception as e:
                logger.warning(f"Unicode processing failed for CSV {file_path}: {str(e)}, using fallback")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            # For images, read as binary
            with open(file_path, 'rb') as f:
                content = base64.b64encode(f.read()).decode('utf-8')
            is_binary = True
        
        else:
            # For unsupported formats, try reading as text or binary
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                with open(file_path, 'rb') as f:
                    content = base64.b64encode(f.read()).decode('utf-8')
                is_binary = True
            
    except Exception as e:
        # Log the error but don't fail completely
        logger.error(f"Error reading file {file_path}: {str(e)}")
        error_handler.store_partial_result(f"file_read_error_{file_path}", str(e))
        
        # Return minimal viable data
        return {
            'content': f"[File read error: {str(e)}]",
            'mime_type': 'text/plain',
            'filename': os.path.basename(file_path),
            'is_image': False,
            'is_binary': False,
            'error': str(e)
        }
    
    # Get MIME type
    mime_type_map = {
        'txt': 'text/plain',
        'csv': 'text/csv',
        'pdf': 'application/pdf',
        'doc': 'application/msword',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'rtf': 'text/rtf',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png'
    }
    mime_type = mime_type_map.get(file_extension[1:], 'application/octet-stream')
    
    return {
        'content': content,
        'mime_type': mime_type,
        'filename': os.path.basename(file_path),
        'is_image': mime_type.startswith('image/'),
        'is_binary': is_binary
    }

@with_error_handling(category=ErrorCategory.DATA_PROCESSING, allow_partial=True)
# Claude_categorizer function is now provided by batch_processing_service
# Keeping wrapper for backward compatibility
def Claude_categorizer(files, organization_id=None, user_id=None):
    """Legacy wrapper for Claude_categorizer - delegates to batch_processing_service"""
    # Convert old format to new format
    file_paths = [f[0] if isinstance(f, tuple) else f for f in files]
    result = batch_service.categorize_files_batch(file_paths, organization_id=organization_id, user_id=user_id)
    
    # Convert back to old format
    categorized_files = {'data': [], 'map': [], 'protocol': [], 'other': []}
    for file_path, category in result.items():
        # Find original tuple format
        original_file = next((f for f in files if (isinstance(f, tuple) and f[0] == file_path) or f == file_path), None)
        if original_file:
            categorized_files[category].append(original_file)
    
    return categorized_files

# analyze_with_claude function is now provided by claude_api_service
# Keeping wrapper for backward compatibility
def analyze_with_claude(file_path: str, analysis_type: str, response_type: str = 'json') -> Dict[str, Any]:
    """Legacy wrapper for analyze_with_claude - delegates to claude_api_service"""
    # Process the file for Claude
    file_data = process_file_for_claude(file_path)
    
    # Use the old prompts directly for compatibility
    response = claude_service.send_to_claude(
        content=file_data['content'],
        prompt=analysis_type
    )
    
    # Parse response based on response_type
    if response_type == 'csv':
        # For protocol files, extract CSV data from response
        import re
        csv_match = re.search(r'<csv_output>(.*?)</csv_output>', response, re.DOTALL)
        if csv_match:
            csv_data = csv_match.group(1).strip()
            # Also extract summary if available
            summary_match = re.search(r'<summary>(.*?)</summary>', response, re.DOTALL)
            summary = summary_match.group(1).strip() if summary_match else ""
            return {
                'csv_data': csv_data,
                'summary': summary
            }
        else:
            # Fallback if no CSV tags found
            return {
                'csv_data': response,
                'summary': ""
            }
    else:
        # Parse JSON response
        import json
        return json.loads(response)

# extract_plate_block function is now provided by data_processing_service
# Keeping wrapper for backward compatibility
def extract_plate_block(
    csv_filepath: str,
    analysis: Dict[str, Any],
    block_type: str = 'data'
) -> pd.DataFrame:
    """Legacy wrapper for extract_plate_block - delegates to data_processing_service"""
    return data_service.extract_plate_block(csv_filepath, analysis, block_type)

# process_file function is now provided by data_processing_service  
# Keeping wrapper for backward compatibility
def process_file(file_path: str, analysis: Dict[str, Any], file_type: str = 'auto') -> pd.DataFrame:
    """Legacy wrapper for process_file - delegates to data_processing_service"""
    return data_service.process_file(file_path, analysis, file_type)

# standardize_plate_id function is now provided by data_processing_service
# Keeping wrapper for backward compatibility  
def standardize_plate_id(plate_id_value):
    """Legacy wrapper for standardize_plate_id - delegates to data_processing_service"""
    return data_service.standardize_plate_id(plate_id_value)

# combine_results function is now provided by data_processing_service
# Keeping wrapper for backward compatibility
def combine_results(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Legacy wrapper for combine_results - delegates to data_processing_service"""
    return data_service.combine_results(results)

# sanitize_filename function is now provided by validation_service
# Keeping wrapper for backward compatibility
def sanitize_filename(filename: str) -> str:
    """Legacy wrapper for sanitize_filename - delegates to validation_service"""
    return validation_service.sanitize_filename(filename)

# validate_api_key function is now provided by validation_service
# Keeping wrapper for backward compatibility
def validate_api_key(api_key=None):
    """Legacy wrapper for validate_api_key - delegates to validation_service"""
    return validation_service.validate_api_key(api_key)

# validate_file function is now provided by validation_service
# Keeping wrapper for backward compatibility
def validate_file(file_path: str) -> bool:
    """Legacy wrapper for validate_file - delegates to validation_service"""
    return validation_service.validate_file(file_path)

# discover_files function is now provided by validation_service
# Keeping wrapper for backward compatibility
def discover_files(directory: str, extensions: List[str] = None) -> List[Tuple[str, str]]:
    """Legacy wrapper for discover_files - delegates to validation_service"""
    return validation_service.discover_files(directory, extensions)

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='LLM Data Analysis System - Process experimental data files using Claude AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process files in a directory
  python dsaas2.py --directory /path/to/files
  
  # Process specific files
  python dsaas2.py --files file1.csv file2.pdf file3.txt
  
  # Specify output file
  python dsaas2.py --directory /path/to/files --output results.csv
  
  # Include specific file types only
  python dsaas2.py --directory /path/to/files --extensions csv pdf txt
"""
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--directory', '-d',
        type=str,
        help='Directory containing files to process'
    )
    input_group.add_argument(
        '--files', '-f',
        nargs='+',
        help='Specific files to process'
    )
    
    # Additional options
    parser.add_argument(
        '--extensions', '-e',
        nargs='+',
        default=['csv', 'txt', 'pdf', 'docx', 'doc', 'rtf'],
        help='File extensions to include (default: csv txt pdf docx doc rtf)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='unified_experimental_data.csv',
        help='Output file name (default: unified_experimental_data.csv)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue processing even if some files fail'
    )
    
    return parser.parse_args()

def generate_processing_report(error_handler: RobustErrorHandler, start_time: datetime, files_processed: int, output_file: str) -> str:
    """
    Generate a comprehensive processing report including errors and partial results.
    """
    report = error_handler.generate_error_report()
    end_time = datetime.now()
    processing_time = end_time - start_time
    
    report_lines = [
        "\n" + "="*60,
        "PROCESSING REPORT",
        "="*60,
        f"Processing Time: {processing_time}",
        f"Files Processed: {files_processed}",
        f"Output File: {output_file}",
        f"Status: {report.get('status', 'unknown')}",
        f"Total Errors: {report.get('total_errors', 0)}",
        ""
    ]
    
    total_errors = report.get('total_errors', 0)
    if total_errors > 0:
        report_lines.extend([
            "Error Summary:",
            f"  Recoverable Errors: {report.get('recoverable_errors', 0)}",
            f"  Unrecoverable Errors: {report.get('unrecoverable_errors', 0)}",
            ""
        ])
        
        errors_by_category = report.get('errors_by_category', {})
        if errors_by_category:
            report_lines.append("Errors by Category:")
            for category, count in errors_by_category.items():
                report_lines.append(f"  {category}: {count}")
            report_lines.append("")
        
        if report.get('partial_results_available', False):
            report_lines.append("Partial results were preserved and may be useful.")
            partial_results = error_handler.get_partial_results()
            for key, result in partial_results.items():
                report_lines.append(f"  {key}: {type(result).__name__}")
            report_lines.append("")
    
    report_lines.append("="*60)
    return "\n".join(report_lines)

def main_workflow(files: List[Tuple[str, str]], output_file: str, dry_run: bool = False, continue_on_error: bool = False) -> None:
    """
    Main workflow to process files and generate output.
    
    Args:
        files: List of tuples containing (file_path, extension)
        output_file: Path to output CSV file
        dry_run: If True, only show what would be processed
        continue_on_error: If True, continue processing even if some files fail
    """
    if not files:
        logger.error("No files to process")
        return
    
    # Initialize error handler for this workflow
    workflow_start_time = datetime.now()
    
    logger.info(f"Processing {len(files)} files...")
    logger.info(f"Continue on error: {continue_on_error}")
    
    if dry_run:
        logger.info("DRY RUN - Files that would be processed:")
        for file_path, ext in files:
            logger.info(f"  {os.path.basename(file_path)} ({ext})")
        return
    
    try:
        # Step 1: Categorize files
        logger.info("Step 1: Categorizing files with Claude...")
        try:
            categorized = Claude_categorizer(files)
        except Exception as e:
            if continue_on_error:
                logger.error(f"File categorization failed: {str(e)}, using fallback categorization")
                # Fallback: put all files in 'other' category
                categorized = {'other': files, 'data': [], 'map': [], 'protocol': []}
            else:
                raise
        
        # Log categorization results
        for category, cat_files in categorized.items():
            if cat_files:
                logger.info(f"Category '{category}': {len(cat_files)} files")
                for file_path, _ in cat_files:
                    logger.info(f"  - {os.path.basename(file_path)}")
        
        # Step 2: Process each category
        logger.info("Step 2: Processing files by category...")
        results = {}
        prompts = {
            'data': DATA_PROMPT,
            'map': MAP_PROMPT,
            'protocol': PROTOCOL_PROMPT
        }
        response_types = {
            'data': 'json',
            'map': 'json',
            'protocol': 'csv'
        }
        
        for category, cat_files in categorized.items():
            if category != 'other' and cat_files:
                logger.info(f"Processing {category} files...")
                category_dfs = []
                
                for file_path, _ in cat_files:
                    logger.info(f"  Processing {os.path.basename(file_path)}...")
                    try:
                        # First analyze the file with Claude
                        analysis = analyze_with_claude(
                            file_path,
                            prompts[category],
                            response_types[category]
                        )
                        
                        # Then process the file with the analysis
                        df = process_file(
                            file_path,
                            analysis,
                            category
                        )
                        category_dfs.append(df)
                        logger.info(f"    Extracted {len(df)} records")
                    except Exception as e:
                        logger.error(f"    Failed to process {os.path.basename(file_path)}: {str(e)}")
                        if not continue_on_error:
                            raise
                        continue
                
                if category_dfs:
                    if len(category_dfs) > 1:
                        results[category] = pd.concat(category_dfs, ignore_index=True)
                    else:
                        results[category] = category_dfs[0]
                    logger.info(f"Total {category} records: {len(results[category])}")
        
        # Step 3: Combine results
        if results:
            logger.info("Step 3: Combining results...")
            combined_df = combine_results(results)
            
            # Step 4: Save output
            logger.info(f"Step 4: Saving results to {output_file}...")
            combined_df.to_csv(output_file, index=False)
            logger.info(f"Successfully saved {len(combined_df)} records to {output_file}")
            
            # Log summary
            logger.info("\n" + "="*50)
            logger.info("PROCESSING SUMMARY")
            logger.info("="*50)
            logger.info(f"Files processed: {len([f for cat_files in results.values() for f in cat_files]) if results else 0}")
            logger.info(f"Total records: {len(combined_df)}")
            logger.info(f"Columns: {', '.join(combined_df.columns)}")
            logger.info(f"Output file: {output_file}")
            logger.info("="*50)
        else:
            logger.warning("No processable files found or all processing failed")
            
    except Exception as e:
        logger.error(f"Error in main workflow: {str(e)}")
        
        # Generate error report
        report = generate_processing_report(error_handler, workflow_start_time, len(files), output_file)
        logger.info(report)
        
        if not continue_on_error:
            raise
        else:
            logger.warning("Continuing despite errors due to --continue-on-error flag")

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate API key
    try:
        validate_api_key()
        logger.info("API key validated successfully.")
    except ValueError as e:
        logger.error(f"API key validation failed: {e}")
        sys.exit(1)
    
    # Determine files to process
    files = []
    if args.directory:
        logger.info(f"Discovering files in directory: {args.directory}")
        if not os.path.isdir(args.directory):
            logger.error(f"Directory not found: {args.directory}")
            sys.exit(1)
        files = discover_files(args.directory, args.extensions)
    elif args.files:
        logger.info(f"Processing specified files: {len(args.files)} files")
        for file_path in args.files:
            if validate_file(file_path):
                ext = os.path.splitext(file_path)[1].lower().lstrip('.')
                if ext in args.extensions:
                    files.append((file_path, ext))
                else:
                    logger.warning(f"Skipping file with unsupported extension: {file_path}")
            else:
                logger.error(f"Invalid file: {file_path}")
    
    if not files:
        logger.error("No valid files found to process")
        sys.exit(1)
    
    # Start processing
    start_time = datetime.now()
    logger.info(f"Starting LLM Data Analysis System at {start_time}")
    logger.info(f"Found {len(files)} files to process")
    
    try:
        # Run main workflow
        main_workflow(files, args.output, args.dry_run, args.continue_on_error)
        
        # Generate final report
        final_report = generate_processing_report(error_handler, start_time, len(files), args.output)
        logger.info(final_report)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = end_time - start_time
        logger.info(f"Processing completed in {processing_time}")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)