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
import asyncio
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# Load environment variables
load_dotenv()

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

# Initialize the Anthropic API
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Batch processing configuration
@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    max_concurrent_requests: int = 3
    max_requests_per_minute: int = 50
    max_tokens_per_minute: int = 20000
    request_timeout: int = 120
    retry_attempts: int = 3
    retry_delay: float = 1.0
    batch_size: int = 5
    
    @property
    def timeout(self) -> int:
        """Alias for request_timeout for backward compatibility"""
        return self.request_timeout

# Global batch configuration
batch_config = BatchConfig()

# Rate limiting tracker
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
            time_since_window_start = current_time - self.window_start
            return max(0, 60 - time_since_window_start)

# Global rate limiter
rate_limiter = RateLimiter(batch_config.max_requests_per_minute, batch_config.max_tokens_per_minute)

# System prompt for Claude
SYSTEM_PROMPT = """You are a detail-oriented, precise, and patient researcher. Your long years of experience have taught you that doing the job properly the first time 
is more valuable than anything else, so you do not guess. While some may even call you pedantic, everyone knows that you only make decisions in your job that are logical, 
rational, and supported by the documents you are given."""

# User prompts for Claude
CATEGORIZER_PROMPT = """
        You will be given a list of files along with their content and metadata. 
        Your task is to review this information and categorize each file into one of four categories based on its content and purpose. 
        Here is the list of files with their content and metadata:

        <file_list>
        {file_content}
        </file_list>

        Your task is to categorize each file into one of the following categories:
        1. "data" - for files containing experimental data
        2. "map" - for files mapping sample locations to sample identifiers
        3. "protocol" - for files specifying experimental protocol(s)
        4. "other" - for files containing some other type of information

        Instructions:
        1. Carefully review the content and metadata of each file in the list.
        2. Based on the information provided, determine which category best describes each file.
        3. Categorize each file into one and only one category.
        4. In cases where the categorization is ambiguous or unclear, use the "other" category.
        5. Do not guess or make assumptions about the file's content or purpose if it's not clearly evident from the provided information.

        Output your categorization as a JSON object, where each key is a filename and its value is the corresponding category. Use the following category labels exactly as written: "data", "map", "protocol", "other". Provide no other output.

        Example output format:
        {{
        "file1.txt": "data",
        "file2.csv": "map",
        "file3.docx": "protocol",
        "file4.pdf": "other"
        }}

        Remember:
        - Each file must be categorized into one and only one category.
        - If you're unsure about a file's category, use "other" rather than guessing.
        - Provide your final answer as a single JSON object inside <answer> tags.
        - Do not include any other text or comments in your response.
        """

MAP_PROMPT = """
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

PROTOCOL_PROMPT = """
You are an AI assistant specialized in analyzing scientific documents to extract and summarize experimental protocols and conditions. Your task is to process the given document, identify key experimental information, and organize it into a structured format.

The content of the document you are to analyze is provided in a separate content block.

Please follow these steps to complete your task:

1. Carefully read through the entire document contents.

2. Identify all experimental protocols and conditions mentioned in the document. These may include, but are not limited to:
   - Temperature settings
   - Incubation times
   - Reagent concentrations
   - Equipment settings
   - Experimental procedures

3. Identify any associated identifying information such as Plate IDs, Sample IDs, or Well IDs.
   - IMPORTANT: Extract ONLY the core identifier from Plate ID values (e.g., if you see "Plate ID 1" extract "1", if you see "Sample Plate A" extract "A").

4. Connect the identified protocols and conditions to their respective IDs.

5. CRITICAL: Your output must ALWAYS be on a well-by-well basis for 96-well plates. This means:
   - If conditions apply to an entire plate, expand them to all 96 wells (A1-H12)
   - If conditions apply to specific rows (A-H), expand them to all 12 wells in those rows
   - If conditions apply to specific columns (1-12), expand them to all 8 wells in those columns
   - If conditions apply to specific wells, list them individually
   - Use your reasoning to translate any plate-level, row-level, or column-level information into individual well entries

6. Break down any complex protocols into their individual experimental conditions and parameters. Include these in your structured representation.

7. If you find protocols or conditions that are not explicitly connected to any ID, do not include them in your final output.

8. If you find an ID but cannot identify any protocols or conditions explicitly connected to it, include the ID in your output with empty values for other columns.

9. Only include information that is explicitly stated in the document. Do not make assumptions or guesses about unclear or ambiguous information.

Before providing your final output, please work through the following steps in your thinking block, wrapping your work in <protocol_extraction> tags:

1. List all unique IDs found in the document (Plate IDs, Sample IDs, Well IDs, etc.).
2. For each identified experimental protocol and condition, quote the relevant parts of the document and note any associated IDs.
3. Determine how conditions apply (plate-wide, row-specific, column-specific, or well-specific).
4. Create a sample table structure based on the identified information, showing how you plan to organize the data on a well-by-well basis.
5. Consider and note any challenges or limitations you encountered in extracting the required information.

After completing your analysis, provide your final output in the following format:

<summary>
Provide a brief overview of what you found in the document, including any challenges or limitations in extracting the required information.
</summary>

<csv_output>
Present your structured data in CSV format. Each row MUST correspond to one well, labeled by both Plate ID and Well ID. Include a header row. For example:

Plate ID,Well ID,Temperature,Incubation Time,Treatment
1,A1,37,2h,Control
1,A2,37,2h,Control
1,A3,37,2h,Drug A
...
1,H12,37,2h,Drug B

Remember:
- ALWAYS include both Plate ID and Well ID columns
- ALWAYS expand conditions to individual wells based on how they apply (plate-wide, row, column, or specific wells)
- Well IDs must follow the standard format: letter (A-H) followed by number (1-12)

Ensure that your actual output reflects the specific parameters and IDs found in the document you analyzed.
</csv_output>

Remember, your final output should only include the <summary> and <csv_output> sections. Do not include your thought process or any intermediate steps in the final output, and do not duplicate or rehash any of the work you did in the thinking block.
"""

# Core functions

def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens for rate limiting purposes"""
    # Rough approximation: 1 token ≈ 4 characters
    return len(str(text)) // 4

async def send_to_claude_async(
    content: str,
    system_prompt: str = SYSTEM_PROMPT,
    prompt: str = None,
    model: str = "claude-4-sonnet-20250514",
    max_tokens: int = 5000,
    temperature: float = 0.0,
    semaphore: asyncio.Semaphore = None,
    organization_id: str = None,
    user_id: str = None
) -> str:
    """
    Async version of send_to_claude with rate limiting and retry logic.
    
    Args:
        content: Content to send to Claude
        system_prompt: The system prompt to use
        prompt: The specific prompt to use
        model: The Claude model to use
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature parameter for response generation
        semaphore: Semaphore for controlling concurrency
        organization_id: Organization ID for usage tracking (optional)
        user_id: User ID for usage tracking (optional)
        
    Returns:
        str: Claude's response
    """
    if prompt is None:
        raise ValueError("A specific prompt must be provided")
    
    # Estimate tokens for rate limiting
    full_content = f"{system_prompt}\n{prompt}\n{content}"
    estimated_tokens = estimate_tokens(full_content)
    
    # Rate limiting check
    while not rate_limiter.can_make_request(estimated_tokens):
        wait_time = rate_limiter.get_wait_time()
        logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
        await asyncio.sleep(wait_time + 1)  # Add small buffer
    
    # Use semaphore if provided
    if semaphore:
        await semaphore.acquire()
    
    try:
        # Format the content for Claude
        if isinstance(content, dict):
            formatted_content = f"File: {content['filename']}\n"
            if content['is_binary']:
                formatted_content += f"Content (base64 encoded): {content['content'][:1000]}...\n"
            else:
                formatted_content += f"Content:\n{content['content']}\n"
        else:
            formatted_content = content
            
        full_prompt = f"{prompt}\n\nContent:\n{formatted_content}"
        
        # Retry logic with exponential backoff
        for attempt in range(batch_config.retry_attempts):
            try:
                # Run the synchronous API call in a thread pool
                loop = asyncio.get_event_loop()
                message = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: anthropic.messages.create(
                            model=model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            system=system_prompt,
                            messages=[{"role": "user", "content": full_prompt}]
                        )
                    ),
                    timeout=batch_config.request_timeout
                )
                
                # Record successful request
                rate_limiter.record_request(estimated_tokens)
                
                # Track API call usage if organization/user context provided
                if organization_id and user_id:
                    try:
                        # Import here to avoid circular imports
                        from auth.database import track_usage
                        track_usage(organization_id, user_id, 'api_call')
                        print(f"DEBUG: Async API call tracked for org {organization_id[:8]}...")
                    except Exception as e:
                        print(f"WARNING: Failed to track async API usage: {e}")
                
                return message.content[0].text
                
            except asyncio.TimeoutError:
                if attempt < batch_config.retry_attempts - 1:
                    delay = batch_config.retry_delay * (2 ** attempt)
                    logger.warning(f"Request timeout, retrying in {delay} seconds (attempt {attempt + 1}/{batch_config.retry_attempts})")
                    await asyncio.sleep(delay)
                else:
                    raise Exception(f"Request timed out after {batch_config.retry_attempts} attempts")
                    
            except Exception as e:
                if attempt < batch_config.retry_attempts - 1:
                    delay = batch_config.retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed: {str(e)}, retrying in {delay} seconds (attempt {attempt + 1}/{batch_config.retry_attempts})")
                    await asyncio.sleep(delay)
                else:
                    raise Exception(f"Request failed after {batch_config.retry_attempts} attempts: {str(e)}")
    
    finally:
        if semaphore:
            semaphore.release()

async def batch_send_to_claude(
    requests: List[Dict[str, Any]],
    max_concurrent: int = None,
    organization_id: str = None,
    user_id: str = None
) -> List[Dict[str, Any]]:
    """
    Send multiple requests to Claude concurrently with rate limiting.
    
    Args:
        requests: List of request dictionaries, each containing:
            - content: Content to send
            - prompt: Prompt to use
            - file_path: Original file path (for tracking)
            - system_prompt: System prompt (optional)
            - model: Model to use (optional)
        max_concurrent: Maximum concurrent requests (defaults to config)
        
    Returns:
        List of result dictionaries containing response and metadata
    """
    if not requests:
        return []
    
    if max_concurrent is None:
        max_concurrent = batch_config.max_concurrent_requests
    
    # Create semaphore for controlling concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    logger.info(f"Starting batch processing of {len(requests)} requests with max concurrency {max_concurrent}")
    
    async def process_single_request(request_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process a single request and return result with metadata"""
        start_time = time.time()
        try:
            response = await send_to_claude_async(
                content=request_data['content'],
                prompt=request_data['prompt'],
                system_prompt=request_data.get('system_prompt', SYSTEM_PROMPT),
                model=request_data.get('model', "claude-4-sonnet-20250514"),
                semaphore=semaphore,
                organization_id=organization_id,
                user_id=user_id
            )
            
            duration = time.time() - start_time
            logger.info(f"Request {index + 1}/{len(requests)} completed in {duration:.1f}s")
            
            return {
                'success': True,
                'response': response,
                'file_path': request_data.get('file_path', ''),
                'duration': duration,
                'error': None
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Request {index + 1}/{len(requests)} failed after {duration:.1f}s: {str(e)}")
            
            return {
                'success': False,
                'response': None,
                'file_path': request_data.get('file_path', ''),
                'duration': duration,
                'error': str(e)
            }
    
    # Process all requests concurrently
    tasks = [process_single_request(req, i) for i, req in enumerate(requests)]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    
    # Log summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_time = sum(r['duration'] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    logger.info(f"Batch processing completed: {successful} successful, {failed} failed, avg time {avg_time:.1f}s")
    
    return results

def send_to_claude(
    content: str,
    system_prompt: str = SYSTEM_PROMPT,
    prompt: str = None,
    model: str = "claude-4-sonnet-20250514",
    max_tokens: int = 5000,
    temperature: float = 0.0,
    organization_id: str = None,
    user_id: str = None
) -> str:
    """
    Sends content to Claude API and gets response back.
    
    Args:
        content: Content to send to Claude
        system_prompt: The system prompt to use
        prompt: The specific prompt to use (CATEGORIZER_PROMPT, MAP_PROMPT, etc.)
        model: The Claude model to use
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature parameter for response generation
        organization_id: Organization ID for usage tracking (optional)
        user_id: User ID for usage tracking (optional)
        
    Returns:
        str: Claude's response
    """
    if prompt is None:
        raise ValueError("A specific prompt must be provided")
        
    try:
        # Format the content for Claude
        if isinstance(content, dict):
            # If content is a dictionary from process_file_for_claude
            formatted_content = f"File: {content['filename']}\n"
            if content['is_binary']:
                formatted_content += f"Content (base64 encoded): {content['content'][:1000]}...\n"
            else:
                formatted_content += f"Content:\n{content['content']}\n"
        else:
            # If content is a plain string
            formatted_content = content
            
        # Construct the full message content
        full_prompt = f"{prompt}\n\nContent:\n{formatted_content}"
        
        message = anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": full_prompt}]
        )
        
        # Track API call usage if organization/user context provided
        if organization_id and user_id:
            try:
                # Import here to avoid circular imports
                from auth.database import track_usage
                track_usage(organization_id, user_id, 'api_call')
                print(f"DEBUG: Batch API call tracked for org {organization_id[:8]}...")
            except Exception as e:
                print(f"WARNING: Failed to track batch API usage: {e}")
        
        return message.content[0].text
            
    except Exception as e:
        print(f"Error communicating with Claude API: {str(e)}")
        raise

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
        Exception: If there is an error processing the file
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
                raise ValueError(f"Could not extract text from PDF {file_path}: {str(e)}")
            
        elif file_extension == '.txt':
            # For text files, detect encoding and read
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding']
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        
        elif file_extension == '.rtf':
            # For RTF files, strip RTF formatting
            with open(file_path, 'r', encoding='utf-8') as f:
                rtf_content = f.read()
                content = rtf_to_text(rtf_content)
        
        elif file_extension in ['.doc', '.docx']:
            # For Word documents
            doc = docx.Document(file_path)
            content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
        elif file_extension == '.csv':
            # For CSV files, read as text
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
        raise Exception(f"Error reading file {file_path}: {str(e)}")
    
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

def Claude_categorizer(files, organization_id=None, user_id=None):
    """
    Categorizes files into protocol, data, map, or other using Claude API with batch processing.
    
    Args:
        files: List of tuples containing (file_path, extension)
        organization_id: Organization ID for usage tracking (optional)
        user_id: User ID for usage tracking (optional)
        
    Returns:
        Dictionary with categories as keys and lists of files as values
    """
    categorized_files = {
        'protocol': [],
        'data': [],
        'map': [],
        'other': []
    }
    
    if not files:
        return categorized_files
    
    # Build file content list for Claude
    file_content_list = []
    valid_files = []
    
    for file in files:
        try:
            file_data = process_file_for_claude(file[0])
            file_content_list.append(f"File: {file_data['filename']}\nContent:\n{file_data['content'][:1000]}...\n")
            valid_files.append(file)
        except Exception as e:
            logger.error(f"Error reading file {os.path.basename(file[0])}: {str(e)}")
            categorized_files['other'].append(file)
    
    if not file_content_list:
        return categorized_files
    
    # For categorization, we can process all files in a single request
    # or split into smaller batches if there are many files
    max_files_per_batch = 10  # Adjust based on token limits
    
    if len(valid_files) <= max_files_per_batch:
        # Process all files in one request
        file_content = "\n".join(file_content_list)
        formatted_prompt = CATEGORIZER_PROMPT.format(file_content=file_content)
        
        try:
            response = send_to_claude(
                content="",
                prompt=formatted_prompt,
                organization_id=organization_id,
                user_id=user_id
            )
            
            # Parse the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                
                # Categorize each file
                for file in valid_files:
                    filename = os.path.basename(file[0])
                    category = result.get(filename, result.get(file[0], "other"))
                    categorized_files[category].append(file)
            else:
                logger.warning("Could not parse Claude's categorization response")
                for file in valid_files:
                    categorized_files['other'].append(file)
                    
        except Exception as e:
            logger.error(f"Error categorizing files: {str(e)}")
            for file in valid_files:
                categorized_files['other'].append(file)
    
    else:
        # Process files in batches using async batch processing
        logger.info(f"Processing {len(valid_files)} files in batches for categorization...")
        
        # Split files into batches
        batches = []
        for i in range(0, len(valid_files), max_files_per_batch):
            batch_files = valid_files[i:i + max_files_per_batch]
            batch_content = file_content_list[i:i + max_files_per_batch]
            batch_content_str = "\n".join(batch_content)
            formatted_prompt = CATEGORIZER_PROMPT.format(file_content=batch_content_str)
            
            batches.append({
                'content': "",
                'prompt': formatted_prompt,
                'file_path': f"batch_{i//max_files_per_batch + 1}",
                'files': batch_files
            })
        
        # Process batches concurrently
        try:
            results = asyncio.run(batch_send_to_claude(batches, organization_id=organization_id, user_id=user_id))
            
            # Process results
            for i, result in enumerate(results):
                if result['success']:
                    try:
                        # Parse the response
                        json_match = re.search(r'\{.*\}', result['response'], re.DOTALL)
                        if json_match:
                            batch_result = json.loads(json_match.group(0))
                            
                            # Categorize files in this batch
                            for file in batches[i]['files']:
                                filename = os.path.basename(file[0])
                                category = batch_result.get(filename, batch_result.get(file[0], "other"))
                                categorized_files[category].append(file)
                        else:
                            logger.warning(f"Could not parse response for batch {i+1}")
                            for file in batches[i]['files']:
                                categorized_files['other'].append(file)
                    except Exception as e:
                        logger.error(f"Error parsing batch {i+1} response: {str(e)}")
                        for file in batches[i]['files']:
                            categorized_files['other'].append(file)
                else:
                    logger.error(f"Batch {i+1} failed: {result['error']}")
                    for file in batches[i]['files']:
                        categorized_files['other'].append(file)
                        
        except Exception as e:
            logger.error(f"Error in batch categorization: {str(e)}")
            for file in valid_files:
                categorized_files['other'].append(file)
    
    return categorized_files

def analyze_with_claude(file_path: str, prompt: str, response_type: str = 'json', organization_id: str = None, user_id: str = None) -> Dict[str, Any]:
    """
    Analyzes a file with Claude and returns structured data.
    
    Args:
        file_path: Path to the file to analyze
        prompt: The prompt to use for analysis
        response_type: Type of response to extract ('json' or 'csv')
        organization_id: Organization ID for usage tracking (optional)
        user_id: User ID for usage tracking (optional)
        
    Returns:
        Dict containing the analysis results
    """
    try:
        # Process the file for Claude
        file_data = process_file_for_claude(file_path)
        
        # Send to Claude
        response = send_to_claude(
            content=file_data['content'],
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            organization_id=organization_id,
            user_id=user_id
        )
        
        # Parse response based on type
        if response_type == 'json':
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                raise ValueError("No JSON found in Claude's response")
                
        elif response_type == 'csv':
            # Extract CSV data from response
            csv_match = re.search(r'<csv_output>(.*?)</csv_output>', response, re.DOTALL)
            if csv_match:
                return {'csv_data': csv_match.group(1).strip()}
            else:
                raise ValueError("No CSV data found in Claude's response")
                
    except Exception as e:
        logger.error(f"Error analyzing file with Claude: {str(e)}")
        raise

async def batch_analyze_files(
    file_paths: List[str], 
    prompt: str, 
    response_type: str = 'json',
    organization_id: str = None,
    user_id: str = None
) -> List[Dict[str, Any]]:
    """
    Analyze multiple files concurrently using Claude.
    
    Args:
        file_paths: List of file paths to analyze
        prompt: The prompt to use for analysis
        response_type: Type of response to extract ('json' or 'csv')
        
    Returns:
        List of analysis results with metadata
    """
    if not file_paths:
        return []
    
    # Prepare requests for batch processing
    requests = []
    for file_path in file_paths:
        try:
            file_data = process_file_for_claude(file_path)
            requests.append({
                'content': file_data['content'],
                'prompt': prompt,
                'file_path': file_path,
                'system_prompt': SYSTEM_PROMPT
            })
        except Exception as e:
            logger.error(f"Error preparing file {file_path} for analysis: {str(e)}")
            # Add failed request for tracking
            requests.append({
                'content': "",
                'prompt': prompt,
                'file_path': file_path,
                'system_prompt': SYSTEM_PROMPT,
                'pre_error': str(e)
            })
    
    # Process batch
    results = await batch_send_to_claude(requests, organization_id=organization_id, user_id=user_id)
    
    # Parse responses
    parsed_results = []
    for i, result in enumerate(results):
        file_path = file_paths[i]
        
        if 'pre_error' in requests[i]:
            # File processing failed before sending to Claude
            parsed_results.append({
                'file_path': file_path,
                'success': False,
                'data': None,
                'error': requests[i]['pre_error']
            })
            continue
        
        if result['success']:
            try:
                # Parse response based on type
                if response_type == 'json':
                    json_match = re.search(r'\{.*\}', result['response'], re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                        parsed_results.append({
                            'file_path': file_path,
                            'success': True,
                            'data': data,
                            'error': None
                        })
                    else:
                        parsed_results.append({
                            'file_path': file_path,
                            'success': False,
                            'data': None,
                            'error': "No JSON found in Claude's response"
                        })
                        
                elif response_type == 'csv':
                    csv_match = re.search(r'<csv_output>(.*?)</csv_output>', result['response'], re.DOTALL)
                    if csv_match:
                        data = {'csv_data': csv_match.group(1).strip()}
                        parsed_results.append({
                            'file_path': file_path,
                            'success': True,
                            'data': data,
                            'error': None
                        })
                    else:
                        parsed_results.append({
                            'file_path': file_path,
                            'success': False,
                            'data': None,
                            'error': "No CSV data found in Claude's response"
                        })
                        
            except Exception as e:
                parsed_results.append({
                    'file_path': file_path,
                    'success': False,
                    'data': None,
                    'error': f"Error parsing response: {str(e)}"
                })
        else:
            parsed_results.append({
                'file_path': file_path,
                'success': False,
                'data': None,
                'error': result['error']
            })
    
    return parsed_results

def extract_plate_block(
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
    """
    # Read the CSV - always use header=None to preserve all rows
    df = pd.read_csv(csv_filepath, header=None)
    
    # Get indices
    if block_type == 'map':
        indices = analysis['raw_mapping_indices']
    else:
        indices = analysis[f'raw_{block_type}_indices']
    start_row = indices['start_row']
    end_row = indices['end_row']
    start_col = indices['start_col']
    end_col = indices['end_col']
    
    # Fix for mapping file: Claude sometimes returns end_row: 7 but it should be 8
    # This happens when Claude misses the last row of an 8x12 plate
    if block_type == 'map' and end_row == 7 and start_row == 1:
        # Check if there's actually a row 8 with data
        if len(df) > 8 and not df.iloc[8, start_col:end_col+1].isna().all():
            # Note: Adjusting mapping end_row from end_row to 8 to capture all 96 wells
            end_row = 8
    
    # Fix for inconsistent end_row detection
    # Claude sometimes returns end_row: 7 for 8-row plates
    if end_row == 7 and start_row == 1:
        # Check if there's actually a row 8 with data
        if len(df) > 8 and not df.iloc[8, start_col:end_col+1].isna().all():
            # Note: Adjusting end_row from end_row to 8 to capture all 96 wells
            end_row = 8
    
    # Extract the block
    block = df.iloc[start_row:end_row+1, start_col:end_col+1]
    
    # Get plate ID
    plate_id = analysis['metadata'].get('plate_id', {}).get('value', 'Unknown')
    
    # Process each cell
    reformatted_data = []
    for row_idx in range(block.shape[0]):
        for col_idx in range(block.shape[1]):
            # Create well ID - row_idx directly maps to rows A-H
            row_letter = chr(65 + row_idx)
            col_number = col_idx + 1
            well_id = f"{row_letter}{col_number}"
            
            # Get value
            value = block.iloc[row_idx, col_idx]
            if pd.isna(value) or value == '':
                value = None
            
            column_name = 'raw mapping' if block_type == 'map' else f'raw {block_type}'
            reformatted_data.append({
                'Plate ID': plate_id,
                'Well ID': well_id,
                column_name: value
            })
    
    return pd.DataFrame(reformatted_data)

def process_file(file_path: str, prompt: str, response_type: str = 'json') -> pd.DataFrame:
    """
    Generic function to process files using Claude.
    
    Args:
        file_path: Path to the file to process
        prompt: The prompt to use for analysis
        response_type: Type of response to extract ('json' or 'csv')
        
    Returns:
        DataFrame with processed data
    """
    try:
        # Get analysis from Claude
        analysis = analyze_with_claude(file_path, prompt, response_type)
        
        # For protocol files, return CSV data directly
        if response_type == 'csv':
            df = pd.read_csv(StringIO(analysis['csv_data']))
            return df
            
        # For data and map files, extract and process the data
        if 'raw_data_indices' in analysis:
            df = extract_plate_block(file_path, analysis, 'data')
        elif 'raw_mapping_indices' in analysis:
            df = extract_plate_block(file_path, analysis, 'map')
        else:
            raise ValueError("Unknown analysis type")
            
        # Convert Plate ID to string
        df['Plate ID'] = df['Plate ID'].astype(str)
        
        return df
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        raise

def standardize_plate_id(plate_id_value):
    """
    Standardize plate ID values to extract only the core identifier.
    
    Args:
        plate_id_value: The original plate ID value
        
    Returns:
        The standardized core identifier
    """
    if pd.isna(plate_id_value):
        return plate_id_value
    
    plate_id_str = str(plate_id_value).strip()
    
    # Common patterns to extract core identifier
    patterns = [
        r'Plate\s*ID\s*(\w+)',  # "Plate ID 1" -> "1"
        r'Plate\s*(\w+)',       # "Plate 1" -> "1" 
        r'Sample\s*Plate\s*(\w+)', # "Sample Plate A" -> "A"
        r'^(\w+)$'              # Just the identifier itself
    ]
    
    for pattern in patterns:
        match = re.search(pattern, plate_id_str, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return plate_id_str

def combine_results(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine results from different file types into a single DataFrame.
    
    Args:
        results: Dictionary containing DataFrames for each file type
            (protocol, data, map)
            
    Returns:
        Combined DataFrame with all data merged
    """
    if not results:
        raise ValueError("No data to combine")
    
    # Standardize Plate ID values across all DataFrames
    for category, df in results.items():
        if 'Plate ID' in df.columns:
            results[category]['Plate ID'] = df['Plate ID'].apply(standardize_plate_id).astype(str)
    
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
    
    # Merge remaining dataframes
    for category in merge_order:
        if category in results:
            # Get overlapping columns (excluding merge keys)
            merge_keys = ['Plate ID', 'Well ID']
            overlapping_cols = set(combined_df.columns) & set(results[category].columns) - set(merge_keys)
            
            # For overlapping columns, keep only those that don't already exist in combined_df
            cols_to_merge = merge_keys + [col for col in results[category].columns 
                                         if col not in merge_keys and col not in overlapping_cols]
            
            combined_df = pd.merge(
                combined_df,
                results[category][cols_to_merge],
                on=merge_keys,
                how='outer'
            )
    
    return combined_df

def validate_api_key():
    """Validate that the API key is properly loaded."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please ensure your .env file is properly configured.")
    return api_key

def validate_file(file_path: str) -> bool:
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

def discover_files(directory: str, extensions: List[str] = None) -> List[Tuple[str, str]]:
    """
    Discover files in a directory with specified extensions.
    
    Args:
        directory: Directory to search for files
        extensions: List of file extensions to include (e.g., ['csv', 'pdf', 'txt'])
                   If None, includes common document types
        
    Returns:
        List of tuples containing (file_path, extension)
    """
    if extensions is None:
        extensions = ['csv', 'txt', 'pdf', 'docx', 'doc', 'rtf']
    
    files = []
    for ext in extensions:
        pattern = os.path.join(directory, f"*.{ext}")
        for file_path in glob.glob(pattern):
            if validate_file(file_path):
                files.append((file_path, ext))
                logger.info(f"Found file: {os.path.basename(file_path)}")
    
    return files

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
    
    # Batch processing options
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=3,
        help='Maximum concurrent requests (default: 3)'
    )
    
    parser.add_argument(
        '--max-requests-per-minute',
        type=int,
        default=50,
        help='Maximum requests per minute (default: 50)'
    )
    
    parser.add_argument(
        '--max-tokens-per-minute',
        type=int,
        default=20000,
        help='Maximum tokens per minute (default: 20000)'
    )
    
    parser.add_argument(
        '--request-timeout',
        type=int,
        default=120,
        help='Request timeout in seconds (default: 120)'
    )
    
    parser.add_argument(
        '--retry-attempts',
        type=int,
        default=3,
        help='Number of retry attempts (default: 3)'
    )
    
    parser.add_argument(
        '--disable-batch',
        action='store_true',
        help='Disable batch processing and use sequential processing'
    )
    
    return parser.parse_args()

def main_workflow(files: List[Tuple[str, str]], output_file: str, dry_run: bool = False, use_batch: bool = True) -> None:
    """
    Main workflow to process files and generate output.
    
    Args:
        files: List of tuples containing (file_path, extension)
        output_file: Path to output CSV file
        dry_run: If True, only show what would be processed
    """
    if not files:
        logger.error("No files to process")
        return
    
    logger.info(f"Processing {len(files)} files...")
    
    if dry_run:
        logger.info("DRY RUN - Files that would be processed:")
        for file_path, ext in files:
            logger.info(f"  {os.path.basename(file_path)} ({ext})")
        return
    
    try:
        # Step 1: Categorize files
        logger.info("Step 1: Categorizing files with Claude...")
        categorized = Claude_categorizer(files)
        
        # Log categorization results
        for category, cat_files in categorized.items():
            if cat_files:
                logger.info(f"Category '{category}': {len(cat_files)} files")
                for file_path, _ in cat_files:
                    logger.info(f"  - {os.path.basename(file_path)}")
        
        # Step 2: Process each category with batch processing
        logger.info("Step 2: Processing files by category with batch processing...")
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
                
                # Extract file paths for batch processing
                file_paths = [file_path for file_path, _ in cat_files]
                
                # Use batch processing if we have multiple files and batch processing is enabled
                if len(file_paths) > 1 and use_batch:
                    logger.info(f"  Batch processing {len(file_paths)} {category} files...")
                    try:
                        # Process files in batch
                        batch_results = asyncio.run(batch_analyze_files(
                            file_paths, 
                            prompts[category], 
                            response_types[category]
                        ))
                        
                        # Process results
                        for batch_result in batch_results:
                            if batch_result['success']:
                                try:
                                    # Convert analysis to DataFrame
                                    if response_types[category] == 'csv':
                                        df = pd.read_csv(StringIO(batch_result['data']['csv_data']))
                                    else:
                                        # For JSON responses, extract plate data
                                        analysis = batch_result['data']
                                        if 'raw_data_indices' in analysis:
                                            df = extract_plate_block(batch_result['file_path'], analysis, 'data')
                                        elif 'raw_mapping_indices' in analysis:
                                            df = extract_plate_block(batch_result['file_path'], analysis, 'map')
                                        else:
                                            logger.error(f"Unknown analysis type for {batch_result['file_path']}")
                                            continue
                                    
                                    # Convert Plate ID to string
                                    df['Plate ID'] = df['Plate ID'].astype(str)
                                    category_dfs.append(df)
                                    logger.info(f"    Extracted {len(df)} records from {os.path.basename(batch_result['file_path'])}")
                                    
                                except Exception as e:
                                    logger.error(f"    Failed to process result from {os.path.basename(batch_result['file_path'])}: {str(e)}")
                                    continue
                            else:
                                logger.error(f"    Failed to analyze {os.path.basename(batch_result['file_path'])}: {batch_result['error']}")
                                
                    except Exception as e:
                        logger.error(f"  Batch processing failed for {category}: {str(e)}")
                        # Fallback to sequential processing
                        logger.info(f"  Falling back to sequential processing...")
                        for file_path, _ in cat_files:
                            try:
                                df = process_file(file_path, prompts[category], response_types[category])
                                category_dfs.append(df)
                                logger.info(f"    Extracted {len(df)} records from {os.path.basename(file_path)}")
                            except Exception as e:
                                logger.error(f"    Failed to process {os.path.basename(file_path)}: {str(e)}")
                                continue
                else:
                    # Single file or batch processing disabled - use regular processing
                    processing_mode = "single file" if len(file_paths) == 1 else "sequential (batch disabled)"
                    logger.info(f"  Processing {len(file_paths)} {category} file(s) using {processing_mode} mode...")
                    for file_path, _ in cat_files:
                        try:
                            df = process_file(file_path, prompts[category], response_types[category])
                            category_dfs.append(df)
                            logger.info(f"    Extracted {len(df)} records from {os.path.basename(file_path)}")
                        except Exception as e:
                            logger.error(f"    Failed to process {os.path.basename(file_path)}: {str(e)}")
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
        raise

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configure batch processing settings
    batch_config.max_concurrent_requests = args.max_concurrent
    batch_config.max_requests_per_minute = args.max_requests_per_minute
    batch_config.max_tokens_per_minute = args.max_tokens_per_minute
    batch_config.request_timeout = args.request_timeout
    batch_config.retry_attempts = args.retry_attempts
    
    # Update rate limiter with new settings
    rate_limiter = RateLimiter(batch_config.max_requests_per_minute, batch_config.max_tokens_per_minute)
    
    logger.info(f"Batch processing configuration:")
    logger.info(f"  - Max concurrent requests: {batch_config.max_concurrent_requests}")
    logger.info(f"  - Max requests per minute: {batch_config.max_requests_per_minute}")
    logger.info(f"  - Max tokens per minute: {batch_config.max_tokens_per_minute}")
    logger.info(f"  - Request timeout: {batch_config.request_timeout}s")
    logger.info(f"  - Retry attempts: {batch_config.retry_attempts}")
    logger.info(f"  - Batch processing: {'disabled' if args.disable_batch else 'enabled'}")
    
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
        main_workflow(files, args.output, args.dry_run, not args.disable_batch)
        
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