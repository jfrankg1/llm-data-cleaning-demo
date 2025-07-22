"""
Claude API Service - Centralized Claude API communication
Extracted from src/dsaas2.py as part of Phase 2 refactoring
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union
from anthropic import Anthropic
from dotenv import load_dotenv

# Import error handling
try:
    from src.error_handler import (
        ProcessingError, ErrorCategory, ErrorSeverity, with_error_handling
    )
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class ClaudeAPIService:
    """Centralized service for Claude API interactions"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude API service
        
        Args:
            api_key: Optional API key override
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in environment or passed directly")
        
        self.anthropic = Anthropic(api_key=self.api_key)
        
        # System prompt for Claude
        self.SYSTEM_PROMPT = """You are a detail-oriented, precise, and patient researcher. Your long years of experience have taught you that doing the job properly the first time 
is more valuable than anything else, so you do not guess. While some may even call you pedantic, everyone knows that you only make decisions in your job that are logical, 
rational, and supported by the documents you are given."""

    def send_to_claude(
        self,
        content: Union[str, Dict[str, Any]],
        prompt: str,
        model: str = "claude-4-sonnet-20250514",
        max_tokens: int = 5000,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Sends content to Claude API and gets response back.
        
        Args:
            content: Content to send to Claude (string or dict from file processing)
            prompt: The specific prompt to use (CATEGORIZER_PROMPT, MAP_PROMPT, etc.)
            model: The Claude model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter for response generation
            system_prompt: Override system prompt (optional)
            organization_id: Organization ID for usage tracking (optional)
            user_id: User ID for usage tracking (optional)
            
        Returns:
            str: Claude's response
            
        Raises:
            ValueError: If required parameters are missing
            ProcessingError: If API call fails after retries
        """
        if not prompt:
            raise ValueError("A specific prompt must be provided")
        
        # Use provided system prompt or default
        system = system_prompt or self.SYSTEM_PROMPT
        
        try:
            # Check rate limits if tracking is available
            self._check_rate_limits(user_id, organization_id)
            
            # Format content for Claude
            formatted_content = self._format_content(content)
            
            # Construct the full message
            full_prompt = f"{prompt}\n\nContent:\n{formatted_content}"
            
            # Make API call
            response = self._make_api_call(full_prompt, system, model, max_tokens, temperature)
            
            # Track usage if context provided
            self._track_usage(organization_id, user_id)
            
            return response
            
        except Exception as e:
            return self._handle_api_error(e, model, formatted_content, full_prompt, system, max_tokens, temperature)

    def analyze_with_claude(
        self,
        content: Union[str, Dict[str, Any]],
        analysis_type: str,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Analyze content with Claude using predefined prompts
        
        Args:
            content: Content to analyze
            analysis_type: Type of analysis ('categorize', 'map', 'data', 'protocol')
            organization_id: Organization ID for tracking
            user_id: User ID for tracking
            
        Returns:
            str: Claude's analysis response
        """
        # Get the appropriate prompt for analysis type
        prompt = self._get_analysis_prompt(analysis_type)
        
        # Set analysis-type-specific max_tokens
        max_tokens = self._get_max_tokens_for_analysis_type(analysis_type)
        
        return self.send_to_claude(
            content=content,
            prompt=prompt,
            max_tokens=max_tokens,
            organization_id=organization_id,
            user_id=user_id
        )

    def validate_api_key(self) -> bool:
        """Validate that the API key works
        
        Returns:
            bool: True if API key is valid
        """
        try:
            # Simple test call to validate API key
            self.anthropic.messages.create(
                model="claude-4-sonnet-20250514",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            return False

    def _check_rate_limits(self, user_id: Optional[str], organization_id: Optional[str]) -> None:
        """Check rate limits for user and organization"""
        if not (user_id and organization_id):
            return
            
        try:
            from auth.rate_limiter import rate_limiter
            
            # Check user API limits
            allowed, info = rate_limiter.check_user_api_limit(user_id)
            if not allowed:
                reset_time = datetime.fromtimestamp(info['reset_time']).strftime('%H:%M:%S')
                raise Exception(f"User API rate limit exceeded. Limit: {info['limit']}/hour. Try again at {reset_time}")
            
            # Check organization API limits
            subscription_tier = 'free'  # Default, could be passed as parameter
            allowed, info = rate_limiter.check_org_api_limit(organization_id, subscription_tier)
            if not allowed:
                reset_time = datetime.fromtimestamp(info['reset_time']).strftime('%H:%M:%S')
                raise Exception(f"Organization API rate limit exceeded. Limit: {info['limit']}/hour. Try again at {reset_time}")
                
        except ImportError:
            # Rate limiting not available, continue without it
            logger.debug("Rate limiting not available, skipping rate limit checks")

    def _format_content(self, content: Union[str, Dict[str, Any]]) -> str:
        """Format content for Claude API"""
        if isinstance(content, dict):
            # If content is a dictionary from process_file_for_claude
            formatted = f"File: {content['filename']}\n"
            if content.get('is_binary', False):
                formatted += f"Content (base64 encoded): {content['content'][:1000]}...\n"
            else:
                formatted += f"Content:\n{content['content']}\n"
            return formatted
        else:
            # If content is a plain string
            return str(content)

    def _make_api_call(
        self,
        full_prompt: str,
        system: str,
        model: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Make the actual API call to Claude"""
        message = self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return message.content[0].text

    def _track_usage(self, organization_id: Optional[str], user_id: Optional[str]) -> None:
        """Track API usage if tracking is available"""
        if not (organization_id and user_id):
            return
            
        try:
            from auth.database import track_usage
            track_usage(organization_id, user_id, 'api_call')
            logger.debug(f"API call tracked for org {organization_id[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to track API usage: {e}")

    def _handle_api_error(
        self,
        error: Exception,
        model: str,
        formatted_content: str,
        full_prompt: str,
        system: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Handle API errors with retry logic"""
        if ERROR_HANDLING_AVAILABLE:
            # Use the robust error handling system
            api_error = ProcessingError(
                f"Claude API error: {str(error)}",
                category=ErrorCategory.API_ERROR,
                severity=ErrorSeverity.HIGH,
                details={'model': model, 'content_length': len(formatted_content)},
                original_exception=error
            )
            
            # Try recovery (includes retry logic)
            from src.error_handler import RobustErrorHandler
            error_handler = RobustErrorHandler()
            recovery_result = error_handler.handle_error(
                api_error,
                {'retry_count': 0, 'allow_degraded': True}
            )
            
            if recovery_result and isinstance(recovery_result, dict) and recovery_result.get('retry'):
                # Retry the API call once
                try:
                    return self._make_api_call(full_prompt, system, model, max_tokens, temperature)
                except Exception:
                    pass
            
            logger.error(f"Claude API communication failed: {str(error)}")
            raise api_error
        else:
            # Fallback error handling without robust system
            logger.error(f"Claude API communication failed: {str(error)}")
            raise Exception(f"Claude API error: {str(error)}")

    def _get_analysis_prompt(self, analysis_type: str) -> str:
        """Get the appropriate prompt for analysis type"""
        prompts = {
            'categorize': self._get_categorizer_prompt(),
            'categorize_extended': self._get_extended_categorizer_prompt(),
            'map': self._get_map_prompt(),
            'data': self._get_data_prompt(),
            'protocol': self._get_protocol_prompt(),
            'time_series': self._get_time_series_prompt()
        }
        
        if analysis_type not in prompts:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        return prompts[analysis_type]
    
    def _get_max_tokens_for_analysis_type(self, analysis_type: str) -> int:
        """Get appropriate max_tokens for different analysis types"""
        token_limits = {
            'categorize': 1000,   # Simple categorization
            'categorize_extended': 1500,  # Extended categorization with logs
            'map': 5000,          # Mapping analysis
            'data': 5000,         # Data analysis
            'protocol': 12000,    # Protocol needs more tokens for JSON-only output
            'time_series': 8000   # Time-series analysis with detailed JSON
        }
        
        return token_limits.get(analysis_type, 5000)  # Default to 5000

    def _get_categorizer_prompt(self) -> str:
        """Get the categorizer prompt"""
        return """
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

    def _get_extended_categorizer_prompt(self) -> str:
        """Get the extended categorizer prompt that includes instrument log detection"""
        return """
        You will be given a list of files along with their content and metadata. 
        Your task is to review this information and categorize each file into one of five categories based on its content and purpose. 
        Here is the list of files with their content and metadata:

        <file_list>
        {file_content}
        </file_list>

        Your task is to categorize each file into one of the following categories:
        1. "data" - for files containing experimental data from 96-well or 384-well plates
        2. "map" - for files mapping sample locations to sample identifiers (plate maps)
        3. "protocol" - for files specifying experimental protocol(s)
        4. "log" - for instrument log files containing time-series or event-based data from IoT devices, automated equipment, or monitoring systems
        5. "other" - for files containing some other type of information

        Specific guidance for "log" category:
        - Look for files with timestamp columns, sensor readings, equipment status updates
        - Common log file indicators: timestamps, sequential measurements, system events, monitoring data
        - May include temperature logs, pressure readings, flow rates, equipment status, error logs
        - Can be from manufacturing equipment, laboratory instruments, environmental monitoring systems
        - May contain irregular intervals, event-triggered entries, or regular time-series data

        Instructions:
        1. Carefully review the content and metadata of each file in the list.
        2. Based on the information provided, determine which category best describes each file.
        3. Categorize each file into one and only one category.
        4. In cases where the categorization is ambiguous or unclear, use the "other" category.
        5. Do not guess or make assumptions about the file's content or purpose if it's not clearly evident from the provided information.
        6. Pay special attention to files that may contain instrument logs, sensor data, or equipment monitoring records.

        Output your categorization as a JSON object, where each key is a filename and its value is the corresponding category. Use the following category labels exactly as written: "data", "map", "protocol", "log", "other". Provide no other output.

        Example output format:
        {{
        "plate_data.csv": "data",
        "sample_map.csv": "map",
        "protocol.docx": "protocol",
        "equipment_log.csv": "log",
        "notes.txt": "other"
        }}

        Remember:
        - Each file must be categorized into one and only one category.
        - If you're unsure about a file's category, use "other" rather than guessing.
        - Provide your final answer as a single JSON object inside <answer> tags.
        - Do not include any other text or comments in your response.
        """

    def _get_time_series_prompt(self) -> str:
        """Get the time-series analysis prompt from external file"""
        try:
            from pathlib import Path
            prompt_path = Path(__file__).parent.parent / "prompts" / "time_series_alignment.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load time-series prompt: {e}")
            # Fallback prompt
            return """
            Analyze the provided log files and determine their structure.
            For each file, identify:
            1. Whether it's event-based or time-based
            2. Timestamp column location
            3. Data structure indices
            
            Output results in JSON format with 0-based indices.
            """

    def _get_map_prompt(self) -> str:
        """Get the map analysis prompt"""
        return """
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

    def _get_data_prompt(self) -> str:
        """Get the data analysis prompt"""
        return """
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

    def _get_protocol_prompt(self) -> str:
        """Get the protocol analysis prompt"""
        return """
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

IMPORTANT: Your response must contain ONLY the JSON output. Do not include any explanatory text, markdown formatting, or code blocks. Return only the raw JSON structure that maps each well to its experimental conditions and protocols.
"""


# Convenience functions for backward compatibility and global access
_claude_service_instance = None

def get_claude_service() -> ClaudeAPIService:
    """Get singleton instance of Claude API service"""
    global _claude_service_instance
    if _claude_service_instance is None:
        _claude_service_instance = ClaudeAPIService()
    return _claude_service_instance

def send_to_claude(
    content: Union[str, Dict[str, Any]],
    prompt: str,
    model: str = "claude-4-sonnet-20250514",
    max_tokens: int = 5000,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> str:
    """Convenience function for backward compatibility"""
    service = get_claude_service()
    return service.send_to_claude(
        content=content,
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        organization_id=organization_id,
        user_id=user_id
    )

def analyze_with_claude(
    content: Union[str, Dict[str, Any]],
    analysis_type: str,
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> str:
    """Convenience function for backward compatibility"""
    service = get_claude_service()
    return service.analyze_with_claude(
        content=content,
        analysis_type=analysis_type,
        organization_id=organization_id,
        user_id=user_id
    )

def validate_api_key() -> bool:
    """Convenience function for backward compatibility"""
    service = get_claude_service()
    return service.validate_api_key()