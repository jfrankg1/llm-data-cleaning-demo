#!/usr/bin/env python3
"""
Filename Sanitization System - Fix 2
Addresses filename path issues with special characters and long names
"""

import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

class FilenameSanitizer:
    """
    Handles filename sanitization for cross-platform compatibility
    and resolves path issues with special characters.
    """
    
    def __init__(self, max_length: int = 200):
        self.max_length = max_length
        
        # Characters that are problematic in filenames
        self.problematic_chars = {
            '<': '_lt_',
            '>': '_gt_', 
            ':': '_colon_',
            '"': '_quote_',
            '/': '_slash_',
            '\\': '_backslash_',
            '|': '_pipe_',
            '?': '_question_',
            '*': '_star_',
            '+': '_plus_',
            '#': '_hash_',
            '%': '_percent_',
            '&': '_and_',
            '{': '_lbrace_',
            '}': '_rbrace_',
            '[': '_lbracket_',
            ']': '_rbracket_',
            '(': '_lparen_',
            ')': '_rparen_',
            '=': '_equals_',
            '!': '_exclamation_',
            '@': '_at_',
            '$': '_dollar_',
            '^': '_caret_',
            '`': '_backtick_',
            '~': '_tilde_',
            ';': '_semicolon_',
            "'": '_apostrophe_'
        }
        
        # Additional replacements for readability
        self.replacements = {
            ' ': '_',
            ',': '_',
            '.': '_dot_',
            '–': '_dash_',
            '—': '_dash_',
            '…': '_ellipsis_'
        }
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename for cross-platform compatibility.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for file system use
        """
        if not filename:
            return "unnamed_file"
        
        # Step 1: Unicode normalization
        filename = unicodedata.normalize('NFKD', filename)
        
        # Step 2: Replace problematic characters
        for char, replacement in self.problematic_chars.items():
            filename = filename.replace(char, replacement)
        
        # Step 3: Replace additional characters for readability
        for char, replacement in self.replacements.items():
            filename = filename.replace(char, replacement)
        
        # Step 4: Remove or replace any remaining non-ASCII characters
        filename = self._handle_unicode_chars(filename)
        
        # Step 5: Collapse multiple underscores
        filename = re.sub(r'_+', '_', filename)
        
        # Step 6: Remove leading/trailing underscores and dots
        filename = filename.strip('_.')
        
        # Step 7: Ensure it doesn't start with a number (some systems don't like this)
        if filename and filename[0].isdigit():
            filename = 'file_' + filename
        
        # Step 8: Limit length while preserving extension
        filename = self._limit_length(filename)
        
        # Step 9: Ensure it's not empty
        if not filename:
            filename = "sanitized_file"
        
        return filename
    
    def _handle_unicode_chars(self, filename: str) -> str:
        """Handle Unicode characters in filenames"""
        result = ""
        for char in filename:
            if ord(char) < 128:  # ASCII
                result += char
            else:
                # Replace common Unicode characters with readable equivalents
                unicode_replacements = {
                    'α': 'alpha',
                    'β': 'beta', 
                    'γ': 'gamma',
                    'δ': 'delta',
                    'μ': 'mu',
                    'π': 'pi',
                    'σ': 'sigma',
                    'τ': 'tau',
                    'φ': 'phi',
                    'χ': 'chi',
                    'ψ': 'psi',
                    'ω': 'omega',
                    '°': 'deg',
                    '±': 'plusminus',
                    '≤': 'lte',
                    '≥': 'gte',
                    '≠': 'neq',
                    '≈': 'approx',
                    '∞': 'inf',
                    '∑': 'sum',
                    '∏': 'prod',
                    '∫': 'integral',
                    '∂': 'partial',
                    '∇': 'nabla',
                    '√': 'sqrt',
                    '∝': 'proportional',
                    '∈': 'in',
                    '∉': 'notin',
                    '⊂': 'subset',
                    '⊃': 'superset',
                    '⊆': 'subseteq',
                    '⊇': 'superseteq',
                    '∪': 'union',
                    '∩': 'intersection',
                    '¹': '1',
                    '²': '2',
                    '³': '3',
                    '⁴': '4',
                    '⁵': '5',
                    '⁶': '6',
                    '⁷': '7',
                    '⁸': '8',
                    '⁹': '9',
                    '⁰': '0',
                    '₁': '1',
                    '₂': '2',
                    '₃': '3',
                    '₄': '4',
                    '₅': '5',
                    '₆': '6',
                    '₇': '7',
                    '₈': '8',
                    '₉': '9',
                    '₀': '0'
                }
                
                if char in unicode_replacements:
                    result += unicode_replacements[char]
                else:
                    # For other Unicode characters, use their Unicode name or skip
                    try:
                        char_name = unicodedata.name(char).lower().replace(' ', '_')
                        if len(char_name) <= 10:  # Only use short names
                            result += char_name
                        else:
                            result += 'unicode_char'
                    except:
                        result += 'unicode_char'
        
        return result
    
    def _limit_length(self, filename: str) -> str:
        """Limit filename length while preserving extension"""
        if len(filename) <= self.max_length:
            return filename
        
        # Split into name and extension
        if '.' in filename:
            parts = filename.rsplit('.', 1)
            name, ext = parts[0], '.' + parts[1]
        else:
            name, ext = filename, ''
        
        # Calculate available space for name
        available_length = self.max_length - len(ext) - 10  # Leave some buffer
        
        if available_length > 0:
            # Truncate name but try to keep meaningful parts
            truncated_name = name[:available_length]
            # Try to break at word boundaries
            if '_' in truncated_name:
                last_underscore = truncated_name.rfind('_')
                if last_underscore > available_length * 0.7:  # If we don't lose too much
                    truncated_name = truncated_name[:last_underscore]
            
            return truncated_name + ext
        else:
            # Extension is too long, just return a short name
            return filename[:self.max_length]
    
    def sanitize_path(self, filepath: str, create_dirs: bool = True) -> str:
        """
        Sanitize a full file path and optionally create directories.
        
        Args:
            filepath: Original file path
            create_dirs: Whether to create parent directories
            
        Returns:
            Sanitized file path
        """
        path = Path(filepath)
        
        # Sanitize directory parts
        sanitized_parts = []
        for part in path.parts:
            if part in ['', '.', '..']:
                sanitized_parts.append(part)
            else:
                sanitized_parts.append(self.sanitize_filename(part))
        
        # Reconstruct path
        sanitized_path = Path(*sanitized_parts)
        
        # Create parent directories if requested
        if create_dirs and sanitized_path.parent:
            try:
                sanitized_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory {sanitized_path.parent}: {e}")
        
        return str(sanitized_path)
    
    def get_unique_path(self, filepath: str) -> str:
        """
        Get a unique file path by appending numbers if file exists.
        
        Args:
            filepath: Desired file path
            
        Returns:
            Unique file path that doesn't exist
        """
        path = Path(filepath)
        
        if not path.exists():
            return filepath
        
        # Split into name and extension
        if path.suffix:
            stem = path.stem
            suffix = path.suffix
        else:
            stem = path.name
            suffix = ''
        
        parent = path.parent
        counter = 1
        
        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = parent / new_name
            
            if not new_path.exists():
                return str(new_path)
            
            counter += 1
            
            # Prevent infinite loop
            if counter > 1000:
                import time
                timestamp = int(time.time())
                new_name = f"{stem}_{timestamp}{suffix}"
                return str(parent / new_name)


def sanitize_test_filename(test_name: str) -> str:
    """
    Convenience function to sanitize test case names for file output.
    Specifically designed for the test runner.
    """
    sanitizer = FilenameSanitizer(max_length=100)  # Shorter for test files
    
    # Pre-process common test name patterns
    test_name = test_name.replace('(', '_').replace(')', '_')
    test_name = test_name.replace('#', 'num').replace('/', '_or_')
    
    return sanitizer.sanitize_filename(test_name)


if __name__ == "__main__":
    # Test the sanitizer with problematic filenames from test results
    sanitizer = FilenameSanitizer()
    
    test_cases = [
        "Mixed ID formats (S###, Sample_###, CTRL+/-, Blank)",
        "Missing values (empty, N/A, NaN, null, etc.)",
        "Unicode characters and special symbols",
        "test_unified_Mixed_ID_formats_(S###,_Sample_###,_CTRL+",
        "file with spaces and (parentheses)",
        "file#with%special&chars",
        "αβγδμ-unicode-test",
        "very_long_filename_that_exceeds_normal_length_limits_and_should_be_truncated_properly_while_maintaining_readability.csv",
        ""  # Empty string test
    ]
    
    print("Filename Sanitization Test Results:")
    print("=" * 60)
    
    for original in test_cases:
        sanitized = sanitizer.sanitize_filename(original)
        print(f"Original:  '{original}'")
        print(f"Sanitized: '{sanitized}'")
        print(f"Length:    {len(sanitized)}")
        print("-" * 40)
    
    # Test path sanitization
    print("\nPath Sanitization Test:")
    problematic_path = "test_results/test_unified_Mixed_ID_formats_(S###,_Sample_###,_CTRL+.csv"
    sanitized_path = sanitizer.sanitize_path(problematic_path, create_dirs=False)
    print(f"Original path:  {problematic_path}")
    print(f"Sanitized path: {sanitized_path}")
    
    # Test unique path generation
    print(f"Unique path:    {sanitizer.get_unique_path(sanitized_path)}")