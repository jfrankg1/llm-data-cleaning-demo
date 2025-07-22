#!/usr/bin/env python3
"""
Unicode and Encoding Management System
Implements Phase 1 TODO item #10 for robust Unicode handling in scientific data processing
"""

import re
import unicodedata
import chardet
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import codecs

# Set up logging
logger = logging.getLogger(__name__)


class UnicodeProcessor:
    """
    Comprehensive Unicode and encoding management for scientific data.
    Handles special characters, Greek letters, international symbols, and numeric Unicode.
    """
    
    def __init__(self):
        # Common encodings in order of preference
        self.common_encodings = [
            'utf-8',
            'utf-16',
            'utf-16le',
            'utf-16be',
            'latin-1',
            'iso-8859-1',
            'cp1252',
            'windows-1252',
            'ascii'
        ]
        
        # Scientific character mappings
        self.scientific_char_map = {
            # Greek letters commonly used in science
            'α': 'alpha',
            'β': 'beta', 
            'γ': 'gamma',
            'δ': 'delta',
            'ε': 'epsilon',
            'ζ': 'zeta',
            'η': 'eta',
            'θ': 'theta',
            'ι': 'iota',
            'κ': 'kappa',
            'λ': 'lambda',
            'μ': 'mu',
            'ν': 'nu',
            'ξ': 'xi',
            'ο': 'omicron',
            'π': 'pi',
            'ρ': 'rho',
            'σ': 'sigma',
            'τ': 'tau',
            'υ': 'upsilon',
            'φ': 'phi',
            'χ': 'chi',
            'ψ': 'psi',
            'ω': 'omega',
            
            # Uppercase Greek letters
            'Α': 'Alpha',
            'Β': 'Beta',
            'Γ': 'Gamma',
            'Δ': 'Delta',
            'Ε': 'Epsilon',
            'Ζ': 'Zeta',
            'Η': 'Eta',
            'Θ': 'Theta',
            'Ι': 'Iota',
            'Κ': 'Kappa',
            'Λ': 'Lambda',
            'Μ': 'Mu',
            'Ν': 'Nu',
            'Ξ': 'Xi',
            'Ο': 'Omicron',
            'Π': 'Pi',
            'Ρ': 'Rho',
            'Σ': 'Sigma',
            'Τ': 'Tau',
            'Υ': 'Upsilon',
            'Φ': 'Phi',
            'Χ': 'Chi',
            'Ψ': 'Psi',
            'Ω': 'Omega',
            
            # Common scientific symbols
            '±': '+/-',
            '∓': '-/+',
            '×': 'x',
            '÷': '/',
            '°': 'deg',
            '℃': 'C',
            '℉': 'F',
            'Å': 'Angstrom',
            'ñ': 'n',
            'ë': 'e',
            'ï': 'i',
            'ö': 'o',
            'ü': 'u',
            'ā': 'a',
            'ē': 'e',
            'ī': 'i',
            'ō': 'o',
            'ū': 'u',
            
            # Mathematical symbols
            '∞': 'infinity',
            '≈': 'approx',
            '≤': '<=',
            '≥': '>=',
            '≠': '!=',
            '∝': 'proportional',
            '∑': 'sum',
            '∏': 'product',
            '∫': 'integral',
            '∂': 'partial',
            '∇': 'nabla',
            '√': 'sqrt',
            
            # Currency and misc
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '©': '(c)',
            '®': '(R)',
            '™': '(TM)',
            '§': 'section',
            '¶': 'paragraph',
        }
        
        # Superscript mappings
        self.superscript_map = {
            '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
            '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
            '⁺': '+', '⁻': '-', '⁼': '=', '⁽': '(', '⁾': ')',
            'ⁿ': 'n', 'ⁱ': 'i', 'ᵃ': 'a', 'ᵇ': 'b', 'ᶜ': 'c',
            'ᵈ': 'd', 'ᵉ': 'e', 'ᶠ': 'f', 'ᵍ': 'g', 'ʰ': 'h',
            'ʲ': 'j', 'ᵏ': 'k', 'ˡ': 'l', 'ᵐ': 'm', 'ᵒ': 'o',
            'ᵖ': 'p', 'ʳ': 'r', 'ˢ': 's', 'ᵗ': 't', 'ᵘ': 'u',
            'ᵛ': 'v', 'ʷ': 'w', 'ˣ': 'x', 'ʸ': 'y', 'ᶻ': 'z'
        }
        
        # Subscript mappings
        self.subscript_map = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
            '₊': '+', '₋': '-', '₌': '=', '₍': '(', '₎': ')',
            'ₐ': 'a', 'ₑ': 'e', 'ₕ': 'h', 'ᵢ': 'i', 'ⱼ': 'j',
            'ₖ': 'k', 'ₗ': 'l', 'ₘ': 'm', 'ₙ': 'n', 'ₒ': 'o',
            'ₚ': 'p', 'ᵣ': 'r', 'ₛ': 's', 'ₜ': 't', 'ᵤ': 'u',
            'ᵥ': 'v', 'ₓ': 'x'
        }
        
        # Fraction mappings
        self.fraction_map = {
            '½': '1/2', '⅓': '1/3', '⅔': '2/3', '¼': '1/4', '¾': '3/4',
            '⅕': '1/5', '⅖': '2/5', '⅗': '3/5', '⅘': '4/5', '⅙': '1/6',
            '⅚': '5/6', '⅐': '1/7', '⅛': '1/8', '⅜': '3/8', '⅝': '5/8',
            '⅞': '7/8', '⅑': '1/9', '⅒': '1/10'
        }
    
    def detect_encoding(self, file_path: Union[str, Path], sample_size: int = 50000) -> Tuple[str, float]:
        """
        Detect file encoding using multiple methods.
        
        Args:
            file_path: Path to the file
            sample_size: Number of bytes to sample for detection
            
        Returns:
            Tuple of (encoding, confidence)
        """
        file_path = Path(file_path)
        
        try:
            # Read sample of file
            with open(file_path, 'rb') as f:
                sample = f.read(sample_size)
            
            if not sample:
                return 'utf-8', 1.0
            
            # Use chardet for detection
            detected = chardet.detect(sample)
            encoding = detected.get('encoding', 'utf-8')
            confidence = detected.get('confidence', 0.0)
            
            logger.info(f"Detected encoding for {file_path.name}: {encoding} (confidence: {confidence:.2f})")
            
            # Validate the detected encoding by trying to decode
            if encoding and confidence > 0.7:
                try:
                    sample.decode(encoding)
                    return encoding, confidence
                except (UnicodeDecodeError, LookupError):
                    logger.warning(f"Detected encoding {encoding} failed validation")
            
            # Fallback: try common encodings
            for test_encoding in self.common_encodings:
                try:
                    sample.decode(test_encoding)
                    logger.info(f"Fallback encoding selected: {test_encoding}")
                    return test_encoding, 0.8
                except (UnicodeDecodeError, LookupError):
                    continue
            
            # Last resort
            logger.warning(f"Could not determine encoding for {file_path.name}, using utf-8 with error handling")
            return 'utf-8', 0.1
            
        except Exception as e:
            logger.error(f"Error detecting encoding for {file_path}: {str(e)}")
            return 'utf-8', 0.1
    
    def read_file_with_encoding(self, file_path: Union[str, Path], encoding: Optional[str] = None) -> Tuple[str, str]:
        """
        Read file with robust encoding handling.
        
        Args:
            file_path: Path to the file
            encoding: Specific encoding to use (optional)
            
        Returns:
            Tuple of (content, encoding_used)
        """
        file_path = Path(file_path)
        
        if encoding:
            encodings_to_try = [encoding]
        else:
            detected_encoding, confidence = self.detect_encoding(file_path)
            encodings_to_try = [detected_encoding] + [enc for enc in self.common_encodings if enc != detected_encoding]
        
        for enc in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=enc, errors='strict') as f:
                    content = f.read()
                logger.info(f"Successfully read {file_path.name} with encoding: {enc}")
                return content, enc
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.warning(f"Error reading {file_path.name} with encoding {enc}: {str(e)}")
                continue
        
        # Final fallback with error replacement
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            logger.warning(f"Read {file_path.name} with UTF-8 and error replacement")
            return content, 'utf-8-with-replacement'
        except Exception as e:
            raise ValueError(f"Could not read file {file_path}: {str(e)}")
    
    def normalize_unicode(self, text: str, form: str = 'NFKC') -> str:
        """
        Normalize Unicode text using specified normalization form.
        
        Args:
            text: Input text to normalize
            form: Unicode normalization form (NFC, NFD, NFKC, NFKD)
            
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ''
        
        try:
            normalized = unicodedata.normalize(form, text)
            return normalized
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {str(e)}")
            return text
    
    def convert_scientific_characters(self, text: str, preserve_original: bool = False) -> str:
        """
        Convert scientific Unicode characters to ASCII equivalents.
        
        Args:
            text: Input text containing scientific characters
            preserve_original: If True, append original in parentheses
            
        Returns:
            Text with converted characters
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ''
        
        converted_text = text
        
        # Convert scientific symbols
        for unicode_char, ascii_equiv in self.scientific_char_map.items():
            if unicode_char in converted_text:
                if preserve_original:
                    replacement = f"{ascii_equiv}({unicode_char})"
                else:
                    replacement = ascii_equiv
                converted_text = converted_text.replace(unicode_char, replacement)
        
        return converted_text
    
    def convert_numeric_unicode(self, text: str) -> str:
        """
        Convert Unicode superscripts, subscripts, and fractions to ASCII.
        
        Args:
            text: Input text containing numeric Unicode
            
        Returns:
            Text with converted numeric Unicode
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ''
        
        converted_text = text
        
        # Convert superscripts
        for unicode_char, ascii_equiv in self.superscript_map.items():
            if unicode_char in converted_text:
                # Mark as superscript
                converted_text = converted_text.replace(unicode_char, f"^{ascii_equiv}")
        
        # Convert subscripts
        for unicode_char, ascii_equiv in self.subscript_map.items():
            if unicode_char in converted_text:
                # Mark as subscript
                converted_text = converted_text.replace(unicode_char, f"_{ascii_equiv}")
        
        # Convert fractions
        for unicode_char, ascii_equiv in self.fraction_map.items():
            if unicode_char in converted_text:
                converted_text = converted_text.replace(unicode_char, ascii_equiv)
        
        return converted_text
    
    def clean_control_characters(self, text: str) -> str:
        """
        Remove or replace problematic control characters.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ''
        
        # Remove most control characters except common ones
        cleaned = ''
        for char in text:
            category = unicodedata.category(char)
            if category.startswith('C'):  # Control characters
                if char in '\t\n\r':  # Keep common whitespace
                    cleaned += char
                elif char in '\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f':  # Replace problematic control chars with space
                    cleaned += ' '
                else:  # Replace other control chars with space
                    cleaned += ' '
            else:
                cleaned += char
        
        # Clean up multiple consecutive spaces, but preserve line breaks and tabs
        # Only collapse spaces that are not newlines or tabs
        lines = cleaned.split('\n')
        processed_lines = []
        for line in lines:
            # For each line, clean up spaces and tabs separately
            parts = line.split('\t')
            cleaned_parts = []
            for part in parts:
                # Clean up multiple spaces in each part
                cleaned_part = re.sub(r' +', ' ', part).strip()
                cleaned_parts.append(cleaned_part)
            processed_lines.append('\t'.join(cleaned_parts))
        
        result = '\n'.join(processed_lines)
        return result.strip()
    
    def process_text_comprehensive(
        self, 
        text: str, 
        normalize: bool = True,
        convert_scientific: bool = True,
        convert_numeric: bool = True,
        clean_control: bool = True,
        preserve_original: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive Unicode text processing.
        
        Args:
            text: Input text to process
            normalize: Apply Unicode normalization
            convert_scientific: Convert scientific characters
            convert_numeric: Convert numeric Unicode
            clean_control: Remove control characters
            preserve_original: Preserve original characters in output
            
        Returns:
            Dictionary with processed text and metadata
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ''
        
        original_text = text
        processed_text = text
        processing_steps = []
        
        try:
            # Step 1: Unicode normalization
            if normalize:
                processed_text = self.normalize_unicode(processed_text)
                processing_steps.append('normalized')
            
            # Step 2: Clean control characters
            if clean_control:
                processed_text = self.clean_control_characters(processed_text)
                processing_steps.append('control_chars_cleaned')
            
            # Step 3: Convert scientific characters
            if convert_scientific:
                processed_text = self.convert_scientific_characters(processed_text, preserve_original)
                processing_steps.append('scientific_chars_converted')
            
            # Step 4: Convert numeric Unicode
            if convert_numeric:
                processed_text = self.convert_numeric_unicode(processed_text)
                processing_steps.append('numeric_unicode_converted')
            
            # Analyze changes
            has_changes = original_text != processed_text
            unicode_chars_found = []
            
            for char in original_text:
                if ord(char) > 127 and char not in unicode_chars_found:
                    unicode_chars_found.append(char)
            
            return {
                'original_text': original_text,
                'processed_text': processed_text,
                'has_changes': has_changes,
                'processing_steps': processing_steps,
                'unicode_chars_found': unicode_chars_found,
                'original_length': len(original_text),
                'processed_length': len(processed_text),
                'ascii_compatible': all(ord(char) < 128 for char in processed_text)
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive text processing: {str(e)}")
            return {
                'original_text': original_text,
                'processed_text': original_text,  # Return original on error
                'has_changes': False,
                'processing_steps': ['error'],
                'unicode_chars_found': [],
                'error': str(e),
                'original_length': len(original_text),
                'processed_length': len(original_text),
                'ascii_compatible': False
            }
    
    def process_file(
        self, 
        file_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None,
        **processing_options
    ) -> Dict[str, Any]:
        """
        Process an entire file with Unicode handling.
        
        Args:
            file_path: Input file path
            output_path: Output file path (optional)
            **processing_options: Options for text processing
            
        Returns:
            Dictionary with processing results and metadata
        """
        file_path = Path(file_path)
        
        try:
            # Read file with encoding detection
            content, encoding_used = self.read_file_with_encoding(file_path)
            
            # Process the content
            processing_result = self.process_text_comprehensive(content, **processing_options)
            
            # Save processed file if output path provided
            if output_path:
                output_path = Path(output_path)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(processing_result['processed_text'])
                logger.info(f"Processed file saved to: {output_path}")
            
            return {
                'file_path': str(file_path),
                'output_path': str(output_path) if output_path else None,
                'encoding_detected': encoding_used,
                'file_size_bytes': file_path.stat().st_size,
                'processing_successful': True,
                **processing_result
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                'file_path': str(file_path),
                'output_path': None,
                'encoding_detected': None,
                'file_size_bytes': 0,
                'processing_successful': False,
                'error': str(e),
                'original_text': '',
                'processed_text': '',
                'has_changes': False,
                'processing_steps': ['error'],
                'unicode_chars_found': [],
                'original_length': 0,
                'processed_length': 0,
                'ascii_compatible': False
            }
    
    def analyze_text_unicode_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze Unicode content in text for reporting.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with Unicode analysis
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ''
        
        analysis = {
            'total_chars': len(text),
            'ascii_chars': 0,
            'unicode_chars': 0,
            'unicode_categories': {},
            'problematic_chars': [],
            'scientific_chars': [],
            'numeric_unicode': [],
            'control_chars': [],
            'needs_processing': False
        }
        
        for char in text:
            char_code = ord(char)
            category = unicodedata.category(char)
            
            if char_code < 128:
                analysis['ascii_chars'] += 1
            else:
                analysis['unicode_chars'] += 1
                analysis['needs_processing'] = True
                
                # Categorize Unicode character
                if category not in analysis['unicode_categories']:
                    analysis['unicode_categories'][category] = 0
                analysis['unicode_categories'][category] += 1
                
                # Check for specific types
                if char in self.scientific_char_map:
                    analysis['scientific_chars'].append(char)
                
                if char in self.superscript_map or char in self.subscript_map or char in self.fraction_map:
                    analysis['numeric_unicode'].append(char)
                
                if category.startswith('C'):
                    analysis['control_chars'].append(char)
                
                # Flag potentially problematic characters
                if category in ['Cc', 'Cf', 'Cn', 'Co', 'Cs']:
                    analysis['problematic_chars'].append(char)
        
        # Remove duplicates
        analysis['scientific_chars'] = list(set(analysis['scientific_chars']))
        analysis['numeric_unicode'] = list(set(analysis['numeric_unicode']))
        analysis['control_chars'] = list(set(analysis['control_chars']))
        analysis['problematic_chars'] = list(set(analysis['problematic_chars']))
        
        return analysis


# Convenience functions for easy integration

def process_scientific_text(text: str, **options) -> str:
    """
    Quick function to process scientific text with Unicode handling.
    
    Args:
        text: Text to process
        **options: Processing options
        
    Returns:
        Processed text
    """
    processor = UnicodeProcessor()
    result = processor.process_text_comprehensive(text, **options)
    return result['processed_text']


def read_scientific_file(file_path: Union[str, Path], encoding: Optional[str] = None) -> str:
    """
    Quick function to read a scientific file with robust encoding detection.
    
    Args:
        file_path: Path to file
        encoding: Specific encoding (optional)
        
    Returns:
        File content as string
    """
    processor = UnicodeProcessor()
    content, _ = processor.read_file_with_encoding(file_path, encoding)
    return content


def detect_file_encoding(file_path: Union[str, Path]) -> Tuple[str, float]:
    """
    Quick function to detect file encoding.
    
    Args:
        file_path: Path to file
        
    Returns:
        Tuple of (encoding, confidence)
    """
    processor = UnicodeProcessor()
    return processor.detect_encoding(file_path)