#!/usr/bin/env python3
"""
Admin Interface Input Validation
Provides validation helpers specifically for admin interfaces

Author: Claude AI  
Created: 2025-07-18
"""

import streamlit as st
from typing import Optional, Tuple
from .input_validation import validate_and_sanitize_input, ValidationResult


def validate_admin_search_input(search_value: str, field_name: str = "search") -> Tuple[bool, str]:
    """
    Validate admin search input and display errors if invalid.
    
    Args:
        search_value: The search input to validate
        field_name: Name of the field for error messages
        
    Returns:
        Tuple of (is_valid, sanitized_value)
    """
    if not search_value:
        return True, ""
    
    validation_result = validate_and_sanitize_input('search', search_value)
    
    if not validation_result.is_valid:
        for error in validation_result.errors:
            st.error(f"❌ {field_name.title()}: {error}")
        return False, ""
    
    if validation_result.warnings:
        for warning in validation_result.warnings:
            st.warning(f"⚠️ {field_name.title()}: {warning}")
    
    return True, validation_result.sanitized_value


def validate_admin_text_input(text_value: str, input_type: str, field_name: str) -> Tuple[bool, str]:
    """
    Validate admin text input and display errors if invalid.
    
    Args:
        text_value: The text input to validate
        input_type: Type of validation ('organization', 'user_name', 'email', 'url', etc.)
        field_name: Name of the field for error messages
        
    Returns:
        Tuple of (is_valid, sanitized_value)
    """
    if not text_value:
        return True, ""
    
    validation_result = validate_and_sanitize_input(input_type, text_value)
    
    if not validation_result.is_valid:
        for error in validation_result.errors:
            st.error(f"❌ {field_name}: {error}")
        return False, ""
    
    if validation_result.warnings:
        for warning in validation_result.warnings:
            st.warning(f"⚠️ {field_name}: {warning}")
    
    return True, validation_result.sanitized_value


def safe_admin_text_input(label: str, value: str = "", input_type: str = "search", 
                         max_chars: Optional[int] = None, key: Optional[str] = None) -> str:
    """
    Create a validated text input field for admin interfaces.
    
    Args:
        label: Label for the input field
        value: Default value
        input_type: Type of validation to apply
        max_chars: Maximum character limit (optional)
        key: Streamlit key for the widget
        
    Returns:
        Validated and sanitized input value
    """
    # Create the input field
    user_input = st.text_input(label, value=value, max_chars=max_chars, key=key)
    
    # Validate the input
    is_valid, sanitized_value = validate_admin_text_input(user_input, input_type, label)
    
    # Return sanitized value if valid, empty string if invalid
    return sanitized_value if is_valid else ""


def safe_admin_selectbox(label: str, options: list, index: int = 0, 
                        key: Optional[str] = None) -> str:
    """
    Create a safe selectbox for admin interfaces (selectbox values are controlled, so minimal validation needed).
    
    Args:
        label: Label for the selectbox
        options: List of valid options
        index: Default selected index
        key: Streamlit key for the widget
        
    Returns:
        Selected value (no sanitization needed for controlled options)
    """
    return st.selectbox(label, options, index=index, key=key)


def display_validation_errors(validation_result: ValidationResult, field_name: str) -> bool:
    """
    Display validation errors and warnings in Streamlit interface.
    
    Args:
        validation_result: Result from validation
        field_name: Name of the field for error messages
        
    Returns:
        True if valid, False if errors exist
    """
    if not validation_result.is_valid:
        for error in validation_result.errors:
            st.error(f"❌ {field_name}: {error}")
    
    if validation_result.warnings:
        for warning in validation_result.warnings:
            st.warning(f"⚠️ {field_name}: {warning}")
    
    return validation_result.is_valid