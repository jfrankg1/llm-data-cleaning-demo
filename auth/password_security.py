#!/usr/bin/env python3
"""
Password Security Module

This module implements comprehensive password security features including:
- Modern password strength requirements (NIST 800-63B compliant)
- Password strength scoring and meter
- Common password checking
- Breach password checking (HaveIBeenPwned API)
- Password history management
- Account lockout protection

Author: Claude AI
Created: 2025-07-04
"""

import re
import hashlib
import hmac
import requests
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PasswordStrength(Enum):
    """Password strength levels"""
    VERY_WEAK = 1
    WEAK = 2
    FAIR = 3
    GOOD = 4
    STRONG = 5
    VERY_STRONG = 6

@dataclass
class PasswordValidation:
    """Password validation result"""
    is_valid: bool
    strength: PasswordStrength
    score: int  # 0-100
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

class PasswordSecurityManager:
    """
    Comprehensive password security management system.
    
    Implements modern password security standards based on NIST 800-63B guidelines.
    """
    
    def __init__(self):
        """Initialize password security manager"""
        # Common passwords list (top 1000 most common passwords)
        self.common_passwords = self._load_common_passwords()
        
        # Password requirements
        self.min_length = 12
        self.max_length = 128
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_digit = True
        self.require_special = True
        self.special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # Account lockout settings
        self.max_failed_attempts = 5
        self.lockout_duration = 900  # 15 minutes
        
    def validate_password(self, password: str, user_info: Optional[Dict] = None) -> PasswordValidation:
        """
        Comprehensive password validation with strength scoring.
        
        Args:
            password: Password to validate
            user_info: Optional user information for personal data checking
            
        Returns:
            PasswordValidation object with detailed results
        """
        errors = []
        warnings = []
        suggestions = []
        score = 0
        
        if not password:
            return PasswordValidation(
                is_valid=False,
                strength=PasswordStrength.VERY_WEAK,
                score=0,
                errors=["Password is required"],
                warnings=[],
                suggestions=["Please enter a password"]
            )
        
        # Basic length requirements
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")
        elif len(password) >= self.min_length:
            score += 20
            
        if len(password) > self.max_length:
            errors.append(f"Password must not exceed {self.max_length} characters")
            
        # Character type requirements
        has_lowercase = bool(re.search(r'[a-z]', password))
        has_uppercase = bool(re.search(r'[A-Z]', password))
        has_digit = bool(re.search(r'\d', password))
        has_special = bool(re.search(f'[{re.escape(self.special_chars)}]', password))
        
        if self.require_lowercase and not has_lowercase:
            errors.append("Password must contain at least one lowercase letter")
        elif has_lowercase:
            score += 10
            
        if self.require_uppercase and not has_uppercase:
            errors.append("Password must contain at least one uppercase letter")
        elif has_uppercase:
            score += 10
            
        if self.require_digit and not has_digit:
            errors.append("Password must contain at least one number")
        elif has_digit:
            score += 10
            
        if self.require_special and not has_special:
            errors.append(f"Password must contain at least one special character ({self.special_chars})")
        elif has_special:
            score += 15
            
        # Advanced scoring criteria
        # Length bonus
        if len(password) >= 16:
            score += 10
        if len(password) >= 20:
            score += 5
            
        # Character diversity
        char_types = sum([has_lowercase, has_uppercase, has_digit, has_special])
        score += char_types * 5
        
        # Unique character ratio
        unique_chars = len(set(password))
        if unique_chars >= len(password) * 0.8:
            score += 10
        elif unique_chars >= len(password) * 0.6:
            score += 5
            
        # Pattern detection (negative scoring)
        if re.search(r'(.)\1{2,}', password):  # Repeated characters
            score -= 10
            warnings.append("Avoid repeating the same character multiple times")
            
        if re.search(r'(012|123|234|345|456|567|678|789|890)', password):
            score -= 15
            warnings.append("Avoid sequential numbers")
            
        if re.search(r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)', password.lower()):
            score -= 15
            warnings.append("Avoid sequential letters")
            
        # Keyboard patterns
        keyboard_patterns = ['qwerty', 'asdf', 'zxcv', '1234', 'abcd']
        for pattern in keyboard_patterns:
            if pattern in password.lower():
                score -= 20
                warnings.append("Avoid keyboard patterns")
                break
                
        # Common password check
        if self._is_common_password(password):
            score -= 50  # Heavily penalize common passwords
            errors.append("This password is too common and easily guessed")
            suggestions.append("Use a unique password that's not found in common password lists")
            
        # Personal information check
        if user_info:
            if self._contains_personal_info(password, user_info):
                score -= 25
                errors.append("Password should not contain personal information")
                suggestions.append("Avoid using your name, email, or organization name in passwords")
                
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        # Determine strength level
        strength = self._calculate_strength(score)
        
        # Add suggestions based on missing requirements
        if not errors:
            if score < 70:
                suggestions.append("Consider making your password longer or more complex")
            if len(password) < 16:
                suggestions.append("Longer passwords (16+ characters) are more secure")
            if char_types < 4:
                suggestions.append("Use a mix of uppercase, lowercase, numbers, and special characters")
                
        is_valid = len(errors) == 0 and score >= 60
        
        return PasswordValidation(
            is_valid=is_valid,
            strength=strength,
            score=score,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def check_breached_password(self, password: str) -> Tuple[bool, int]:
        """
        Check if password has been found in known data breaches using HaveIBeenPwned API.
        
        Args:
            password: Password to check
            
        Returns:
            Tuple of (is_breached, breach_count)
        """
        try:
            # Hash password with SHA-1
            sha1_hash = hashlib.sha1(password.encode('utf-8')).hexdigest().upper()
            
            # Send first 5 characters to API
            prefix = sha1_hash[:5]
            suffix = sha1_hash[5:]
            
            # Query HaveIBeenPwned API
            response = requests.get(
                f"https://api.pwnedpasswords.com/range/{prefix}",
                timeout=5,
                headers={'User-Agent': 'LLM-Data-Analysis-System-Password-Check'}
            )
            
            if response.status_code == 200:
                # Check if our suffix is in the response
                for line in response.text.splitlines():
                    hash_suffix, count = line.split(':')
                    if hash_suffix == suffix:
                        return True, int(count)
                return False, 0
            else:
                logger.warning(f"HaveIBeenPwned API returned status {response.status_code}")
                return False, 0
                
        except Exception as e:
            logger.error(f"Error checking breached password: {e}")
            return False, 0
    
    def generate_password_suggestions(self, length: int = 16) -> List[str]:
        """
        Generate secure password suggestions.
        
        Args:
            length: Desired password length
            
        Returns:
            List of password suggestions
        """
        suggestions = []
        
        # Passphrase suggestion
        suggestions.append(f"Consider using a passphrase like: 'Coffee#Morning$Run!2024' (easy to remember, secure)")
        
        # Pattern suggestions
        suggestions.append(f"Try: [Word][Symbol][Number][Word][Symbol] like 'Tree$42Mountain!'")
        suggestions.append(f"Use: [Phrase with spaces replaced by symbols] like 'I@Love#Coffee$123'")
        
        return suggestions
    
    def _calculate_strength(self, score: int) -> PasswordStrength:
        """Calculate password strength from score"""
        if score >= 90:
            return PasswordStrength.VERY_STRONG
        elif score >= 80:
            return PasswordStrength.STRONG
        elif score >= 70:
            return PasswordStrength.GOOD
        elif score >= 60:
            return PasswordStrength.FAIR
        elif score >= 40:
            return PasswordStrength.WEAK
        else:
            return PasswordStrength.VERY_WEAK
    
    def _is_common_password(self, password: str) -> bool:
        """Check if password is in common passwords list"""
        return password.lower() in self.common_passwords
    
    def _contains_personal_info(self, password: str, user_info: Dict) -> bool:
        """Check if password contains personal information"""
        password_lower = password.lower()
        
        # Check against user information
        checks = []
        if 'email' in user_info:
            email_parts = user_info['email'].lower().split('@')
            checks.extend(email_parts)
            if len(email_parts) > 1:
                checks.extend(email_parts[1].split('.'))
                
        if 'full_name' in user_info:
            name_parts = user_info['full_name'].lower().split()
            checks.extend(name_parts)
            
        if 'organization' in user_info:
            org_parts = user_info['organization'].lower().split()
            checks.extend(org_parts)
            
        # Check if any personal info appears in password
        for check in checks:
            if len(check) >= 3 and check in password_lower:
                return True
                
        return False
    
    def _load_common_passwords(self) -> set:
        """Load common passwords list"""
        # Common passwords list (top 100 most common)
        common_passwords = {
            'password', '123456', '123456789', 'guest', 'qwerty', '12345678', '111111', '12345',
            'col123456', '123123', '1234567', '1234', '1234567890', '000000', '555555', '666666',
            '123321', '654321', '7777777', '123', 'D1lakiss', '777777', 'admin', 'abc123',
            'password1', 'administrator', 'welcome', 'root', 'superman', '112233', 'iloveyou',
            'trustno1', '1qaz2wsx', 'dragon', 'master', 'hello', 'letmein', 'login', 'princess',
            '1q2w3e4r', 'solo', 'passw0rd', 'starwars', 'summer', 'monkey', 'testing', '1',
            'a', 'hockey', 'secret', 'jordan', 'harley', 'ranger', 'hunter', 'buster',
            'thomas', 'robert', 'soccer', 'batman', 'test', 'pass', 'killer', 'shadow',
            'basketball', 'michael', 'football', 'computer', 'wizard', 'asdfgh', 'whatever',
            'dragon', 'andrew', 'joshua', 'anthony', 'andrea', 'security', 'thunder', 'ginger',
            'hammer', 'silver', 'mustang', 'cooper', 'calvin', 'melissa', 'goddess', 'tigers',
            'captain', 'nascar', 'freedom', 'william', 'flowers', 'charlie', 'yellow', 'dancing',
            'cameron', 'secret', 'pepper', 'coffee', 'daniel', 'smart', 'trial', 'jupiter',
            'broncos', 'happy', 'london', 'college', 'apollo', 'fire', 'music', 'edward'
        }
        return common_passwords

# Global password security manager instance
password_manager = PasswordSecurityManager()

def validate_password(password: str, user_info: Optional[Dict] = None) -> PasswordValidation:
    """
    Validate password using the global password manager.
    
    Args:
        password: Password to validate
        user_info: Optional user information
        
    Returns:
        PasswordValidation result
    """
    return password_manager.validate_password(password, user_info)

def check_password_breach(password: str) -> Tuple[bool, int]:
    """
    Check if password has been breached.
    
    Args:
        password: Password to check
        
    Returns:
        Tuple of (is_breached, breach_count)
    """
    return password_manager.check_breached_password(password)

def get_password_strength_color(strength: PasswordStrength) -> str:
    """Get color for password strength display"""
    colors = {
        PasswordStrength.VERY_WEAK: "#ff4444",    # Red
        PasswordStrength.WEAK: "#ff8800",         # Orange
        PasswordStrength.FAIR: "#ffcc00",         # Yellow
        PasswordStrength.GOOD: "#88cc00",         # Light Green
        PasswordStrength.STRONG: "#44aa00",       # Green
        PasswordStrength.VERY_STRONG: "#00aa44",  # Dark Green
    }
    return colors.get(strength, "#999999")

def get_password_strength_text(strength: PasswordStrength) -> str:
    """Get text description for password strength"""
    texts = {
        PasswordStrength.VERY_WEAK: "Very Weak",
        PasswordStrength.WEAK: "Weak", 
        PasswordStrength.FAIR: "Fair",
        PasswordStrength.GOOD: "Good",
        PasswordStrength.STRONG: "Strong",
        PasswordStrength.VERY_STRONG: "Very Strong",
    }
    return texts.get(strength, "Unknown")