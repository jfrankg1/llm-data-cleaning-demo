"""
Authentication Service
Handles user authentication, session management, and security
"""

import os
import hashlib
import streamlit as st
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple, Any

# Import database functions
from auth.database import (
    authenticate_user, create_user, create_session, get_session, 
    update_session_activity, invalidate_session, is_system_admin
)
from auth.validation import validate_email
from auth.input_validation import validate_and_sanitize_input

# Import rate limiting if available
try:
    from config.environment import get_environment_config
    env_config = get_environment_config()
    RATE_LIMITING_AVAILABLE = env_config.get('RATE_LIMITING_AVAILABLE', False)
except Exception:
    RATE_LIMITING_AVAILABLE = False


class AuthService:
    """Centralized authentication service"""
    
    def __init__(self):
        self.session_duration_hours = 24
    
    def init_session_state(self) -> None:
        """Initialize session state with default values"""
        defaults = {
            'authentication_status': None,
            'username': None,
            'user_id': None,
            'organization_id': None,
            'organization_name': None,
            'is_admin': False,
            'batch_processing_enabled': True,
            'login_email': '',
            'login_password': '',
            'register_full_name': '',
            'register_email': '',
            'register_organization': '',
            'register_password': '',
            'register_password_confirm': '',
            'register_account_type': 'Academic Lab',
            'current_tab': 'Login',
            'form_errors': {},
            'current_page': 'main',
            'session_token': None
        }
        
        # Set defaults first
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Try to restore session after defaults are set
        self.restore_session()
    
    def generate_session_token(self, user_id: str) -> str:
        """Generate a unique session token for the user"""
        timestamp = datetime.now(timezone.utc).isoformat()
        token_string = f"{user_id}:{timestamp}:{os.urandom(16).hex()}"
        return hashlib.sha256(token_string.encode()).hexdigest()
    
    def save_session(self) -> bool:
        """Save session data to database for persistence"""
        if not st.session_state.get('authentication_status') or not st.session_state.get('user_id'):
            return False
        
        # Prevent infinite loops during URL updates
        if st.session_state.get('_saving_session'):
            return False
        
        st.session_state['_saving_session'] = True
        
        try:
            # Generate session token if not exists
            if not st.session_state.get('session_token'):
                st.session_state['session_token'] = self.generate_session_token(st.session_state['user_id'])
            
            # Set expiration to 24 hours from now
            expires_at = datetime.now(timezone.utc) + timedelta(hours=self.session_duration_hours)
            
            # Create session in database
            success = create_session(
                user_id=st.session_state['user_id'],
                session_token=st.session_state['session_token'],
                expires_at=expires_at,
                user_agent=None,  # Could add browser info if needed
                ip_address=None   # Could add IP if needed
            )
            
            if success:
                print(f"DEBUG: Session saved to database successfully")
                return True
            else:
                print(f"DEBUG: Failed to save session to database")
                return False
                
        except Exception as e:
            print(f"DEBUG: Error saving session: {e}")
            return False
        finally:
            # Clear the saving flag
            st.session_state['_saving_session'] = False
    
    def restore_session(self) -> bool:
        """Restore session from database using session token"""
        # Only run restoration if not already authenticated
        if st.session_state.get('authentication_status'):
            return True
        
        # Check if we've already attempted restoration in this specific page load
        if st.session_state.get('_session_restore_attempted'):
            return False
        
        # Mark that we've attempted restoration to avoid multiple attempts
        st.session_state['_session_restore_attempted'] = True
        
        print(f"DEBUG: Starting session restore...")
        
        # URL session token retrieval removed for security
        # Sessions will now rely on Streamlit's built-in session state only
        session_token = None
        
        if False:  # Disabled URL token restoration for security
            try:
                session_data = get_session(session_token)
                
                if session_data:
                    print(f"DEBUG: Valid session found in database")
                    
                    # Update session activity
                    update_session_activity(session_token)
                    
                    # Restore all user session data
                    st.session_state['authentication_status'] = True
                    st.session_state['user_id'] = session_data['user_id']
                    st.session_state['username'] = session_data['email']
                    st.session_state['organization_id'] = session_data['organization_id']
                    st.session_state['organization_name'] = session_data.get('organization_name', '')
                    st.session_state['is_admin'] = session_data.get('is_admin', False)
                    st.session_state['is_system_admin'] = is_system_admin(session_data['user_id'])
                    st.session_state['session_token'] = session_token
                    
                    print(f"DEBUG: ✅ Session restored successfully!")
                    return True
                else:
                    print(f"DEBUG: Session token not found or expired in database")
                    return False
            except Exception as e:
                print(f"DEBUG: Error checking session in database: {e}")
                return False
        else:
            print(f"DEBUG: No session token found in URL parameters")
            return False
    
    def clear_session(self) -> None:
        """Clear all session data"""
        # Invalidate current session token in database if it exists
        current_token = st.session_state.get('session_token')
        if current_token:
            try:
                invalidate_session(current_token)
                print(f"DEBUG: Session token invalidated in database")
            except Exception as e:
                print(f"DEBUG: Error invalidating session: {e}")
        
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Clear URL parameters
        try:
            # URL session token cleanup removed (no longer used for security)
            pass
        except Exception:
            pass
        
        # Reset the restoration attempt flag so session restore can work again
        st.session_state['_session_restore_attempted'] = False
    
    def login_user(self, email: str, password: str, ip_address: str = "127.0.0.1") -> Tuple[bool, str]:
        """
        Authenticate user and create session
        Returns: (success, message)
        """
        try:
            # Check rate limits if available
            if RATE_LIMITING_AVAILABLE:
                try:
                    from auth.rate_limiter import rate_limiter
                    
                    # Check IP-based login limits
                    allowed, info = rate_limiter.check_ip_login_limit(ip_address)
                    if not allowed:
                        return False, f"Too many login attempts. {info.get('message', 'Please try again later.')}"
                    
                    # Check user-based login limits
                    allowed, info = rate_limiter.check_user_login_limit(email)
                    if not allowed:
                        return False, f"Too many login attempts for this account. {info.get('message', 'Please try again later.')}"
                        
                except Exception as e:
                    print(f"DEBUG: Rate limiting check failed: {e}")
                    # Continue without rate limiting if it fails
            
            # Authenticate user
            user_data = authenticate_user(email, password, ip_address)
            if not user_data:
                return False, "Invalid email or password"
            
            # Check if account is active
            if not user_data.get('is_active', True):
                return False, "Account is deactivated. Please contact support."
            
            # Set session state
            st.session_state['authentication_status'] = True
            st.session_state['user_id'] = user_data['id']  # Fixed: use 'id' not 'user_id'
            st.session_state['username'] = user_data['email']
            st.session_state['organization_id'] = user_data['organization_id']
            st.session_state['organization_name'] = user_data.get('organization_name', '')
            st.session_state['is_admin'] = user_data.get('is_admin', False)
            st.session_state['is_system_admin'] = is_system_admin(user_data['id'])  # Fixed: use 'id'
            
            # Save session to database
            session_saved = self.save_session()
            if session_saved:
                return True, "Login successful!"
            else:
                return True, "Login successful! (Session may not persist across page refreshes)"
                
        except Exception as e:
            print(f"DEBUG: Login error: {e}")
            return False, "An error occurred during login. Please try again."
    
    def register_user(self, email: str, password: str, full_name: str, 
                     organization_name: str, account_type: str = "Academic Lab") -> Tuple[bool, str]:
        """
        Register new user and create session
        Returns: (success, message)
        """
        try:
            # Validate and sanitize inputs
            email_validation = validate_and_sanitize_input('email', email)
            if not email_validation.is_valid:
                return False, "; ".join(email_validation.errors)
            
            name_validation = validate_and_sanitize_input('user_name', full_name)
            if not name_validation.is_valid:
                return False, "; ".join(name_validation.errors)
            
            org_validation = validate_and_sanitize_input('organization', organization_name)
            if not org_validation.is_valid:
                return False, "; ".join(org_validation.errors)
            
            # Use sanitized values
            sanitized_email = email_validation.sanitized_value
            sanitized_name = name_validation.sanitized_value
            sanitized_org = org_validation.sanitized_value
            
            # Create user account with sanitized values
            result = create_user(sanitized_email, password, sanitized_name, sanitized_org)
            
            # Check if result is a user dict (successful creation)
            if result and isinstance(result, dict) and 'id' in result:
                # Auto-login after successful registration
                st.session_state['authentication_status'] = True
                st.session_state['user_id'] = result['id']
                st.session_state['username'] = result['email']
                st.session_state['organization_id'] = result['organization_id']
                st.session_state['organization_name'] = result.get('organization_name', '')
                st.session_state['is_admin'] = result.get('is_admin', False)
                st.session_state['is_system_admin'] = False  # New users are never system admins
                
                # Save session
                self.save_session()
                
                return True, "Account created successfully! You are now logged in."
            else:
                return False, "Failed to create account"
                
        except Exception as e:
            print(f"DEBUG: Registration error: {e}")
            return False, "An error occurred during registration. Please try again."
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        return bool(st.session_state.get('authentication_status'))
    
    def is_admin(self) -> bool:
        """Check if current user is an admin"""
        return bool(st.session_state.get('is_admin'))
    
    def is_system_admin(self) -> bool:
        """Check if current user is a system admin"""
        return bool(st.session_state.get('is_system_admin'))
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current user information"""
        if not self.is_authenticated():
            return None
        
        return {
            'user_id': st.session_state.get('user_id'),
            'username': st.session_state.get('username'),
            'organization_id': st.session_state.get('organization_id'),
            'organization_name': st.session_state.get('organization_name'),
            'is_admin': st.session_state.get('is_admin', False),
            'is_system_admin': st.session_state.get('is_system_admin', False)
        }
    
    def get_organization_id(self) -> Optional[str]:
        """Get current user's organization ID"""
        return st.session_state.get('organization_id')
    
    def get_user_id(self) -> Optional[str]:
        """Get current user's ID"""
        return st.session_state.get('user_id')


# Global instance
auth_service = AuthService()