"""
Fixed Authentication UI Components
Temporary fix for register tab issue
"""

import streamlit as st
from typing import Optional
from auth.validation import validate_email
from services.auth_service import auth_service
from auth.input_validation import validate_and_sanitize_input

def show_auth_page():
    """Display the authentication page with login/register tabs - FIXED VERSION"""
    
    # Custom CSS
    st.markdown("""
    <style>
    .auth-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main authentication container
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    # App title and description
    st.title("🧬 LLM Data Cleaning System")
    st.markdown("**AI-powered experimental data cleaning for research labs**")
    st.markdown("---")
    
    # Authentication tabs - SIMPLIFIED WITHOUT RERUN
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        show_login_form()
    
    with tab2:
        show_register_form()
    
    st.markdown('</div>', unsafe_allow_html=True)


def show_login_form():
    """Display the login form"""
    with st.form("login_form", clear_on_submit=False):
        st.subheader("Login to Your Account")
        
        email = st.text_input("Email Address", placeholder="user@example.com")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        submit = st.form_submit_button("Login", use_container_width=True, type="primary")
        
        if submit:
            if not email:
                st.error("Email address is required")
            elif not validate_email(email):
                st.error("Please enter a valid email address")
            elif not password:
                st.error("Password is required")
            else:
                # Attempt login
                success, message = auth_service.login_user(email, password)
                if success:
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error(message if message else 'Login failed')


def show_register_form():
    """Display the registration form - WORKING VERSION"""
    with st.form("register_form", clear_on_submit=False):
        st.subheader("Create New Account")
        
        # Form fields
        full_name = st.text_input("Full Name", placeholder="Dr. Jane Smith")
        email = st.text_input("Email Address", placeholder="jane.smith@university.edu")
        organization = st.text_input("Organization/Lab Name", placeholder="Smith Research Lab")
        
        account_type = st.selectbox(
            "Account Type",
            options=["Academic Lab", "Research Institute", "Biotech Company", "Pharmaceutical", "Other"],
            index=0
        )
        
        password = st.text_input("Password", type="password", placeholder="Min. 12 characters")
        password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
        
        # Terms checkbox
        agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
        
        # Submit button
        submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")
        
        if submit:
            # Validation with input sanitization
            errors = []
            sanitized_values = {}
            
            # Validate and sanitize full name
            if not full_name:
                errors.append("Full name is required")
            else:
                name_validation = validate_and_sanitize_input('user_name', full_name)
                if not name_validation.is_valid:
                    errors.extend(name_validation.errors)
                else:
                    sanitized_values['full_name'] = name_validation.sanitized_value
                    # Show warnings for unusual characters
                    for warning in name_validation.warnings:
                        st.warning(f"⚠️ {warning}")
            
            # Validate and sanitize email
            if not email:
                errors.append("Email address is required")
            else:
                email_validation = validate_and_sanitize_input('email', email)
                if not email_validation.is_valid:
                    errors.extend(email_validation.errors)
                else:
                    sanitized_values['email'] = email_validation.sanitized_value
            
            # Validate and sanitize organization name
            if not organization:
                errors.append("Organization name is required")
            else:
                org_validation = validate_and_sanitize_input('organization', organization)
                if not org_validation.is_valid:
                    errors.extend(org_validation.errors)
                else:
                    sanitized_values['organization'] = org_validation.sanitized_value
                    # Show warnings for unusual characters
                    for warning in org_validation.warnings:
                        st.warning(f"⚠️ {warning}")
            
            if not password:
                errors.append("Password is required")
            else:
                # Use comprehensive password validation
                try:
                    from auth.password_security import validate_password
                    user_info = {
                        'email': sanitized_values.get('email', email),
                        'full_name': sanitized_values.get('full_name', full_name),
                        'organization': sanitized_values.get('organization', organization)
                    }
                    validation_result = validate_password(password, user_info)
                    
                    if not validation_result.is_valid:
                        errors.extend(validation_result.errors)
                    
                    # Show warnings as info messages
                    if validation_result.warnings:
                        for warning in validation_result.warnings:
                            st.warning(f"⚠️ {warning}")
                    
                    # Show suggestions for weak passwords
                    if validation_result.suggestions:
                        st.info("💡 Password suggestions:\n" + "\n".join(f"• {s}" for s in validation_result.suggestions))
                    
                    # Display password strength
                    if password:
                        strength_colors = {
                            1: "#ff4444",  # Very Weak - Red
                            2: "#ff8800",  # Weak - Orange  
                            3: "#ffcc00",  # Fair - Yellow
                            4: "#88cc00",  # Good - Light Green
                            5: "#44aa00",  # Strong - Green
                            6: "#00aa44",  # Very Strong - Dark Green
                        }
                        strength_text = ["", "Very Weak", "Weak", "Fair", "Good", "Strong", "Very Strong"]
                        
                        color = strength_colors.get(validation_result.strength.value, "#999999")
                        text = strength_text[validation_result.strength.value] if validation_result.strength.value < len(strength_text) else "Unknown"
                        
                        st.markdown(f"""
                        <div style="margin: 10px 0;">
                            <strong>Password Strength:</strong> 
                            <span style="color: {color}; font-weight: bold;">{text}</span>
                            <span style="color: #666;">({validation_result.score}/100)</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except ImportError:
                    # Fallback to basic length check if password security module not available
                    if len(password) < 12:
                        errors.append("Password must be at least 12 characters long")
            
            if password != password_confirm:
                errors.append("Passwords do not match")
            
            if not agree_terms:
                errors.append("You must agree to the terms to create an account")
            
            # Show errors or proceed
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Attempt registration with sanitized values
                success, message = auth_service.register_user(
                    email=sanitized_values.get('email', email),
                    password=password,
                    full_name=sanitized_values.get('full_name', full_name),
                    organization_name=sanitized_values.get('organization', organization),
                    account_type=account_type
                )
                result = {'success': success, 'error': message if not success else None}
                
                if result['success']:
                    st.success("Registration successful! You are now logged in.")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(result.get('error', 'Registration failed'))


def show_logout_button():
    """Display logout button in sidebar"""
    if st.sidebar.button("Logout", use_container_width=True):
        auth_service.clear_session()
        st.rerun()