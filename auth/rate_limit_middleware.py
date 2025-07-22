"""
Rate limiting middleware for Streamlit application.
Integrates rate limiting into the application flow with proper error handling.
"""

import streamlit as st
import functools
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from .rate_limiter import rate_limiter, RateLimitExceeded, RateLimitType

logger = logging.getLogger(__name__)


def get_client_ip() -> str:
    """Get client IP address from Streamlit context"""
    try:
        # Try to get real IP from headers (behind proxy)
        if hasattr(st, 'get_query_params'):
            # This is a fallback - in production, you'd get this from request headers
            return "127.0.0.1"  # Default for local development
        return "127.0.0.1"
    except Exception:
        return "127.0.0.1"


def get_current_user_id() -> Optional[str]:
    """Get current user ID from session state"""
    return st.session_state.get('user_id')


def get_current_org_id() -> Optional[str]:
    """Get current organization ID from session state"""
    return st.session_state.get('organization_id')


def handle_rate_limit_exceeded(error: RateLimitExceeded):
    """Handle rate limit exceeded errors with user-friendly messages"""
    
    reset_time_str = error.reset_time.strftime('%H:%M:%S')
    
    if error.limit_type == RateLimitType.REQUEST:
        st.error(
            f"🚫 **Too Many Requests**\n\n"
            f"You've exceeded the request limit of {error.info['limit']} per minute.\n"
            f"Please wait until {reset_time_str} before trying again."
        )
    
    elif error.limit_type == RateLimitType.API_CALL:
        st.error(
            f"🚫 **API Limit Exceeded**\n\n"
            f"You've reached your API call limit of {error.info['limit']} per hour.\n"
            f"This limit resets at {reset_time_str}."
        )
        st.info(
            "💡 **Tip**: Consider upgrading your subscription for higher API limits."
        )
    
    elif error.limit_type == RateLimitType.FILE_UPLOAD:
        st.error(
            f"🚫 **Upload Limit Exceeded**\n\n"
            f"You've reached your file upload limit of {error.info['limit']} per hour.\n"
            f"You can upload more files after {reset_time_str}."
        )
    
    elif error.limit_type == RateLimitType.LOGIN:
        st.error(
            f"🚫 **Too Many Login Attempts**\n\n"
            f"Too many failed login attempts. Please wait until {reset_time_str} before trying again."
        )
    
    elif error.limit_type == RateLimitType.IP:
        st.error(
            f"🚫 **Rate Limit Exceeded**\n\n"
            f"Too many requests from your connection. Please wait until {reset_time_str}."
        )
    
    else:
        st.error(
            f"🚫 **Rate Limit Exceeded**\n\n"
            f"Please wait until {reset_time_str} before trying again."
        )
    
    # Show remaining time dynamically
    with st.expander("ℹ️ Rate Limit Details"):
        st.write(f"**Limit**: {error.info['limit']} per window")
        st.write(f"**Current Count**: {error.info['current_count']}")
        st.write(f"**Reset Time**: {reset_time_str}")
        st.write(f"**Limit Type**: {error.limit_type.value}")


def rate_limit_check(limit_type: RateLimitType, check_global: bool = True):
    """Decorator for rate limiting Streamlit functions"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                user_id = get_current_user_id()
                org_id = get_current_org_id()
                ip_address = get_client_ip()
                
                # Check global burst protection first
                if check_global:
                    allowed, info = rate_limiter.check_global_burst_limit()
                    if not allowed:
                        rate_limiter.log_rate_limit_violation(
                            user_id, ip_address, RateLimitType.REQUEST, info
                        )
                        st.error("🚫 **Service Temporarily Unavailable**\n\nThe service is experiencing high load. Please try again in a few minutes.")
                        st.stop()
                
                # Check IP-based limits
                allowed, info = rate_limiter.check_ip_request_limit(ip_address)
                if not allowed:
                    rate_limiter.log_rate_limit_violation(
                        user_id, ip_address, RateLimitType.IP, info
                    )
                    handle_rate_limit_exceeded(RateLimitExceeded(RateLimitType.IP, info))
                    st.stop()
                
                # Check user-specific limits
                if user_id:
                    if limit_type == RateLimitType.REQUEST:
                        allowed, info = rate_limiter.check_user_request_limit(user_id)
                    elif limit_type == RateLimitType.API_CALL:
                        allowed, info = rate_limiter.check_user_api_limit(user_id)
                        
                        # Also check organization limits
                        if allowed and org_id:
                            subscription_tier = st.session_state.get('subscription_tier', 'free')
                            allowed, org_info = rate_limiter.check_org_api_limit(org_id, subscription_tier)
                            if not allowed:
                                info = org_info  # Use org limit info
                    
                    elif limit_type == RateLimitType.FILE_UPLOAD:
                        allowed, info = rate_limiter.check_user_upload_limit(user_id)
                    elif limit_type == RateLimitType.LOGIN:
                        allowed, info = rate_limiter.check_user_login_limit(user_id)
                    else:
                        allowed = True
                    
                    if not allowed:
                        rate_limiter.log_rate_limit_violation(
                            user_id, ip_address, limit_type, info
                        )
                        handle_rate_limit_exceeded(RateLimitExceeded(limit_type, info))
                        st.stop()
                
                # Store rate limit info in session for UI display
                if 'rate_limit_info' not in st.session_state:
                    st.session_state['rate_limit_info'] = {}
                
                st.session_state['rate_limit_info'][limit_type.value] = info
                
                # Call the original function
                return func(*args, **kwargs)
                
            except RateLimitExceeded as e:
                handle_rate_limit_exceeded(e)
                st.stop()
            except Exception as e:
                logger.error(f"Rate limiting error: {e}")
                # Don't block the request if rate limiting fails
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def check_and_enforce_rate_limits():
    """Check all relevant rate limits for current request"""
    try:
        user_id = get_current_user_id()
        ip_address = get_client_ip()
        
        # Check IP limits
        allowed, info = rate_limiter.check_ip_request_limit(ip_address)
        if not allowed:
            handle_rate_limit_exceeded(RateLimitExceeded(RateLimitType.IP, info))
            st.stop()
        
        # Check user request limits
        if user_id:
            allowed, info = rate_limiter.check_user_request_limit(user_id)
            if not allowed:
                handle_rate_limit_exceeded(RateLimitExceeded(RateLimitType.REQUEST, info))
                st.stop()
    
    except Exception as e:
        logger.error(f"Rate limit check failed: {e}")
        # Don't block requests if rate limiting fails


def show_rate_limit_status():
    """Show current rate limit status in sidebar"""
    try:
        user_id = get_current_user_id()
        org_id = get_current_org_id()
        
        if not user_id:
            return
        
        with st.sidebar:
            with st.expander("📊 Rate Limits", expanded=False):
                
                # User request limits (STATUS ONLY - don't consume)
                allowed, info = rate_limiter.get_user_request_status(user_id)
                remaining_pct = (info['remaining'] / info['limit']) * 100
                
                st.metric(
                    "Requests/Min", 
                    f"{info['remaining']}/{info['limit']}",
                    delta=f"{remaining_pct:.0f}% remaining"
                )
                
                if remaining_pct < 20:
                    st.warning("⚠️ Request limit nearly reached")
                
                # API call limits (STATUS ONLY - don't consume)
                allowed, info = rate_limiter.get_user_api_status(user_id)
                api_remaining_pct = (info['remaining'] / info['limit']) * 100
                
                st.metric(
                    "API Calls/Hour",
                    f"{info['remaining']}/{info['limit']}",
                    delta=f"{api_remaining_pct:.0f}% remaining"
                )
                
                if api_remaining_pct < 20:
                    st.warning("⚠️ API limit nearly reached")
                
                # Upload limits (STATUS ONLY - don't consume)
                allowed, info = rate_limiter.get_user_upload_status(user_id)
                upload_remaining_pct = (info['remaining'] / info['limit']) * 100
                
                st.metric(
                    "Uploads/Hour",
                    f"{info['remaining']}/{info['limit']}",
                    delta=f"{upload_remaining_pct:.0f}% remaining"
                )
                
                if upload_remaining_pct < 20:
                    st.warning("⚠️ Upload limit nearly reached")
                
                # Organization API limits (STATUS ONLY - don't consume)
                if org_id:
                    subscription_tier = st.session_state.get('subscription_tier', 'free')
                    allowed, info = rate_limiter.get_org_api_status(org_id, subscription_tier)
                    org_remaining_pct = (info['remaining'] / info['limit']) * 100
                    
                    st.metric(
                        f"Org API ({subscription_tier.title()})",
                        f"{info['remaining']}/{info['limit']}",
                        delta=f"{org_remaining_pct:.0f}% remaining"
                    )
                    
                    if org_remaining_pct < 20:
                        st.warning("⚠️ Organization API limit nearly reached")
    
    except Exception as e:
        logger.error(f"Failed to show rate limit status: {e}")


def init_rate_limiting():
    """Initialize rate limiting for the application"""
    # Add rate limiting check to session if not already present
    if 'rate_limiting_initialized' not in st.session_state:
        check_and_enforce_rate_limits()
        st.session_state['rate_limiting_initialized'] = True