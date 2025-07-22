"""
Admin UI components for the Streamlit app
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Try to import plotly, but make it optional
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from auth.repositories import (
    get_user_repository, get_organization_repository, get_experiment_repository
)
from auth.org_admin_advanced import (
    update_organization_profile, get_organization_profile, update_usage_limits,
    create_user_invitation, get_organization_invitations, cancel_user_invitation,
    get_org_admin_logs, get_organization_settings, update_organization_settings,
    get_notification_settings, update_notification_setting, export_organization_data,
    bulk_user_operation
)
from auth.permissions import require_admin, is_org_admin
from auth.admin_validation import safe_admin_text_input, safe_admin_selectbox, validate_admin_search_input, validate_admin_text_input


@require_admin
def show_admin_dashboard():
    """Main admin dashboard showing organization overview."""
    st.title("👨‍💼 LLM Data Cleaning System - Admin Dashboard")
    
    # Get organization stats
    org_repo = get_organization_repository()
    stats = org_repo.get_organization_stats(st.session_state['organization_id'])
    
    # Add daily activity data if not present
    if 'daily_activity' not in stats:
        stats['daily_activity'] = []
    
    # Overview metrics
    st.markdown("### 📊 Organization Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Users",
            stats['users']['total_users'],
            f"{stats['users']['active_users']} active"
        )
    
    with col2:
        st.metric(
            "Total Experiments",
            stats['experiments']['total_experiments'],
            f"{stats['experiments']['experiments_this_month']} this month"
        )
    
    with col3:
        st.metric(
            "Completion Rate",
            f"{(stats['experiments']['completed_experiments'] / max(stats['experiments']['total_experiments'], 1) * 100):.1f}%",
            f"{stats['experiments']['completed_experiments']} completed"
        )
    
    with col4:
        st.metric(
            "Admin Users",
            stats['users']['admin_users'],
            f"{(stats['users']['admin_users'] / max(stats['users']['total_users'], 1) * 100):.0f}% of users"
        )
    
    st.divider()
    
    # Activity chart
    if stats['daily_activity']:
        st.markdown("### 📈 30-Day Activity")
        
        if PLOTLY_AVAILABLE:
            # Convert to DataFrame for plotting
            activity_df = pd.DataFrame(stats['daily_activity'])
            activity_df['date'] = pd.to_datetime(activity_df['date'])
            
            # Create full date range for last 30 days
            date_range = pd.date_range(
                end=datetime.now().date(),
                periods=30,
                freq='D'
            )
            full_df = pd.DataFrame({'date': date_range})
            full_df = full_df.merge(activity_df, on='date', how='left')
            full_df['count'] = full_df['count'].fillna(0).astype(int)
            
            # Create plot
            fig = px.line(
                full_df,
                x='date',
                y='count',
                title='Experiments Created Per Day',
                markers=True
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Experiments",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to simple table
            activity_df = pd.DataFrame(stats['daily_activity'])
            st.dataframe(activity_df, use_container_width=True)
    
    st.divider()
    
    # Top users
    if stats['top_users']:
        st.markdown("### 🏆 Top Users by Experiment Count")
        
        top_users_df = pd.DataFrame(stats['top_users'])
        
        if PLOTLY_AVAILABLE and not top_users_df.empty:
            # Add display name for chart
            top_users_df['display_name'] = top_users_df.apply(
                lambda row: row['full_name'] if row.get('full_name') else row.get('username', row['email']), 
                axis=1
            )
            
            # Create horizontal bar chart
            fig = px.bar(
                top_users_df.head(10),
                y='display_name',
                x='experiment_count',
                orientation='h',
                title='Most Active Users',
                labels={'experiment_count': 'Experiments', 'display_name': 'User'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to metric display
            for idx, user in enumerate(stats['top_users'][:5], 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    display_name = user.get('full_name') or user.get('username') or user['email']
                    st.text(f"{idx}. {display_name} ({user['email']})")
                with col2:
                    st.metric("Experiments", user['experiment_count'])
    
    # Experiment status breakdown
    st.markdown("### 📊 Experiment Status Breakdown")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        status_data = {
            'Status': ['Completed', 'Processing', 'Failed'],
            'Count': [
                stats['experiments']['completed_experiments'],
                stats['experiments']['processing_experiments'],
                stats['experiments']['failed_experiments']
            ]
        }
        status_df = pd.DataFrame(status_data)
        
        if PLOTLY_AVAILABLE:
            fig = px.pie(
                status_df,
                values='Count',
                names='Status',
                color_discrete_map={
                    'Completed': '#28a745',
                    'Processing': '#ffc107',
                    'Failed': '#dc3545'
                }
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to metrics
            total = sum(status_data['Count'])
            for status, count in zip(status_data['Status'], status_data['Count']):
                pct = (count / total * 100) if total > 0 else 0
                emoji = {'Completed': '🟢', 'Processing': '🟡', 'Failed': '🔴'}[status]
                st.metric(f"{emoji} {status}", count, f"{pct:.0f}%")
    
    with col2:
        # Recent experiments table
        st.markdown("#### Recent Experiments")
        exp_repo = get_experiment_repository()
        recent_experiments = exp_repo.get_organization_experiments(
            st.session_state['organization_id'],
            limit=5
        )
        
        if recent_experiments:
            for exp in recent_experiments:
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.text(f"📄 {exp['name']}")
                    with col2:
                        st.text(f"👤 {exp['user_name'] or exp['user_email']}")
                    with col3:
                        status_color = {
                            'completed': '🟢',
                            'processing': '🟡',
                            'failed': '🔴'
                        }.get(exp['status'], '⚪')
                        st.text(f"{status_color} {exp['status'].title()}")


@require_admin
def show_user_management():
    """User management interface for admins."""
    st.title("👥 User Management")
    
    # Get all users in organization
    user_repo = get_user_repository()
    users = user_repo.get_organization_users(st.session_state['organization_id'])
    
    # Summary stats
    active_users = sum(1 for u in users if u['is_active'])
    admin_users = sum(1 for u in users if u['is_admin'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", len(users))
    with col2:
        st.metric("Active Users", active_users)
    with col3:
        st.metric("Admin Users", admin_users)
    
    st.divider()
    
    # User table with actions
    st.markdown("### User List")
    
    # Search and filters - SECURED with input validation
    col1, col2, col3 = st.columns(3)
    with col1:
        search_input = st.text_input("🔍 Search users by name or email", "", max_chars=100)
        is_valid, search = validate_admin_search_input(search_input, "user search")
        if not is_valid:
            search = ""  # Clear invalid search input
    with col2:
        status_filter = safe_admin_selectbox("Filter by status", ["All", "Active", "Inactive", "Admin"])
    with col3:
        sort_by = safe_admin_selectbox("Sort by", ["Name", "Email", "Last Login", "Experiments"])
    
    # Bulk operations
    st.markdown("#### Bulk Operations")
    selected_users = st.multiselect(
        "Select users for bulk operations:",
        options=[f"{u['email']} - {u.get('full_name') or u.get('username') or 'No name'}" for u in users],
        key="bulk_users"
    )
    
    if selected_users:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("✅ Activate Selected"):
                user_ids = [u['id'] for u in users if f"{u['email']} - {u.get('full_name') or u.get('username') or 'No name'}" in selected_users]
                results = bulk_user_operation(
                    st.session_state['organization_id'], 
                    user_ids, 
                    'activate', 
                    st.session_state['user_id']
                )
                st.success(f"Activated {len(results['success'])} users")
                if results['failed']:
                    st.warning(f"Failed to activate {len(results['failed'])} users")
                st.rerun()
        
        with col2:
            if st.button("❌ Deactivate Selected"):
                user_ids = [u['id'] for u in users if f"{u['email']} - {u.get('full_name') or u.get('username') or 'No name'}" in selected_users]
                results = bulk_user_operation(
                    st.session_state['organization_id'], 
                    user_ids, 
                    'deactivate', 
                    st.session_state['user_id']
                )
                st.success(f"Deactivated {len(results['success'])} users")
                if results['failed']:
                    st.warning(f"Failed to deactivate {len(results['failed'])} users")
                st.rerun()
        
        with col3:
            if st.button("👑 Make Admin"):
                user_ids = [u['id'] for u in users if f"{u['email']} - {u.get('full_name') or u.get('username') or 'No name'}" in selected_users]
                results = bulk_user_operation(
                    st.session_state['organization_id'], 
                    user_ids, 
                    'make_admin', 
                    st.session_state['user_id']
                )
                st.success(f"Made {len(results['success'])} users admin")
                if results['failed']:
                    st.warning(f"Failed to make {len(results['failed'])} users admin")
                st.rerun()
        
        with col4:
            if st.button("👤 Remove Admin"):
                user_ids = [u['id'] for u in users if f"{u['email']} - {u.get('full_name') or u.get('username') or 'No name'}" in selected_users]
                results = bulk_user_operation(
                    st.session_state['organization_id'], 
                    user_ids, 
                    'remove_admin', 
                    st.session_state['user_id']
                )
                st.success(f"Removed admin from {len(results['success'])} users")
                if results['failed']:
                    st.warning(f"Failed to remove admin from {len(results['failed'])} users")
                st.rerun()
    
    st.divider()
    
    # Get detailed usage for all users - create empty dict for now since this function needs org-wide usage
    usage_dict = {}
    
    # Filter users
    filtered_users = users
    
    # Apply search filter
    if search:
        search_lower = search.lower()
        filtered_users = [
            u for u in filtered_users
            if search_lower in u['email'].lower() or
            (u.get('full_name') and search_lower in u['full_name'].lower()) or
            (u.get('username') and search_lower in u['username'].lower())
        ]
    
    # Apply status filter
    if status_filter == "Active":
        filtered_users = [u for u in filtered_users if u['is_active']]
    elif status_filter == "Inactive":
        filtered_users = [u for u in filtered_users if not u['is_active']]
    elif status_filter == "Admin":
        filtered_users = [u for u in filtered_users if u['is_admin']]
    
    # Apply sorting
    if sort_by == "Name":
        filtered_users.sort(key=lambda u: u.get('full_name') or u.get('username') or u['email'])
    elif sort_by == "Email":
        filtered_users.sort(key=lambda u: u['email'])
    elif sort_by == "Last Login":
        filtered_users.sort(key=lambda u: u['last_login'] or datetime.min, reverse=True)
    elif sort_by == "Experiments":
        filtered_users.sort(key=lambda u: usage_dict.get(u['id'], {}).get('total_experiments', 0), reverse=True)
    
    # Display users
    for user in filtered_users:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 2])
            
            with col1:
                # User info
                st.markdown(f"**{user.get('full_name') or user.get('username') or 'No name'}**")
                st.text(user['email'])
            
            with col2:
                # Usage info
                usage = usage_dict.get(user['id'], {})
                st.text(f"📊 {usage.get('experiments_this_month', 0)} experiments/mo")
                st.text(f"💾 {usage.get('storage_mb', 0):.1f} MB storage")
            
            with col3:
                # Status badges
                if user['is_admin']:
                    st.markdown("👑 **Admin**")
                if user['is_active']:
                    st.markdown("✅ Active")
                else:
                    st.markdown("❌ Inactive")
                
                if user['last_login']:
                    last_login = user['last_login']
                    if isinstance(last_login, str):
                        last_login = datetime.fromisoformat(last_login.replace('Z', '+00:00'))
                    days_ago = (datetime.now() - last_login.replace(tzinfo=None)).days
                    st.text(f"Last login: {days_ago}d ago")
            
            with col4:
                # Quick stats
                st.metric("Total Exp.", usage.get('total_experiments', 0))
            
            with col5:
                # Action buttons
                if user['id'] != st.session_state['user_id']:  # Can't modify self
                    btn_col1, btn_col2 = st.columns(2)
                    
                    with btn_col1:
                        # Toggle active status
                        if user['is_active']:
                            if st.button("🚫 Deactivate", key=f"deact_{user['id']}"):
                                if user_repo.update_user_status(user['id'], False):
                                    st.success("User deactivated")
                                    st.rerun()
                                else:
                                    st.error("Failed to deactivate user")
                        else:
                            if st.button("✅ Activate", key=f"act_{user['id']}"):
                                if user_repo.update_user_status(user['id'], True):
                                    st.success("User activated")
                                    st.rerun()
                                else:
                                    st.error("Failed to activate user")
                    
                    with btn_col2:
                        # Toggle admin status
                        if user['is_admin']:
                            if st.button("👤 Remove Admin", key=f"rmadm_{user['id']}"):
                                if user_repo.update_user_admin_status(user['id'], False):
                                    st.success("Admin privileges removed")
                                    st.rerun()
                                else:
                                    st.error("Cannot remove last admin")
                        else:
                            if st.button("👑 Make Admin", key=f"mkadm_{user['id']}"):
                                if user_repo.update_user_admin_status(user['id'], True):
                                    st.success("Admin privileges granted")
                                    st.rerun()
                                else:
                                    st.error("Failed to grant admin privileges")
            
            st.divider()


@require_admin
def show_organization_settings():
    """Organization settings interface for admins."""
    st.title("🔧 Organization Settings")
    
    # Create tabs for different settings categories
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Profile", "📊 Usage Limits", "👥 User Invitations", 
        "🔔 Notifications", "📋 Audit Logs", "📦 Data Export"
    ])
    
    with tab1:
        show_organization_profile()
    
    with tab2:
        show_usage_limits_config()
    
    with tab3:
        show_user_invitations()
    
    with tab4:
        show_notification_settings()
    
    with tab5:
        show_admin_audit_logs()
    
    with tab6:
        show_data_export()


@require_admin
def show_organization_profile():
    """Organization profile management."""
    st.markdown("### 🏢 Organization Profile")
    
    # Get current profile
    profile = get_organization_profile(st.session_state['organization_id'])
    
    with st.form("organization_profile"):
        col1, col2 = st.columns(2)
        
        with col1:
            # SECURED: Organization name input with validation
            name_input = st.text_input("Organization Name", value=profile.get('name', ''), max_chars=100)
            # SECURED: Website URL input with validation
            website_input = st.text_input("Website", value=profile.get('website', ''), max_chars=500)
            subscription_tier = st.selectbox(
                "Subscription Tier", 
                ["free", "premium", "enterprise"],
                index=["free", "premium", "enterprise"].index(profile.get('subscription_tier', 'free'))
            )
        
        with col2:
            # SECURED: Description with HTML sanitization
            description_input = st.text_area("Description", value=profile.get('description', ''), max_chars=1000)
            # SECURED: Logo URL input with validation
            logo_url_input = st.text_input("Logo URL", value=profile.get('logo_url', ''), max_chars=500)
        
        submitted = st.form_submit_button("Update Profile")
        
        if submitted:
            # Validate all inputs before processing
            name_valid, name = validate_admin_text_input(name_input, 'organization', 'Organization Name')
            website_valid, website = validate_admin_text_input(website_input, 'url', 'Website')
            description_valid, description = validate_admin_text_input(description_input, 'search', 'Description')
            logo_url_valid, logo_url = validate_admin_text_input(logo_url_input, 'url', 'Logo URL')
            
            # Only proceed if all inputs are valid
            if name_valid and website_valid and description_valid and logo_url_valid:
                if update_organization_profile(
                    st.session_state['organization_id'], 
                    name=name, 
                    description=description, 
                    website=website, 
                    logo_url=logo_url,
                    admin_user_id=st.session_state['user_id']
                ):
                    st.success("✅ Organization profile updated successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to update organization profile")


@require_admin
def show_usage_limits_config():
    """Usage limits configuration."""
    st.markdown("### 📊 Usage Limits Configuration")
    
    # Get current limits and usage
    org_repo = get_organization_repository()
    current_usage = org_repo.get_organization_usage(st.session_state['organization_id'])
    
    # Display current usage
    st.markdown("#### Current Usage")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Experiments This Month",
            f"{current_usage['experiments']}/{current_usage['experiments_limit']}",
            f"{(current_usage['experiments']/current_usage['experiments_limit']*100):.0f}% used"
        )
    
    with col2:
        st.metric(
            "API Calls Today",
            f"{current_usage['api_calls']}/{current_usage['api_calls_limit']}",
            f"{(current_usage['api_calls']/current_usage['api_calls_limit']*100):.0f}% used"
        )
    
    with col3:
        st.metric(
            "Storage Used",
            f"{current_usage['storage_mb']:.1f}/{current_usage['storage_limit_mb']} MB",
            f"{(current_usage['storage_mb']/current_usage['storage_limit_mb']*100):.0f}% used"
        )
    
    st.divider()
    
    # Configuration form
    st.markdown("#### Update Limits")
    with st.form("usage_limits"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            experiments_monthly = st.number_input(
                "Monthly Experiments Limit",
                min_value=1,
                max_value=10000,
                value=current_usage['experiments_limit'],
                step=10
            )
        
        with col2:
            api_calls_daily = st.number_input(
                "Daily API Calls Limit",
                min_value=10,
                max_value=100000,
                value=current_usage['api_calls_limit'],
                step=100
            )
        
        with col3:
            storage_mb = st.number_input(
                "Storage Limit (MB)",
                min_value=100,
                max_value=100000,
                value=current_usage['storage_limit_mb'],
                step=100
            )
        
        st.warning("⚠️ Changing limits will affect all users in your organization.")
        
        submitted = st.form_submit_button("Update Usage Limits")
        
        if submitted:
            if update_usage_limits(
                st.session_state['organization_id'],
                experiments_monthly=experiments_monthly,
                storage_mb=storage_mb,
                api_calls_daily=api_calls_daily,
                admin_user_id=st.session_state['user_id']
            ):
                st.success("✅ Usage limits updated successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to update usage limits")


@require_admin
def show_user_invitations():
    """User invitation management."""
    st.markdown("### 👥 User Invitations")
    
    # Invite new user form
    st.markdown("#### Invite New User")
    with st.form("invite_user"):
        col1, col2 = st.columns(2)
        
        with col1:
            # SECURED: Email input with validation
            email_input = st.text_input("Email Address", max_chars=254)
        
        with col2:
            role = safe_admin_selectbox("Role", ["member", "admin"])
        
        submitted = st.form_submit_button("Send Invitation")
        
        if submitted:
            # Validate email input before processing
            email_valid, email = validate_admin_text_input(email_input, 'email', 'Email Address')
            
            if email_valid and email:
                invitation_token = create_user_invitation(
                    st.session_state['organization_id'],
                    email,
                    role,
                    st.session_state['user_id']
                )
                
                if invitation_token:
                    st.success(f"✅ Invitation sent to {email}")
                    # In a real implementation, you would send an email here
                    st.info(f"📧 Invitation token: `{invitation_token}`")
                    st.rerun()
                else:
                    st.error("❌ Failed to create invitation. User may already exist or have a pending invitation.")
            elif not email:
                st.error("❌ Please enter an email address")
    
    st.divider()
    
    # Show pending invitations
    st.markdown("#### Pending Invitations")
    invitations = get_organization_invitations(st.session_state['organization_id'])
    
    if invitations:
        for invitation in invitations:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.text(f"📧 {invitation['email']}")
                
                with col2:
                    st.text(f"👤 {invitation['role'].title()}")
                
                with col3:
                    expires_at = invitation['expires_at']
                    if isinstance(expires_at, str):
                        expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                    days_left = (expires_at.replace(tzinfo=None) - datetime.now()).days
                    st.text(f"⏰ {days_left} days left")
                
                with col4:
                    if st.button("🗑️ Cancel", key=f"cancel_inv_{invitation['id']}"):
                        if cancel_user_invitation(invitation['id'], st.session_state['user_id']):
                            st.success("Invitation cancelled")
                            st.rerun()
                        else:
                            st.error("Failed to cancel invitation")
                
                st.divider()
    else:
        st.info("No pending invitations")


@require_admin
def show_notification_settings():
    """Notification settings management."""
    st.markdown("### 🔔 Notification Settings")
    
    # Get current notification settings
    current_settings = get_notification_settings(st.session_state['organization_id'])
    
    # Define available notification types
    notification_types = {
        'usage_warning': {
            'name': 'Usage Warnings',
            'description': 'Alerts when approaching usage limits (80% threshold)',
            'icon': '⚠️'
        },
        'experiment_failed': {
            'name': 'Experiment Failures',
            'description': 'Notifications when experiments fail',
            'icon': '❌'
        },
        'new_user_joined': {
            'name': 'New Users',
            'description': 'Notifications when new users join the organization',
            'icon': '👋'
        },
        'weekly_summary': {
            'name': 'Weekly Summary',
            'description': 'Weekly activity and usage summary',
            'icon': '📊'
        },
        'security_alerts': {
            'name': 'Security Alerts',
            'description': 'Security-related notifications and warnings',
            'icon': '🔒'
        }
    }
    
    # Show notification settings
    for notification_type, config in notification_types.items():
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{config['icon']} {config['name']}**")
                st.text(config['description'])
            
            with col2:
                current_enabled = current_settings.get(notification_type, {}).get('enabled', True)
                enabled = st.checkbox(
                    "Enabled",
                    value=current_enabled,
                    key=f"notify_{notification_type}"
                )
                
                if enabled != current_enabled:
                    if update_notification_setting(
                        st.session_state['organization_id'],
                        notification_type,
                        enabled,
                        admin_user_id=st.session_state['user_id']
                    ):
                        st.success(f"Updated {config['name']}")
                        st.rerun()
                    else:
                        st.error(f"Failed to update {config['name']}")
            
            st.divider()


@require_admin
def show_admin_audit_logs():
    """Admin audit logs display."""
    st.markdown("### 📋 Admin Audit Logs")
    
    # Get audit logs
    logs = get_org_admin_logs(st.session_state['organization_id'], 50)
    
    if logs:
        st.markdown(f"**Recent Admin Actions ({len(logs)} entries)**")
        
        for log in logs:
            with st.container():
                col1, col2, col3 = st.columns([4, 2, 2])
                
                with col1:
                    # Format action description
                    action_descriptions = {
                        'user_invited': '👥 User invited',
                        'invitation_cancelled': '🗑️ Invitation cancelled',
                        'organization_profile_updated': '🏢 Profile updated',
                        'usage_limits_updated': '📊 Usage limits updated',
                        'notification_setting_updated': '🔔 Notification setting updated',
                        'organization_settings_updated': '⚙️ Settings updated',
                        'user_status_updated': '👤 User status changed',
                        'user_admin_updated': '👑 Admin privileges changed'
                    }
                    
                    action_desc = action_descriptions.get(log['action_type'], f"🔧 {log['action_type']}")
                    st.markdown(f"**{action_desc}**")
                    
                    # Show details if available
                    if log.get('action_details'):
                        details = log['action_details']
                        if isinstance(details, dict):
                            if 'email' in details:
                                st.text(f"Target: {details['email']}")
                            elif 'user_email' in details:
                                st.text(f"Target: {details['user_email']}")
                
                with col2:
                    st.text(f"👤 {log['admin_name'] or log['admin_email']}")
                
                with col3:
                    timestamp = log['timestamp']
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    st.text(f"🕒 {timestamp.strftime('%Y-%m-%d %H:%M')}")
                
                st.divider()
    else:
        st.info("No audit logs found")


@require_admin
def show_data_export():
    """Data export functionality."""
    st.markdown("### 📦 Data Export")
    
    st.info("Export your organization's data for backup or migration purposes.")
    
    # Export options
    export_options = st.multiselect(
        "Select data to export:",
        ["Organization Profile", "Users", "Experiments", "Usage Data", "Settings"],
        default=["Organization Profile", "Users", "Experiments"]
    )
    
    if st.button("📥 Export Data", type="primary"):
        if export_options:
            with st.spinner("Exporting data..."):
                export_data = export_organization_data(st.session_state['organization_id'])
                
                if export_data:
                    # Convert to JSON for download
                    import json
                    json_data = json.dumps(export_data, indent=2, default=str)
                    
                    st.download_button(
                        label="💾 Download Export File",
                        data=json_data,
                        file_name=f"organization_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                    st.success("✅ Data export ready for download!")
                    
                    # Show export summary
                    st.markdown("#### Export Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Users Exported", len(export_data.get('users', [])))
                    with col2:
                        st.metric("Experiments Exported", len(export_data.get('experiments', [])))
                    with col3:
                        st.metric("Usage Records", len(export_data.get('usage_tracking', [])))
                else:
                    st.error("❌ Failed to export data")
        else:
            st.warning("Please select at least one data type to export")


def show_admin_page():
    """Main admin page router."""
    # Check if user is admin
    if not is_org_admin():
        st.error("⛔ Access Denied: Admin privileges required")
        return
    
    # Get current page from session state
    current_page = st.session_state.get('current_page', 'admin_dashboard')
    
    # Page routing
    if current_page == 'admin_users':
        show_user_management()
    elif current_page == 'admin_stats':
        show_admin_dashboard()
    elif current_page == 'admin_settings':
        show_organization_settings()
    elif current_page == 'admin_audit_logs':
        show_admin_audit_logs()
    else:
        show_admin_dashboard()