"""
LLM Data Cleaning System - Main Application
Refactored version using service layer architecture
"""

import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import services
from services.service_container import get_auth_service, get_file_processing_service, get_experiment_service

# Import UI components
from ui.auth_forms_fixed import show_auth_page, show_logout_button
from ui.experiment_ui import show_experiment_history, show_new_experiment_form

# Import database functions for sidebar info
try:
    from auth.database import get_organization_usage, get_usage_warnings
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    st.error("Database module not available. Please check configuration.")
    st.stop()

# Import admin functions
try:
    from auth.permissions import is_org_admin, admin_sidebar_menu
    from streamlit_app_admin import show_admin_page
    ADMIN_AVAILABLE = True
except ImportError as e:
    ADMIN_AVAILABLE = False
    # Only print once on startup
    if 'admin_import_logged' not in st.session_state:
        print(f"Admin module not available: {e}")
        st.session_state['admin_import_logged'] = True

# Import rate limiting if available
try:
    from config.environment import get_environment_config
    env_config = get_environment_config()
    RATE_LIMITING_AVAILABLE = env_config.get('RATE_LIMITING_AVAILABLE', False)
except Exception:
    RATE_LIMITING_AVAILABLE = False

if RATE_LIMITING_AVAILABLE:
    try:
        from auth.rate_limit_middleware import show_rate_limit_status
    except ImportError:
        RATE_LIMITING_AVAILABLE = False

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LLM Data Cleaning System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
}

.status-processing { color: #ffa500; }
.status-completed { color: #28a745; }
.status-failed { color: #dc3545; }

.metric-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
}

.error-message {
    color: #ff4444;
    font-size: 0.9em;
    margin-top: 0.25rem;
    margin-bottom: 0.5rem;
    padding: 0.25rem;
    background-color: #ffe6e6;
    border-radius: 4px;
    border-left: 3px solid #ff4444;
}

.sidebar-section {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


def show_sidebar():
    """Display sidebar with user info and navigation"""
    auth_service = get_auth_service()
    
    if not auth_service.is_authenticated():
        return
    
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    
    # User info
    user = auth_service.get_current_user()
    st.sidebar.markdown(f"**👤 Welcome, {user['username']}**")
    st.sidebar.markdown(f"🏢 {user['organization_name']}")
    
    if user['is_admin']:
        st.sidebar.markdown("👑 **Organization Admin**")
    if user['is_system_admin']:
        st.sidebar.markdown("⚡ **System Admin**")
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Usage information
    if DB_AVAILABLE:
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.sidebar.markdown("### 📊 Usage Status")
        
        try:
            usage = get_organization_usage(user['organization_id'])
            warnings = get_usage_warnings(user['organization_id'])
            
            # Show usage metrics
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Experiments", usage.get('experiment_count', 0))
                st.metric("API Calls", usage.get('api_calls', 0))
            with col2:
                st.metric("Storage (MB)", f"{usage.get('storage_used', 0) / 1024 / 1024:.1f}")
                experiments_limit = usage.get('experiments_limit', 100)
                st.metric("Limit", f"{experiments_limit}")
            
            # Show warnings
            if warnings:
                for warning in warnings:
                    if warning['type'] == 'approaching_limit':
                        st.sidebar.warning(f"⚠️ {warning['message']}")
                    elif warning['type'] == 'near_limit':
                        st.sidebar.error(f"🚨 {warning['message']}")
                        
        except Exception as e:
            st.sidebar.error(f"Error loading usage data: {str(e)}")
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Rate limiting status
    if RATE_LIMITING_AVAILABLE:
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.sidebar.markdown("### ⚡ Rate Limits")
        try:
            show_rate_limit_status()
        except Exception as e:
            st.sidebar.error(f"Rate limit status unavailable: {str(e)}")
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Admin menu if user is org admin
    if ADMIN_AVAILABLE and is_org_admin():
        admin_sidebar_menu()
    
    # Logout button
    show_logout_button()


def main_app():
    """Main application interface for authenticated users"""
    
    # Check if admin page is requested
    if st.session_state.get('current_page', 'main').startswith('admin_'):
        if ADMIN_AVAILABLE:
            show_admin_page()
        else:
            st.error("Admin module not available")
        return
    
    st.markdown("""
    <div class="main-header">
        <h1>🧬 LLM Data Cleaning System</h1>
        <p>Transform messy lab data into clean, analysis-ready datasets</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🧪 New Analysis", "📊 Experiment History", "⏰ Time-Series Analysis", "ℹ️ API Access"])
    
    with tab1:
        show_new_experiment_form()
    
    with tab2:
        show_experiment_history()
    
    with tab3:
        show_time_series_analysis()
    
    with tab4:
        show_api_access()


def show_api_access():
    """Show API access information and documentation"""
    st.subheader("🔌 API Access")
    
    st.markdown("""
    ### API Documentation
    
    The LLM Data Cleaning System provides a REST API for programmatic access.
    
    **Base URL:** `https://your-domain.com/api/v1`
    
    #### Authentication
    All API requests require authentication using your API key:
    
    ```bash
    curl -H "Authorization: Bearer YOUR_API_KEY" \\
         https://your-domain.com/api/v1/experiments
    ```
    
    #### Endpoints
    
    - `GET /experiments` - List your experiments
    - `POST /experiments` - Create new experiment
    - `GET /experiments/{id}` - Get experiment details
    - `POST /experiments/{id}/files` - Upload files to experiment
    - `GET /experiments/{id}/results` - Download experiment results
    
    #### Example: Create Experiment
    
    ```python
    import requests
    
    headers = {"Authorization": "Bearer YOUR_API_KEY"}
    data = {
        "name": "My Experiment",
        "description": "Analysis of plate data"
    }
    
    response = requests.post(
        "https://your-domain.com/api/v1/experiments",
        headers=headers,
        json=data
    )
    ```
    """)
    
    # API key management (placeholder)
    st.markdown("### 🔑 API Key Management")
    st.info("API key management will be available in a future release.")


def show_time_series_analysis():
    """Show time-series analysis interface"""
    st.subheader("⏰ Time-Series Analysis")
    
    st.markdown("""
    ### 📊 IoT Device Log Analysis
    
    This module analyzes log files from IoT devices, automated equipment, and monitoring systems.
    It can handle both **time-based** and **event-based** log files, automatically detecting:
    
    - 📅 Timestamp columns and formats
    - 🔄 Event sequences and patterns  
    - 📈 Time-series data alignment
    - 🔗 Multi-file log correlation
    
    **Supported Log Types:**
    - Manufacturing equipment logs
    - Environmental monitoring data
    - Laboratory instrument logs
    - System event logs
    - Sensor data streams
    """)
    
    # Import the time-series service
    try:
        from services.time_series_service import get_time_series_service
        time_series_service = get_time_series_service()
    except ImportError as e:
        st.error(f"Time-series service not available: {e}")
        return
    
    # Check authentication
    if not st.session_state.get('user_id'):
        st.error("Please log in to use time-series analysis")
        return
    
    # Show current results if available
    if 'time_series_results' in st.session_state:
        show_time_series_results()
    
    # Experiment name input - SECURED with validation
    experiment_name_input = st.text_input(
        "Analysis Name",
        value="",
        placeholder="e.g., Equipment Monitoring 2024-01-15",
        help="Give your time-series analysis a descriptive name",
        max_chars=200
    )
    
    # Validate experiment name
    if experiment_name_input:
        from auth.input_validation import validate_and_sanitize_input
        validation_result = validate_and_sanitize_input('experiment', experiment_name_input)
        if not validation_result.is_valid:
            for error in validation_result.errors:
                st.error(f"❌ Analysis Name: {error}")
            experiment_name = ""  # Clear invalid input
        else:
            experiment_name = validation_result.sanitized_value
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    st.warning(f"⚠️ Analysis Name: {warning}")
    else:
        experiment_name = ""
    
    # File upload specifically for log files
    uploaded_files = st.file_uploader(
        "Upload Log Files",
        accept_multiple_files=True,
        type=['csv', 'txt', 'tsv', 'log'],
        help="Upload IoT device logs, equipment monitoring files, or sensor data"
    )
    
    # Processing options
    with st.expander("⚙️ Analysis Options"):
        col1, col2 = st.columns(2)
        with col1:
            detect_time_zones = st.checkbox(
                "Auto-detect time zones",
                value=True,
                help="Automatically detect and handle different time zones"
            )
        with col2:
            align_timestamps = st.checkbox(
                "Align timestamps",
                value=True,
                help="Align all log files to a common time reference"
            )
    
    # File preview with enhanced info for log files
    if uploaded_files:
        st.markdown("### 📁 Uploaded Log Files")
        for file in uploaded_files:
            file_size = len(file.getbuffer())
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📄 {file.name}")
            with col2:
                st.text(f"{file_size:,} bytes")
    
    # Process button
    if st.button("🚀 Start Time-Series Analysis", type="primary", disabled=not (experiment_name and uploaded_files)):
        if not experiment_name.strip():
            st.error("Please enter an analysis name")
            return
        
        if not uploaded_files:
            st.error("Please upload at least one log file")
            return
        
        # Process files
        with st.spinner("🔄 Processing log files..."):
            try:
                # Use the time-series service to process files
                results = time_series_service.process_time_series_files(
                    uploaded_files=uploaded_files,
                    organization_id=st.session_state['organization_id'],
                    user_id=st.session_state['user_id'],
                    experiment_name=experiment_name
                )
                
                # Store results in session state
                st.session_state['time_series_results'] = results
                
                # Show results immediately
                show_time_series_results()
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.exception(e)


def show_time_series_results():
    """Show time-series analysis results"""
    if 'time_series_results' not in st.session_state:
        return
    
    results = st.session_state['time_series_results']
    
    st.success("✅ Time-series analysis completed!")
    
    # Show summary statistics
    summary = results.get('processing_summary', {})
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Files Processed", summary.get('processed_successfully', 0))
        with col2:
            st.metric("Total Files", summary.get('total_files', 0))
        with col3:
            st.metric("Success Rate", f"{(summary.get('processed_successfully', 0) / max(summary.get('total_files', 1), 1) * 100):.1f}%")
        with col4:
            experiment_id = results.get('experiment_id', 'N/A')
            st.metric("Experiment ID", experiment_id[:8] + "..." if len(str(experiment_id)) > 8 else str(experiment_id))
    
    # Show analysis results
    alignment_analysis = results.get('alignment_analysis', {})
    if alignment_analysis:
        st.markdown("### 📊 Structural Analysis")
        with st.expander("🔍 View File Analysis Details"):
            st.json(alignment_analysis)
    
    # Show unified data if available
    unified_data = results.get('unified_data')
    if unified_data is not None:
        st.markdown("### 📈 Unified Time-Series Data")
        st.dataframe(unified_data.head(20))
        
        # Download button for unified data
        csv_data = unified_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Unified Data",
            data=csv_data,
            file_name=f"unified_timeseries_{experiment_id}.csv",
            mime="text/csv"
        )
    
    # Show downloadable results
    downloadable_results = results.get('downloadable_results', {})
    if downloadable_results:
        st.markdown("### 📥 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'analysis_summary_json' in downloadable_results:
                st.download_button(
                    label="📋 Analysis Summary (JSON)",
                    data=downloadable_results['analysis_summary_json'],
                    file_name=f"analysis_summary_{experiment_id}.json",
                    mime="application/json"
                )
        
        with col2:
            if 'processing_summary_json' in downloadable_results:
                st.download_button(
                    label="📊 Processing Summary (JSON)",
                    data=downloadable_results['processing_summary_json'],
                    file_name=f"processing_summary_{experiment_id}.json",
                    mime="application/json"
                )
    
    # Show any errors
    if results.get('failed_files'):
        st.markdown("### ❌ Failed Files")
        for failed_file in results['failed_files']:
            st.error(f"❌ {failed_file['path']}: {failed_file['error']}")
    
    # Clear results button
    if st.button("🗑️ Clear Results"):
        del st.session_state['time_series_results']
        st.rerun()


def main():
    """Main application entry point"""
    # Initialize services
    auth_service = get_auth_service()
    
    # Initialize session state
    auth_service.init_session_state()
    
    # Show appropriate interface based on authentication status
    if auth_service.is_authenticated():
        # Show sidebar for authenticated users
        show_sidebar()
        
        # Show main application
        main_app()
    else:
        # Show authentication page
        show_auth_page()


if __name__ == "__main__":
    main()