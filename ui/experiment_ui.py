"""
Experiment UI Components
Displays experiment history, statistics, and management interface
"""

import streamlit as st
from services.experiment_service import experiment_service
from auth.input_validation import validate_and_sanitize_input


def show_experiment_history():
    """Display user's experiment history with enhanced UI"""
    st.subheader("📊 Your Experiments")
    
    # Get user's experiments
    user_id = st.session_state.get('user_id')
    if not user_id:
        st.error("User not authenticated")
        return
    
    experiments = experiment_service.get_user_experiments(user_id)
    
    if not experiments:
        st.info("🔬 No experiments yet. Upload files in the 'New Analysis' tab to get started!")
        return
    
    # Show experiment statistics
    stats = experiment_service.get_experiment_statistics(user_id)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Experiments", stats['total_experiments'])
    with col2:
        st.metric("Completed", stats['completed_experiments'])
    with col3:
        st.metric("Failed", stats['failed_experiments'])
    with col4:
        st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
    
    st.markdown("---")
    
    # Display experiments
    for exp in experiments:
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(f"### {exp['name']}")
                st.text(f"ID: {exp['id'][:8]}...")
            
            with col2:
                status_color = {
                    'processing': '🟡',
                    'completed': '🟢',
                    'failed': '🔴'
                }.get(exp['status'], '⚪')
                
                st.markdown(f"{status_color} **Status:** {exp['status'].title()}")
                st.text(f"Created: {exp['created_at'].strftime('%Y-%m-%d %H:%M')}")
            
            with col3:
                if exp['status'] == 'completed' and exp.get('result_file_url'):
                    if st.button("📥 Download", key=f"download_{exp['id']}"):
                        success, data, filename = experiment_service.download_experiment_results(exp)
                        
                        if success:
                            if data.startswith('http'):
                                # Presigned URL
                                st.markdown(f"[📥 Download Results]({data})")
                            else:
                                # Direct file content
                                st.download_button(
                                    label="📥 Download Results",
                                    data=data,
                                    file_name=filename,
                                    mime="text/csv"
                                )
                        else:
                            st.error(filename)  # filename contains error message on failure
            
            # Show experiment details in expandable section
            if exp.get('metadata'):
                with st.expander("📋 View Details"):
                    metadata = exp['metadata']
                    
                    # Create columns for metadata display
                    meta_col1, meta_col2 = st.columns(2)
                    
                    with meta_col1:
                        if 'processing_time' in metadata:
                            st.metric("Processing Time", f"{metadata['processing_time']:.1f}s")
                        
                        if 'files_processed' in metadata:
                            st.metric("Files Processed", metadata['files_processed'])
                    
                    with meta_col2:
                        if 'total_samples' in metadata:
                            st.metric("Total Samples", metadata['total_samples'])
                        
                        if 'unique_plates' in metadata:
                            st.metric("Unique Plates", metadata['unique_plates'])
                    
                    # Show file categories
                    if 'categories' in metadata and metadata['categories']:
                        st.markdown("**📁 File Categories:**")
                        for cat, count in metadata['categories'].items():
                            st.text(f"  • {cat.title()}: {count} files")
                    
                    # Show original files if available
                    if 'original_files' in metadata and metadata['original_files']:
                        st.markdown("**📎 Original Files:**")
                        for filename, file_url in metadata['original_files']:
                            if file_url and file_url.startswith('http'):
                                st.markdown(f"  • [{filename}]({file_url})")
                            else:
                                st.text(f"  • {filename}")
                    
                    # Show error details if failed
                    if exp['status'] == 'failed' and exp.get('error_message'):
                        st.markdown("**❌ Error Details:**")
                        st.error(exp['error_message'])
            
            st.markdown("---")
    
    # Export functionality
    st.markdown("### 📤 Export History")
    if st.button("Export Experiment History as CSV"):
        history_df = experiment_service.export_experiment_history(user_id)
        if not history_df.empty:
            csv_data = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History CSV",
                data=csv_data,
                file_name="experiment_history.csv",
                mime="text/csv"
            )
        else:
            st.info("No experiment history to export")


def show_experiment_statistics_dashboard():
    """Show detailed experiment statistics dashboard"""
    st.subheader("📈 Experiment Analytics")
    
    user_id = st.session_state.get('user_id')
    if not user_id:
        st.error("User not authenticated")
        return
    
    experiments = experiment_service.get_user_experiments(user_id, limit=100)
    
    if not experiments:
        st.info("No experiments found for analysis")
        return
    
    # Time-based analysis
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Convert to DataFrame for analysis
    exp_df = pd.DataFrame(experiments)
    exp_df['created_at'] = pd.to_datetime(exp_df['created_at'])
    exp_df['date'] = exp_df['created_at'].dt.date
    
    # Daily experiment counts
    daily_counts = exp_df.groupby('date').size().reset_index(name='count')
    
    if len(daily_counts) > 1:
        st.markdown("**📅 Daily Experiment Activity**")
        st.line_chart(daily_counts.set_index('date')['count'])
    
    # Status distribution
    status_counts = exp_df['status'].value_counts()
    if len(status_counts) > 0:
        st.markdown("**📊 Status Distribution**")
        st.bar_chart(status_counts)
    
    # Recent activity
    recent_experiments = exp_df[exp_df['created_at'] >= datetime.now() - timedelta(days=7)]
    if len(recent_experiments) > 0:
        st.markdown(f"**🕒 Recent Activity (Last 7 days): {len(recent_experiments)} experiments**")
        
        for _, exp in recent_experiments.iterrows():
            status_emoji = {'completed': '✅', 'failed': '❌', 'processing': '⏳'}.get(exp['status'], '❓')
            st.text(f"{status_emoji} {exp['name']} - {exp['created_at'].strftime('%m/%d %H:%M')}")


def show_experiment_management():
    """Show experiment management interface for admins"""
    if not st.session_state.get('is_admin'):
        st.warning("Admin access required")
        return
    
    st.subheader("🔧 Experiment Management")
    
    # Cleanup options
    with st.expander("🧹 Cleanup Options"):
        days_old = st.number_input("Delete failed experiments older than (days)", min_value=1, value=30)
        if st.button("Clean Up Old Failed Experiments"):
            user_id = st.session_state.get('user_id')
            cleaned = experiment_service.cleanup_old_experiments(user_id, days_old)
            st.success(f"Cleaned up {cleaned} old experiments")
    
    # Bulk operations
    with st.expander("📦 Bulk Operations"):
        st.info("Bulk operations coming soon...")


def show_current_experiment_results():
    """Show results from current session if available"""
    if 'experiment_results' in st.session_state:
        results_data = st.session_state['experiment_results']
        
        st.success("✅ Processing completed successfully!")
        
        # Show summary statistics
        summary = results_data.get('summary_stats', {})
        if summary:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", summary.get('total_samples', 0))
            with col2:
                st.metric("Unique Plates", summary.get('unique_plates', 0))
            with col3:
                st.metric("Processing Time", f"{summary.get('processing_time', 0):.1f}s")
            with col4:
                st.metric("Experiment ID", results_data.get('experiment_id', 'N/A')[:8] + "...")
        
        # Show processing method information if available
        processing_methods = results_data.get('processing_methods', {})
        if processing_methods:
            st.markdown("### 🔧 Processing Methods Used")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                smart_csv_count = processing_methods.get('smart_csv_count', 0)
                if smart_csv_count > 0:
                    st.success(f"⚡ **Smart CSV**: {smart_csv_count} files")
                    st.caption("Fast pandas-based processing")
                
            with col2:
                claude_count = processing_methods.get('claude_analysis_count', 0)
                if claude_count > 0:
                    st.info(f"🧠 **Claude AI**: {claude_count} files")
                    st.caption("AI-powered complex file analysis")
                    
            with col3:
                total_files = processing_methods.get('total_files', 0)
                if total_files > 0:
                    st.metric("Total Files", total_files)
                    
            # Show hybrid processing indicator
            if smart_csv_count > 0 and claude_count > 0:
                st.success("🚀 **Hybrid Processing**: Successfully combined fast and intelligent analysis methods")
            elif claude_count > 0:
                st.info("🧠 **AI-Powered Analysis**: Complex files processed using Claude AI")
            elif smart_csv_count > 0:
                st.success("⚡ **High-Speed Processing**: All files processed using optimized Smart CSV")
        
        # Show data preview
        st.markdown("### 📊 Results Preview")
        df = results_data.get('dataframe')
        if df is not None:
            st.dataframe(df.head(10))
            
            # Download button
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results",
                data=csv_data,
                file_name=f"experiment_{results_data.get('experiment_id', 'unknown')}_results.csv",
                mime="text/csv"
            )
        
        # Clear results button
        if st.button("🗑️ Clear Results"):
            del st.session_state['experiment_results']
            st.rerun()


def show_new_experiment_form():
    """Show form to create a new experiment"""
    st.subheader("🧪 New Analysis")
    
    # Show current results if available
    show_current_experiment_results()
    
    # Experiment name input - SECURED with validation
    experiment_name_input = st.text_input(
        "Experiment Name",
        value="",
        placeholder="e.g., Plate Analysis 2024-01-15",
        help="Give your experiment a descriptive name",
        max_chars=200
    )
    
    # Validate experiment name
    if experiment_name_input:
        validation_result = validate_and_sanitize_input('experiment', experiment_name_input)
        if not validation_result.is_valid:
            for error in validation_result.errors:
                st.error(f"❌ Experiment Name: {error}")
            experiment_name = ""  # Clear invalid input
        else:
            experiment_name = validation_result.sanitized_value
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    st.warning(f"⚠️ Experiment Name: {warning}")
    else:
        experiment_name = ""
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload your experimental files",
        accept_multiple_files=True,
        type=['csv', 'txt', 'pdf', 'docx', 'rtf', 'xlsx'],
        help="Upload data files, protocol files, and sample maps"
    )
    
    # Processing options
    with st.expander("⚙️ Processing Options"):
        batch_processing = st.checkbox(
            "Enable batch processing", 
            value=True,
            help="Process multiple files simultaneously for faster results"
        )
        st.session_state['batch_processing_enabled'] = batch_processing
    
    # File preview
    if uploaded_files:
        st.markdown("### 📁 Uploaded Files")
        for file in uploaded_files:
            file_size = len(file.getbuffer())
            st.text(f"📄 {file.name} ({file_size:,} bytes)")
    
    # Process button
    if st.button("🚀 Start Analysis", type="primary", disabled=not (experiment_name and uploaded_files)):
        if not experiment_name.strip():
            st.error("Please enter an experiment name")
            return
        
        # Validate and sanitize experiment name
        name_validation = validate_and_sanitize_input('experiment', experiment_name)
        if not name_validation.is_valid:
            for error in name_validation.errors:
                st.error(error)
            return
        
        # Show warnings for unusual characters
        for warning in name_validation.warnings:
            st.warning(f"⚠️ {warning}")
        
        # Use sanitized name
        sanitized_experiment_name = name_validation.sanitized_value
        
        if not uploaded_files:
            st.error("Please upload at least one file")
            return
        
        # Prepare file list
        file_list = [(file.name, file) for file in uploaded_files]
        
        # Import and use the file processing service
        from services.file_processing_service import file_processing_service
        
        # Process files with sanitized name
        success, final_df, summary_stats = file_processing_service.process_uploaded_files_with_persistence(
            file_list=file_list,
            experiment_name=sanitized_experiment_name,
            user_id=st.session_state['user_id'],
            organization_id=st.session_state['organization_id']
        )
        
        if success and final_df is not None:
            # Save results
            experiment_id = summary_stats.get('experiment_id')
            if experiment_id:
                # Results are already saved in the file processing service
                # Just mark the experiment as completed
                saved = True
                
                if saved:
                    st.success("✅ Experiment completed and results saved!")
                    
                    # Show results preview
                    st.markdown("### 📊 Results Preview")
                    st.dataframe(final_df.head())
                    
                    # Download button
                    csv_data = final_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results",
                        data=csv_data,
                        file_name=f"{sanitized_experiment_name.replace(' ', '_')}_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Processing completed but results could not be saved to database")
        else:
            st.error("Processing failed. Please check your files and try again.")