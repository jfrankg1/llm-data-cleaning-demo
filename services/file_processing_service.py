"""
File Processing Service
Handles file upload, categorization, and processing workflows
"""

import os
import time
import tempfile
import asyncio
import pandas as pd
import streamlit as st
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

# Import core processing functions
from src.dsaas2 import Claude_categorizer, extract_plate_block, DATA_PROMPT, MAP_PROMPT, PROTOCOL_PROMPT
from src.dsaas2_batch import batch_analyze_files
from auth.database import create_experiment, track_usage, update_experiment_status, save_experiment_results

# Import storage functionality if available
try:
    from auth.storage import upload_file, is_storage_configured
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False

# Import rate limiting if available
try:
    from config.environment import get_environment_config
    env_config = get_environment_config()
    RATE_LIMITING_AVAILABLE = env_config.get('RATE_LIMITING_AVAILABLE', False)
except Exception:
    RATE_LIMITING_AVAILABLE = False

if RATE_LIMITING_AVAILABLE:
    try:
        from auth.rate_limit_middleware import check_and_enforce_rate_limits
    except ImportError:
        RATE_LIMITING_AVAILABLE = False


class FileProcessingService:
    """Centralized file processing service"""
    
    def __init__(self):
        self.prompts = {
            'data': DATA_PROMPT,
            'map': MAP_PROMPT,
            'protocol': PROTOCOL_PROMPT
        }
        self.response_types = {
            'data': 'json',
            'map': 'json',
            'protocol': 'json'  # Changed from 'csv' to 'json' to match claude_api_service.py prompt
        }
    
    def check_rate_limits(self) -> bool:
        """Check rate limits before processing"""
        if RATE_LIMITING_AVAILABLE:
            try:
                check_and_enforce_rate_limits()
                return True
            except Exception as e:
                st.error(f"Rate limiting error: {e}")
                return False
        return True
    
    def upload_files_to_storage(self, file_list: List[Tuple[str, Any]], 
                               organization_id: str, experiment_id: str) -> List[Tuple[str, str]]:
        """Upload files to cloud storage if available"""
        uploaded_file_urls = []
        
        if not STORAGE_AVAILABLE or not is_storage_configured():
            return uploaded_file_urls
        
        for filename, uploaded_file in file_list:
            try:
                # Create a copy of the file content to avoid closing the original
                uploaded_file.seek(0)  # Reset file pointer
                file_content = uploaded_file.read()
                
                # Create a new BytesIO object for upload
                import io
                file_for_upload = io.BytesIO(file_content)
                
                file_url = upload_file(
                    file_for_upload,
                    f"original_{filename}",
                    organization_id,
                    experiment_id
                )
                uploaded_file_urls.append((filename, file_url))
                
                # Reset original file pointer for later use
                uploaded_file.seek(0)
                
                # Track storage usage
                track_usage(
                    organization_id,
                    st.session_state['user_id'],
                    'storage',
                    len(file_content)
                )
                
            except Exception as e:
                error_msg = str(e)
                print(f"CLOUD STORAGE ERROR for {filename}: {error_msg}")
                
                # Provide user-friendly error message
                if "object cannot be re-sized" in error_msg.lower():
                    st.warning(f"⚠️ Cloud storage temporary issue with {filename}: File processing will continue with local storage.")
                elif "access denied" in error_msg.lower() or "forbidden" in error_msg.lower():
                    st.warning(f"⚠️ Cloud storage access issue with {filename}: Please check storage permissions.")
                elif "timeout" in error_msg.lower():
                    st.warning(f"⚠️ Cloud storage timeout for {filename}: Network issue, file processing continues locally.")
                elif "closed file" in error_msg.lower():
                    st.warning(f"⚠️ Cloud storage file handling issue with {filename}: File processing continues locally.")
                else:
                    st.warning(f"⚠️ Cloud storage issue with {filename}: {error_msg[:100]}... File processing continues locally.")
        
        return uploaded_file_urls
    
    def prepare_files(self, file_list: List[Tuple[str, Any]], progress_bar, progress_text) -> Tuple[List[Tuple[str, str]], str]:
        """Prepare uploaded files for processing"""
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        progress_text.text("Preparing files...")
        for i, (filename, uploaded_file) in enumerate(file_list):
            file_path = os.path.join(temp_dir, filename)
            file_content = uploaded_file.getbuffer()
            with open(file_path, 'wb') as f:
                f.write(file_content)
            file_paths.append((file_path, os.path.splitext(filename)[1].lower().lstrip('.')))
            
            progress_bar.progress((i + 1) / len(file_list) * 0.2)
        
        return file_paths, temp_dir
    
    def categorize_files(self, file_paths: List[Tuple[str, str]], organization_id: str, 
                        user_id: str, progress_bar, status_placeholder) -> Dict[str, List[Tuple[str, str]]]:
        """Categorize files using Claude AI"""
        status_placeholder.info("🔍 Analyzing file types...")
        
        categorized = Claude_categorizer(
            file_paths,
            organization_id=organization_id,
            user_id=user_id
        )
        progress_bar.progress(0.3)
        
        # Show categorization results
        categorization_summary = []
        for category, files in categorized.items():
            if files:
                categorization_summary.append(f"**{category.capitalize()}**: {len(files)} files")
        
        status_placeholder.success("✅ Categorization complete!\n\n" + "\n\n".join(categorization_summary))
        
        return categorized
    
    def process_single_file(self, file_path: str, category: str, organization_id: str, user_id: str) -> Optional[pd.DataFrame]:
        """Process a single file"""
        try:
            from services.claude_api_service import get_claude_service
            import json
            
            # Track API usage
            track_usage(organization_id, user_id, 'api_call')
            
            # Get Claude service
            claude_service = get_claude_service()
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Get analysis from Claude
            analysis_response = claude_service.analyze_with_claude(
                content=content,
                analysis_type=category,
                organization_id=organization_id,
                user_id=user_id
            )
            
            # For protocol files, pass raw response to data service (it handles JSON extraction)
            if category == 'protocol':
                analysis = analysis_response  # Pass raw string response
            else:
                # Parse the Claude response as JSON for data/map files
                try:
                    analysis = json.loads(analysis_response)
                except json.JSONDecodeError:
                    # Try to extract JSON from text response
                    import re
                    json_match = re.search(r'\{.*\}', analysis_response, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group())
                    else:
                        st.warning(f"Failed to parse Claude response for {os.path.basename(file_path)}")
                        return None
            
            # Check if this is a well-based format file (like row_a_critical_test.csv)
            df = pd.read_csv(file_path)
            if 'Well' in df.columns and all(df['Well'].str.match(r'^[A-H]\d+$', na=False)):
                # This is already in well-based format, process directly
                result = self._process_well_based_file(df, file_path, category)
                if result is not None and 'Plate ID' in result.columns:
                    result['Plate ID'] = result['Plate ID'].astype(str)
                return result
            
            # Process the analysis response using data processing service
            from services.data_processing_service import get_data_processing_service
            data_service = get_data_processing_service()
            
            result = data_service.process_file(
                file_path, 
                analysis,
                file_type=category
            )
            
            if result is not None and 'Plate ID' in result.columns:
                result['Plate ID'] = result['Plate ID'].astype(str)
            
            return result
            
        except Exception as e:
            st.warning(f"Failed to process {os.path.basename(file_path)}: {str(e)}")
            return None
    
    def _process_well_based_file(self, df: pd.DataFrame, file_path: str, category: str) -> Optional[pd.DataFrame]:
        """Process files that are already in well-based format"""
        try:
            # Extract plate ID from filename if not present
            plate_id = os.path.splitext(os.path.basename(file_path))[0]
            
            # Create standardized format
            result_data = []
            
            for _, row in df.iterrows():
                well_id = row['Well']
                
                # Create row based on category
                if category == 'data':
                    # For data files, include all value columns
                    row_data = {
                        'Plate ID': plate_id,
                        'Well ID': well_id
                    }
                    
                    # Add data columns (exclude Well column)
                    for col in df.columns:
                        if col != 'Well':
                            row_data[f'raw {col.lower()}'] = row[col]
                    
                    result_data.append(row_data)
                
                elif category == 'map':
                    # For map files, create mapping format
                    result_data.append({
                        'Plate ID': plate_id,
                        'Well ID': well_id,
                        'raw mapping': row.get('Treatment', row.get('Sample', 'Unknown'))
                    })
            
            return pd.DataFrame(result_data)
            
        except Exception as e:
            st.warning(f"Failed to process well-based file {os.path.basename(file_path)}: {str(e)}")
            return None
    
    def process_batch_files(self, file_paths: List[str], category: str, organization_id: str, 
                           user_id: str) -> List[pd.DataFrame]:
        """Process multiple files using batch processing"""
        try:
            # Run async batch processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            batch_results = loop.run_until_complete(
                batch_analyze_files(
                    file_paths, 
                    self.prompts[category], 
                    self.response_types[category],
                    organization_id=organization_id,
                    user_id=user_id
                )
            )
            loop.close()
            
            # Process results
            category_dfs = []
            for batch_result in batch_results:
                if batch_result['success']:
                    try:
                        # Convert analysis to DataFrame
                        analysis = batch_result['data']
                        
                        # Use data processing service for all categories
                        from services.data_processing_service import get_data_processing_service
                        data_service = get_data_processing_service()
                        
                        if category == 'protocol':
                            # Protocol files now use JSON format, process accordingly
                            df = data_service._process_protocol_file(analysis)
                        elif 'raw_data_indices' in analysis:
                            df = extract_plate_block(batch_result['file_path'], analysis, 'data')
                        elif 'raw_mapping_indices' in analysis:
                            df = extract_plate_block(batch_result['file_path'], analysis, 'map')
                        else:
                            continue
                        
                        # Convert Plate ID to string
                        if 'Plate ID' in df.columns:
                            df['Plate ID'] = df['Plate ID'].astype(str)
                        category_dfs.append(df)
                        
                    except Exception as e:
                        st.warning(f"Failed to process {os.path.basename(batch_result['file_path'])}: {str(e)}")
            
            return category_dfs
            
        except Exception as e:
            st.error(f"Batch processing failed: {str(e)}")
            return []
    
    def process_categorized_files(self, categorized: Dict[str, List[Tuple[str, str]]], 
                                 organization_id: str, user_id: str, progress_bar, 
                                 status_placeholder) -> Dict[str, pd.DataFrame]:
        """Process all categorized files"""
        results = {}
        total_files_to_process = sum(len(files) for cat, files in categorized.items() if cat != 'other')
        files_processed = 0
        
        # Process files by category
        for category, cat_files in categorized.items():
            if category != 'other' and cat_files:
                status_placeholder.info(f"⚡ Processing {len(cat_files)} {category} files...")
                
                # Extract file paths
                file_paths_only = [file_path for file_path, _ in cat_files]
                
                if st.session_state.get('batch_processing_enabled', True) and len(file_paths_only) > 1:
                    # Use batch processing
                    category_dfs = self.process_batch_files(file_paths_only, category, organization_id, user_id)
                    files_processed += len(file_paths_only)
                    
                    if category_dfs:
                        if len(category_dfs) > 1:
                            # Combine multiple DataFrames
                            results[category] = pd.concat(category_dfs, ignore_index=True)
                        else:
                            results[category] = category_dfs[0]
                
                else:
                    # Process files individually
                    individual_dfs = []
                    for file_path, _ in cat_files:
                        df = self.process_single_file(file_path, category, organization_id, user_id)
                        if df is not None:
                            individual_dfs.append(df)
                        files_processed += 1
                        progress_bar.progress(0.3 + (files_processed / total_files_to_process) * 0.5)
                    
                    if individual_dfs:
                        if len(individual_dfs) > 1:
                            results[category] = pd.concat(individual_dfs, ignore_index=True)
                        else:
                            results[category] = individual_dfs[0]
                
                progress_bar.progress(0.3 + (files_processed / total_files_to_process) * 0.5)
        
        return results
    
    def combine_and_finalize_results(self, results: Dict[str, pd.DataFrame], temp_dir: str,
                                   progress_bar, status_placeholder) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Combine results and prepare final output"""
        status_placeholder.info("🔄 Combining results...")
        
        # Log what results we received
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"combine_and_finalize_results called with categories: {list(results.keys())}")
        for category, df in results.items():
            if df is not None:
                logger.info(f"  {category}: {len(df)} rows, {len(df.columns)} columns")
            else:
                logger.warning(f"  {category}: None (empty result)")
        
        # Clean up temporary directory
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Could not clean up temp directory: {e}")
        
        # Combine results if we have data
        final_df = None
        summary_stats = {}
        
        if 'data' in results and 'map' in results:
            # Merge data, map, and protocol (if available)
            try:
                from src.dsaas2 import combine_results
                combined_results = {'data': results['data'], 'map': results['map']}
                
                # Include protocol data if available
                if 'protocol' in results:
                    logger.info("Including protocol data in combination")
                    combined_results['protocol'] = results['protocol']
                else:
                    logger.info("No protocol data available for combination")
                
                final_df = combine_results(combined_results)
                
                # Calculate summary statistics
                protocol_columns = len([col for col in final_df.columns if col.startswith('protocol_')])
                summary_stats = {
                    'total_samples': len(final_df),
                    'unique_plates': final_df['Plate ID'].nunique() if 'Plate ID' in final_df.columns else 0,
                    'data_columns': len([col for col in final_df.columns if col not in ['Well Position', 'Plate ID', 'Sample Name'] and not col.startswith('protocol_')]),
                    'protocol_columns': protocol_columns,
                    'has_protocol': 'protocol' in results
                }
                
                if protocol_columns > 0:
                    logger.info(f"Successfully included {protocol_columns} protocol columns in final result")
                    status_placeholder.info(f"✅ Combined data, map, and protocol: {summary_stats['total_samples']} wells, {protocol_columns} protocol fields")
                
            except Exception as e:
                st.error(f"Error combining data and map: {str(e)}")
                # Fall back to just data
                final_df = results.get('data')
        
        elif 'data' in results:
            # Data-only workflow, but check if we can merge with protocol
            if 'protocol' in results:
                logger.info("Combining data with protocol (no map available)")
                try:
                    from src.dsaas2 import combine_results
                    combined_results = {'data': results['data'], 'protocol': results['protocol']}
                    final_df = combine_results(combined_results)
                    
                    protocol_columns = len([col for col in final_df.columns if col.startswith('protocol_')])
                    summary_stats = {
                        'total_samples': len(final_df),
                        'unique_plates': final_df['Plate ID'].nunique() if 'Plate ID' in final_df.columns else 0,
                        'data_columns': len([col for col in final_df.columns if not col.startswith('protocol_') and col not in ['Plate ID', 'Well ID']]),
                        'protocol_columns': protocol_columns,
                        'has_protocol': True,
                        'file_type': 'data_with_protocol'
                    }
                    logger.info(f"Successfully combined data with protocol: {protocol_columns} protocol columns")
                    status_placeholder.info(f"✅ Combined data with protocol: {summary_stats['total_samples']} wells, {protocol_columns} protocol fields")
                except Exception as e:
                    logger.error(f"Failed to combine data with protocol: {e}")
                    # Fall back to data only
                    final_df = results['data']
                    summary_stats = {
                        'total_samples': len(final_df),
                        'unique_plates': final_df['Plate ID'].nunique() if 'Plate ID' in final_df.columns else 0,
                        'data_columns': len(final_df.columns),
                        'has_protocol': 'protocol' in results
                    }
            else:
                logger.info("Data-only workflow (no map or protocol)")
                final_df = results['data']
                summary_stats = {
                    'total_samples': len(final_df),
                    'unique_plates': final_df['Plate ID'].nunique() if 'Plate ID' in final_df.columns else 0,
                    'data_columns': len(final_df.columns),
                    'has_protocol': False
                }
        
        elif 'map' in results:
            # Map-only workflow, but check if we can merge with protocol
            if 'protocol' in results:
                logger.info("Combining map with protocol (no data available)")
                try:
                    from src.dsaas2 import combine_results
                    combined_results = {'map': results['map'], 'protocol': results['protocol']}
                    final_df = combine_results(combined_results)
                    
                    protocol_columns = len([col for col in final_df.columns if col.startswith('protocol_')])
                    summary_stats = {
                        'total_samples': len(final_df),
                        'unique_plates': final_df['Plate ID'].nunique() if 'Plate ID' in final_df.columns else 0,
                        'map_columns': len([col for col in final_df.columns if not col.startswith('protocol_') and col not in ['Plate ID', 'Well ID']]),
                        'protocol_columns': protocol_columns,
                        'has_protocol': True,
                        'file_type': 'map_with_protocol'
                    }
                    logger.info(f"Successfully combined map with protocol: {protocol_columns} protocol columns")
                    status_placeholder.info(f"✅ Combined map with protocol: {summary_stats['total_samples']} wells, {protocol_columns} protocol fields")
                except Exception as e:
                    logger.error(f"Failed to combine map with protocol: {e}")
                    # Fall back to map only
                    final_df = results['map']
                    summary_stats = {
                        'total_samples': len(final_df),
                        'unique_plates': final_df['Plate ID'].nunique() if 'Plate ID' in final_df.columns else 0,
                        'map_columns': len(final_df.columns),
                        'has_protocol': False
                    }
            else:
                logger.info("Map-only workflow (no data or protocol)")
                final_df = results['map']
                summary_stats = {
                    'total_samples': len(final_df),
                    'unique_plates': final_df['Plate ID'].nunique() if 'Plate ID' in final_df.columns else 0,
                    'map_columns': len(final_df.columns),
                    'has_protocol': False
                }
        
        elif 'protocol' in results:
            # Support protocol-only files
            logger.info("Processing protocol-only file workflow")
            final_df = results['protocol']
            summary_stats = {
                'total_samples': len(final_df),
                'unique_plates': final_df['Plate ID'].nunique() if 'Plate ID' in final_df.columns else 0,
                'protocol_columns': len([col for col in final_df.columns if col.startswith('protocol_')]),
                'has_protocol': True,
                'file_type': 'protocol_only'
            }
            logger.info(f"Protocol-only result: {summary_stats['total_samples']} wells, {summary_stats['unique_plates']} plates, {summary_stats['protocol_columns']} protocol columns")
            status_placeholder.info(f"✅ Protocol-only processing: {summary_stats['total_samples']} wells, {summary_stats['unique_plates']} plates")
        
        else:
            # No valid data found
            logger.warning(f"No valid data found in results. Available categories: {list(results.keys())}")
            status_placeholder.warning("⚠️ No data, map, or protocol files were successfully processed.")
        
        logger.info(f"combine_and_finalize_results returning: final_df={'DataFrame' if final_df is not None else 'None'}, summary_stats={summary_stats}")
        progress_bar.progress(0.9)
        return final_df, summary_stats
    
    def process_uploaded_files_with_persistence(self, file_list: List[Tuple[str, Any]], 
                                               experiment_name: str, user_id: str, 
                                               organization_id: str) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Main file processing workflow with database persistence
        Returns: (success, final_dataframe, summary_stats)
        """
        
        # Check rate limits
        if not self.check_rate_limits():
            return False, None, {}
        
        try:
            start_time = time.time()
            
            # Create experiment record
            exp_id = create_experiment(
                user_id,
                organization_id,
                experiment_name,
                f"Processing {len(file_list)} files"
            )
            
            # Create progress containers
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                progress_text = st.empty()
            
            with status_container:
                st.markdown("### Processing Status")
                status_placeholder = st.empty()
            
            # Step 1: Prepare files
            file_paths, temp_dir = self.prepare_files(file_list, progress_bar, progress_text)
            
            # Step 2: Upload to storage (optional)
            uploaded_file_urls = self.upload_files_to_storage(file_list, organization_id, exp_id)
            
            # Step 3: Categorize files
            progress_text.text("Categorizing files with Claude...")
            categorized = self.categorize_files(file_paths, organization_id, user_id, progress_bar, status_placeholder)
            
            # Step 4: Process files
            progress_text.text("Processing files...")
            results = self.process_categorized_files(categorized, organization_id, user_id, progress_bar, status_placeholder)
            
            # Step 5: Combine and finalize
            final_df, summary_stats = self.combine_and_finalize_results(results, temp_dir, progress_bar, status_placeholder)
            
            # Complete progress
            progress_bar.progress(1.0)
            progress_text.text("✅ Processing complete!")
            
            # Update experiment with results
            processing_time = time.time() - start_time
            summary_stats['processing_time'] = processing_time
            summary_stats['experiment_id'] = exp_id
            
            if final_df is not None:
                # Save results to database
                try:
                    # Convert DataFrame to CSV string
                    csv_string = final_df.to_csv(index=False)
                    
                    # Save results file
                    file_path = save_experiment_results(exp_id, csv_string, 'results.csv')
                    
                    # Update experiment status with results
                    update_experiment_status(
                        exp_id,
                        'completed',
                        file_path,
                        summary_stats
                    )
                    
                    # Store results in session state for display
                    st.session_state['experiment_results'] = {
                        'experiment_id': exp_id,
                        'dataframe': final_df,
                        'summary_stats': summary_stats,
                        'file_path': file_path
                    }
                    
                    status_placeholder.success(f"🎉 **Processing Complete!**\n\n"
                                             f"📊 **{summary_stats['total_samples']}** samples processed\n\n"
                                             f"🧪 **{summary_stats['unique_plates']}** unique plates\n\n"
                                             f"⏱️ Processing time: **{processing_time:.1f}s**\n\n"
                                             f"💾 Results saved to database")
                    
                except Exception as e:
                    st.error(f"Failed to save results to database: {str(e)}")
                    # Update experiment as failed - use repository directly for error_message
                    from auth.repositories import get_experiment_repository
                    exp_repo = get_experiment_repository()
                    exp_repo.update_experiment_status(exp_id, 'failed', error_message=str(e))
                    return False, None, {}
            else:
                status_placeholder.warning("⚠️ Processing completed but no data was extracted.")
                # Update experiment as failed - use repository directly for error_message
                from auth.repositories import get_experiment_repository
                exp_repo = get_experiment_repository()
                exp_repo.update_experiment_status(exp_id, 'failed', error_message="No data was extracted from files")
                return False, None, {}
            
            return True, final_df, summary_stats
            
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            return False, None, {}


# Global instance
file_processing_service = FileProcessingService()