"""
Time Series Service - Service layer integration for time-series alignment
Follows the existing service layer pattern established in the codebase
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from datetime import datetime

# Import the main processor
from src.time_series_alignment import get_time_series_alignment_processor

# Import existing service dependencies
from .claude_api_service import get_claude_service
from .validation_service import get_validation_service
from .experiment_service import experiment_service

# Set up logging
logger = logging.getLogger(__name__)

class TimeSeriesService:
    """Service layer for time-series alignment functionality"""
    
    def __init__(self):
        """Initialize the time-series service"""
        self.processor = get_time_series_alignment_processor()
        self.claude_service = get_claude_service()
        self.validation_service = get_validation_service()
        self.experiment_service = experiment_service
    
    def process_time_series_files(
        self,
        uploaded_files: List[Any],
        organization_id: str,
        user_id: str,
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process uploaded time-series files through the complete workflow
        
        Args:
            uploaded_files: List of uploaded file objects (from Streamlit)
            organization_id: Organization ID for tracking and data isolation
            user_id: User ID for tracking
            experiment_name: Optional experiment name
            
        Returns:
            dict: Complete processing results
        """
        processing_results = {
            'experiment_id': None,
            'files_processed': 0,
            'successful_files': [],
            'failed_files': [],
            'alignment_analysis': {},
            'unified_data': None,
            'downloadable_results': {},
            'processing_summary': {
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'total_files': len(uploaded_files),
                'success_rate': 0.0
            }
        }
        
        try:
            # 1. Validate uploaded files
            validated_files = self._validate_uploaded_files(uploaded_files)
            if not validated_files:
                processing_results['error'] = "No valid files to process"
                return processing_results
            
            # 2. Create experiment record
            experiment_id = self._create_experiment_record(
                organization_id, user_id, experiment_name, validated_files
            )
            processing_results['experiment_id'] = experiment_id
            
            # 3. Save uploaded files temporarily
            temp_file_paths = self._save_uploaded_files(validated_files)
            
            # 4. Process files with the time-series processor
            alignment_results = self.processor.process_log_files(
                temp_file_paths, organization_id, user_id
            )
            
            # 5. Update processing results
            processing_results.update(alignment_results)
            
            # 6. Prepare downloadable results
            downloadable_results = self._prepare_downloadable_results(alignment_results)
            processing_results['downloadable_results'] = downloadable_results
            
            # 7. Save results to database
            self._save_experiment_results(
                experiment_id, processing_results, organization_id, user_id
            )
            
            # 8. Cleanup temporary files
            self._cleanup_temp_files(temp_file_paths)
            
            # 9. Update processing summary - merge service and processor summaries
            service_summary = processing_results['processing_summary']
            if 'start_time' not in service_summary:
                service_summary['start_time'] = service_summary.get('timestamp', datetime.now().isoformat())
            service_summary['end_time'] = datetime.now().isoformat()
            service_summary['success_rate'] = (
                service_summary.get('processed_successfully', 0) / 
                service_summary.get('total_files', 1)
            ) if service_summary.get('total_files', 0) > 0 else 0.0
            
            return processing_results
            
        except Exception as e:
            logger.error(f"Time-series processing failed: {e}")
            # Provide clear error message to user
            error_message = str(e)
            if "Claude AI analysis failed" in error_message:
                processing_results['error'] = "Claude AI analysis failed. Please check your API configuration and try again."
            elif "Time-series analysis cannot proceed" in error_message:
                processing_results['error'] = error_message
            else:
                processing_results['error'] = f"Processing failed: {error_message}"
            
            processing_results['processing_summary']['end_time'] = datetime.now().isoformat()
            return processing_results
    
    def _validate_uploaded_files(self, uploaded_files: List[Any]) -> List[Any]:
        """Validate uploaded files for time-series processing"""
        validated_files = []
        
        for file in uploaded_files:
            try:
                # Basic validation
                if not hasattr(file, 'name') or not hasattr(file, 'read'):
                    continue
                
                # Check file extension
                if not file.name.lower().endswith(('.csv', '.txt', '.tsv')):
                    logger.warning(f"Skipping non-CSV file: {file.name}")
                    continue
                
                # File size check (max 50MB for log files) - handle both real and mock objects
                if hasattr(file, 'size'):
                    try:
                        # Check if size is comparable (not a mock)
                        if isinstance(file.size, (int, float)) and file.size > 50 * 1024 * 1024:
                            logger.warning(f"Skipping large file: {file.name}")
                            continue
                    except (TypeError, AttributeError):
                        # Mock object or incomparable size, skip size check
                        pass
                
                validated_files.append(file)
                
            except Exception as e:
                logger.error(f"Error validating file {getattr(file, 'name', 'unknown')}: {e}")
                continue
        
        return validated_files
    
    def _create_experiment_record(
        self,
        organization_id: str,
        user_id: str,
        experiment_name: Optional[str],
        validated_files: List[Any]
    ) -> str:
        """Create a new experiment record for time-series processing"""
        try:
            # Generate experiment name if not provided
            if not experiment_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_name = f"Time Series Analysis {timestamp}"
            
            # Create experiment using existing service (note: only required parameters)
            experiment_id = self.experiment_service.create_new_experiment(
                user_id=user_id,
                organization_id=organization_id,
                experiment_name=experiment_name,
                description=f"Time-series analysis of {len(validated_files)} log files"
            )
            
            # Update experiment with metadata
            self.experiment_service.update_experiment(
                experiment_id=experiment_id,
                status='processing',
                metadata={
                    'file_names': [f.name for f in validated_files],
                    'processing_type': 'time_series_alignment',
                    'experiment_type': 'time_series_alignment',
                    'file_count': len(validated_files)
                }
            )
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment record: {e}")
            # Generate a fallback ID
            return f"ts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _save_uploaded_files(self, validated_files: List[Any]) -> List[str]:
        """Save uploaded files to temporary storage"""
        temp_file_paths = []
        
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        
        for file in validated_files:
            try:
                # Generate safe filename
                safe_filename = self.validation_service.sanitize_filename(file.name)
                temp_path = os.path.join(temp_dir, safe_filename)
                
                # Save file content
                with open(temp_path, 'wb') as f:
                    f.write(file.read())
                
                temp_file_paths.append(temp_path)
                
                # Reset file pointer if needed
                if hasattr(file, 'seek'):
                    file.seek(0)
                
            except Exception as e:
                logger.error(f"Failed to save file {file.name}: {e}")
                continue
        
        return temp_file_paths
    
    def _prepare_downloadable_results(self, alignment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for download"""
        downloadable_results = {}
        
        try:
            # Prepare unified data CSV if available
            if alignment_results.get('unified_data') is not None:
                unified_df = alignment_results['unified_data']
                downloadable_results['unified_data_csv'] = unified_df.to_csv(index=False)
            
            # Prepare analysis summary
            if alignment_results.get('alignment_analysis'):
                import json
                analysis_json = json.dumps(alignment_results['alignment_analysis'], indent=2)
                downloadable_results['analysis_summary_json'] = analysis_json
            
            # Prepare processing summary
            if alignment_results.get('processing_summary'):
                import json
                summary_json = json.dumps(alignment_results['processing_summary'], indent=2)
                downloadable_results['processing_summary_json'] = summary_json
            
        except Exception as e:
            logger.error(f"Failed to prepare downloadable results: {e}")
        
        return downloadable_results
    
    def _save_experiment_results(
        self,
        experiment_id: str,
        processing_results: Dict[str, Any],
        organization_id: str,
        user_id: str
    ) -> None:
        """Save experiment results to database"""
        try:
            # Convert results to DataFrame format if unified data exists
            if processing_results.get('unified_data') is not None:
                results_df = processing_results['unified_data']
            else:
                # Create a summary DataFrame if no unified data
                summary_data = {
                    'experiment_id': [experiment_id],
                    'files_processed': [processing_results.get('files_processed', 0)],
                    'success_rate': [processing_results.get('processing_summary', {}).get('success_rate', 0)],
                    'timestamp': [datetime.now().isoformat()]
                }
                results_df = pd.DataFrame(summary_data)
            
            # Use existing experiment service to save results
            success = self.experiment_service.save_experiment_results(
                experiment_id=experiment_id,
                results_df=results_df,
                organization_id=organization_id,
                metadata={
                    'experiment_type': 'time_series_alignment',
                    'processing_summary': processing_results.get('processing_summary', {}),
                    'alignment_analysis': processing_results.get('alignment_analysis', {}),
                    'user_id': user_id
                }
            )
            
            if not success:
                logger.warning(f"Failed to save experiment results for {experiment_id}")
            
        except Exception as e:
            logger.error(f"Failed to save experiment results: {e}")
    
    def _cleanup_temp_files(self, temp_file_paths: List[str]) -> None:
        """Clean up temporary files"""
        for file_path in temp_file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    def get_time_series_experiments(
        self,
        organization_id: str,
        user_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get time-series experiments for an organization"""
        try:
            # Use existing method and filter for time-series experiments
            if user_id:
                experiments = self.experiment_service.get_user_experiments(user_id, limit)
                # Filter for time-series experiments
                return [exp for exp in experiments 
                       if exp.get('metadata', {}).get('experiment_type') == 'time_series_alignment']
            else:
                # For organization-wide, we'd need to implement this in the experiment service
                # For now, return empty list
                return []
        except Exception as e:
            logger.error(f"Failed to get time-series experiments: {e}")
            return []
    
    def get_time_series_experiment_results(
        self,
        experiment_id: str,
        organization_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get results for a specific time-series experiment"""
        try:
            # Use existing method
            return self.experiment_service.get_experiment_details(experiment_id)
        except Exception as e:
            logger.error(f"Failed to get experiment results: {e}")
            return None

# Service factory function
def get_time_series_service() -> TimeSeriesService:
    """Factory function to get time-series service instance"""
    return TimeSeriesService()

# Global instance for consistency with other services
time_series_service = TimeSeriesService()