"""
Experiment Service
Handles experiment creation, tracking, and history management
"""

import os
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Import database functions
from auth.database import (
    create_experiment, update_experiment_status, get_user_experiments, 
    get_experiment_by_id, save_experiment_results
)

# Import storage functionality if available
try:
    from auth.storage import is_storage_configured, get_presigned_url
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False


class ExperimentService:
    """Centralized experiment management service"""
    
    def __init__(self):
        pass
    
    def create_new_experiment(self, user_id: str, organization_id: str, 
                             experiment_name: str, description: str = "") -> str:
        """Create a new experiment record"""
        try:
            experiment_id = create_experiment(
                user_id=user_id,
                org_id=organization_id,
                name=experiment_name,
                description=description
            )
            return experiment_id
        except Exception as e:
            st.error(f"Failed to create experiment: {str(e)}")
            return None
    
    def update_experiment(self, experiment_id: str, status: str, 
                         metadata: Dict[str, Any] = None, error_message: str = None) -> bool:
        """Update experiment status and metadata"""
        try:
            success = update_experiment_status(
                exp_id=experiment_id,
                status=status,
                result_file_url=None,
                metadata=metadata
            )
            return success
        except Exception as e:
            print(f"Failed to update experiment {experiment_id}: {str(e)}")
            return False
    
    def save_experiment_results(self, experiment_id: str, results_df: pd.DataFrame, 
                               organization_id: str, metadata: Dict[str, Any] = None) -> bool:
        """Save experiment results to storage and database"""
        try:
            # Prepare results for saving
            csv_data = results_df.to_csv(index=False)
            
            success = save_experiment_results(
                exp_id=experiment_id,
                results_data=csv_data,
                file_name=f"experiment_{experiment_id[:8]}_results.csv"
            )
            
            if success:
                # Update experiment status to completed
                self.update_experiment(experiment_id, 'completed', metadata)
                return True
            else:
                self.update_experiment(experiment_id, 'failed', metadata, 'Failed to save results')
                return False
                
        except Exception as e:
            error_msg = f"Error saving results: {str(e)}"
            print(error_msg)
            self.update_experiment(experiment_id, 'failed', metadata, error_msg)
            return False
    
    def get_user_experiments(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's experiment history"""
        try:
            experiments = get_user_experiments(user_id, limit=limit)
            return experiments or []
        except Exception as e:
            print(f"Error getting user experiments: {str(e)}")
            return []
    
    def get_experiment_details(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific experiment"""
        try:
            experiment = get_experiment_by_id(experiment_id)
            return experiment
        except Exception as e:
            print(f"Error getting experiment details: {str(e)}")
            return None
    
    def format_experiment_display(self, experiment: Dict[str, Any]) -> Dict[str, str]:
        """Format experiment data for UI display"""
        return {
            'id': experiment['id'],
            'name': experiment.get('name', 'Unnamed Experiment'),
            'status': experiment.get('status', 'unknown').title(),
            'created_at': experiment.get('created_at', datetime.now()).strftime('%Y-%m-%d %H:%M') if experiment.get('created_at') else 'Unknown',
            'description': experiment.get('description', ''),
            'metadata': experiment.get('metadata', {})
        }
    
    def download_experiment_results(self, experiment: Dict[str, Any]) -> Tuple[bool, str, str]:
        """
        Handle experiment results download
        Returns: (success, download_data, filename)
        """
        try:
            if not experiment.get('result_file_url'):
                return False, "", "No results available"
            
            result_url = experiment['result_file_url']
            experiment_id = experiment['id']
            
            # Check if it's a URL or local path
            if result_url.startswith('http'):
                # S3 URL - generate presigned URL if needed
                if STORAGE_AVAILABLE and is_storage_configured():
                    try:
                        # Extract S3 key from URL
                        s3_key = '/'.join(result_url.split('/')[-3:])
                        presigned_url = get_presigned_url(s3_key)
                        return True, presigned_url, f"experiment_{experiment_id[:8]}_results.csv"
                    except Exception as e:
                        print(f"Error generating presigned URL: {e}")
                        return False, "", f"Error accessing cloud storage: {str(e)}"
                else:
                    # Direct URL (not recommended for production)
                    return True, result_url, f"experiment_{experiment_id[:8]}_results.csv"
            else:
                # Local file
                if os.path.exists(result_url):
                    with open(result_url, 'r') as f:
                        csv_content = f.read()
                    return True, csv_content, f"experiment_{experiment_id[:8]}_results.csv"
                else:
                    return False, "", "File not found"
                    
        except Exception as e:
            return False, "", f"Error downloading file: {str(e)}"
    
    def get_experiment_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get user experiment statistics"""
        try:
            experiments = self.get_user_experiments(user_id)
            
            if not experiments:
                return {
                    'total_experiments': 0,
                    'completed_experiments': 0,
                    'failed_experiments': 0,
                    'processing_experiments': 0,
                    'success_rate': 0.0
                }
            
            stats = {
                'total_experiments': len(experiments),
                'completed_experiments': len([e for e in experiments if e.get('status') == 'completed']),
                'failed_experiments': len([e for e in experiments if e.get('status') == 'failed']),
                'processing_experiments': len([e for e in experiments if e.get('status') == 'processing']),
            }
            
            stats['success_rate'] = (stats['completed_experiments'] / stats['total_experiments'] * 100) if stats['total_experiments'] > 0 else 0.0
            
            return stats
            
        except Exception as e:
            print(f"Error calculating experiment statistics: {str(e)}")
            return {
                'total_experiments': 0,
                'completed_experiments': 0,
                'failed_experiments': 0,
                'processing_experiments': 0,
                'success_rate': 0.0
            }
    
    def cleanup_old_experiments(self, user_id: str, days_old: int = 30) -> int:
        """Clean up old failed experiments (admin function)"""
        # This would be implemented with appropriate database cleanup
        # For now, just return 0
        return 0
    
    def export_experiment_history(self, user_id: str) -> pd.DataFrame:
        """Export user's experiment history as DataFrame"""
        try:
            experiments = self.get_user_experiments(user_id)
            
            if not experiments:
                return pd.DataFrame()
            
            # Convert to DataFrame format
            export_data = []
            for exp in experiments:
                export_data.append({
                    'Experiment ID': exp['id'],
                    'Name': exp.get('name', 'Unnamed'),
                    'Status': exp.get('status', 'unknown'),
                    'Created At': exp.get('created_at', ''),
                    'Description': exp.get('description', ''),
                    'Processing Time': exp.get('metadata', {}).get('processing_time', ''),
                    'Files Processed': exp.get('metadata', {}).get('files_processed', ''),
                    'Total Samples': exp.get('metadata', {}).get('total_samples', ''),
                    'Unique Plates': exp.get('metadata', {}).get('unique_plates', '')
                })
            
            return pd.DataFrame(export_data)
            
        except Exception as e:
            print(f"Error exporting experiment history: {str(e)}")
            return pd.DataFrame()


# Global instance
experiment_service = ExperimentService()