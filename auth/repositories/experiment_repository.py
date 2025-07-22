"""
Experiment Repository
Handles all experiment-related database operations
"""

import os
import logging
import psycopg2.extras
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)

class ExperimentRepository(BaseRepository):
    """Repository for experiment-related database operations"""
    
    def __init__(self):
        """Initialize experiment repository"""
        super().__init__()

    def create_experiment(self, user_id: str, org_id: str, name: str, description: str = None) -> str:
        """
        Create a new experiment record
        
        Args:
            user_id: User ID who created the experiment
            org_id: Organization ID
            name: Experiment name
            description: Optional experiment description
            
        Returns:
            str: Experiment ID
            
        Raises:
            Exception: If experiment creation fails
        """
        try:
            # Validate required fields
            self.validate_required_fields({
                'user_id': user_id,
                'org_id': org_id,
                'name': name
            }, ['user_id', 'org_id', 'name'])
            
            # Generate experiment ID
            exp_id = self.generate_id()
            
            # Create experiment
            query = """
                INSERT INTO experiments (id, user_id, organization_id, name, description, status) 
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            params = (exp_id, user_id, org_id, name, description, 'processing')
            
            self.execute_query(query, params, fetch_all=False)
            
            logger.info(f"Created experiment {exp_id} for user {user_id}")
            return exp_id
            
        except Exception as e:
            self.handle_db_error("create_experiment", e, {'user_id': user_id, 'org_id': org_id, 'name': name})

    def get_experiment_by_id(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific experiment by ID
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Experiment information dict or None
        """
        try:
            query = """
                SELECT e.*, u.email as user_email, u.username as user_name,
                       o.name as organization_name
                FROM experiments e
                JOIN users u ON e.user_id = u.id
                JOIN organizations o ON e.organization_id = o.id
                WHERE e.id = %s
            """
            return self.execute_query(query, (exp_id,), fetch_one=True)
            
        except Exception as e:
            self.handle_db_error("get_experiment_by_id", e, {'exp_id': exp_id})
            return None

    def update_experiment_status(self, exp_id: str, status: str, result_file_url: str = None, 
                                metadata: dict = None, error_message: str = None) -> bool:
        """
        Update experiment status and results
        
        Args:
            exp_id: Experiment ID
            status: New status ('processing', 'completed', 'failed')
            result_file_url: Optional URL/path to result file
            metadata: Optional metadata dictionary
            error_message: Optional error message for failed experiments
            
        Returns:
            bool: True if update succeeded
        """
        try:
            if status == 'completed':
                query = """
                    UPDATE experiments 
                    SET status = %s, completed_at = %s, result_file_url = %s, metadata = %s
                    WHERE id = %s
                """
                params = (status, self.get_current_timestamp(), result_file_url, 
                         psycopg2.extras.Json(metadata) if metadata else None, exp_id)
            elif status == 'failed':
                query = """
                    UPDATE experiments 
                    SET status = %s, error_message = %s, completed_at = %s
                    WHERE id = %s
                """
                params = (status, error_message, self.get_current_timestamp(), exp_id)
            else:
                query = "UPDATE experiments SET status = %s WHERE id = %s"
                params = (status, exp_id)
            
            self.execute_query(query, params, fetch_all=False)
            
            logger.info(f"Updated experiment {exp_id} status to {status}")
            return True
            
        except Exception as e:
            self.handle_db_error("update_experiment_status", e, {'exp_id': exp_id, 'status': status})
            return False

    def get_user_experiments(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get user's recent experiments
        
        Args:
            user_id: User ID
            limit: Maximum number of experiments to return
            offset: Number of experiments to skip
            
        Returns:
            List of experiment information dicts
        """
        try:
            query = """
                SELECT e.*, o.name as organization_name
                FROM experiments e
                JOIN organizations o ON e.organization_id = o.id
                WHERE e.user_id = %s 
                ORDER BY e.created_at DESC 
                LIMIT %s OFFSET %s
            """
            return self.execute_query(query, (user_id, limit, offset))
            
        except Exception as e:
            self.handle_db_error("get_user_experiments", e, {'user_id': user_id})
            return []

    def get_organization_experiments(self, org_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get all experiments in an organization
        
        Args:
            org_id: Organization ID
            limit: Maximum number of experiments to return
            offset: Number of experiments to skip
            
        Returns:
            List of experiment information dicts
        """
        try:
            query = """
                SELECT e.*, u.email as user_email, u.username as user_name
                FROM experiments e
                JOIN users u ON e.user_id = u.id
                WHERE e.organization_id = %s 
                ORDER BY e.created_at DESC 
                LIMIT %s OFFSET %s
            """
            return self.execute_query(query, (org_id, limit, offset))
            
        except Exception as e:
            self.handle_db_error("get_organization_experiments", e, {'org_id': org_id})
            return []

    def get_experiment_statistics(self, org_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Get experiment statistics for organization or user
        
        Args:
            org_id: Optional organization ID to filter by
            user_id: Optional user ID to filter by
            
        Returns:
            Dict with experiment statistics
        """
        try:
            # Build query based on filters
            where_conditions = []
            params = []
            
            if org_id:
                where_conditions.append("organization_id = %s")
                params.append(org_id)
            if user_id:
                where_conditions.append("user_id = %s")
                params.append(user_id)
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            query = f"""
                SELECT 
                    COUNT(*) as total_experiments,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_experiments,
                    COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing_experiments,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_experiments,
                    COUNT(CASE WHEN created_at >= date_trunc('month', CURRENT_DATE) THEN 1 END) as experiments_this_month,
                    COUNT(CASE WHEN created_at >= date_trunc('week', CURRENT_DATE) THEN 1 END) as experiments_this_week,
                    COUNT(CASE WHEN created_at >= CURRENT_DATE THEN 1 END) as experiments_today,
                    AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) as avg_processing_time_seconds
                FROM experiments 
                {where_clause}
            """
            
            result = self.execute_query(query, tuple(params), fetch_one=True)
            
            # Convert processing time to minutes
            if result and result['avg_processing_time_seconds']:
                result['avg_processing_time_minutes'] = result['avg_processing_time_seconds'] / 60
            
            return result if result else {}
            
        except Exception as e:
            self.handle_db_error("get_experiment_statistics", e, {'org_id': org_id, 'user_id': user_id})
            return {}

    def get_recent_experiments(self, org_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent experiments across organization or globally
        
        Args:
            org_id: Optional organization ID to filter by
            limit: Maximum number of experiments to return
            
        Returns:
            List of recent experiment information dicts
        """
        try:
            if org_id:
                query = """
                    SELECT e.id, e.name, e.status, e.created_at, e.completed_at,
                           u.email as user_email, u.username as user_name
                    FROM experiments e
                    JOIN users u ON e.user_id = u.id
                    WHERE e.organization_id = %s
                    ORDER BY e.created_at DESC
                    LIMIT %s
                """
                params = (org_id, limit)
            else:
                query = """
                    SELECT e.id, e.name, e.status, e.created_at, e.completed_at,
                           u.email as user_email, u.username as user_name,
                           o.name as organization_name
                    FROM experiments e
                    JOIN users u ON e.user_id = u.id
                    JOIN organizations o ON e.organization_id = o.id
                    ORDER BY e.created_at DESC
                    LIMIT %s
                """
                params = (limit,)
                
            return self.execute_query(query, params)
            
        except Exception as e:
            self.handle_db_error("get_recent_experiments", e, {'org_id': org_id})
            return []

    def save_experiment_results(self, exp_id: str, results_data: str, file_name: str) -> str:
        """
        Save experiment results and return the file path
        
        Args:
            exp_id: Experiment ID
            results_data: Results data as string
            file_name: Name for the results file
            
        Returns:
            str: File path where results were saved
            
        Note:
            For MVP, saves to local filesystem.
            In production, this would upload to S3/cloud storage.
        """
        try:
            # Create results directory
            results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'experiment_results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save results file
            file_path = os.path.join(results_dir, f"{exp_id}_{file_name}")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(results_data)
            
            logger.info(f"Saved experiment results for {exp_id} to {file_path}")
            return file_path
            
        except Exception as e:
            self.log_error("save_experiment_results", e, {'exp_id': exp_id, 'file_name': file_name})
            raise Exception(f"Failed to save experiment results: {str(e)}")

    def delete_experiment(self, exp_id: str, user_id: str) -> bool:
        """
        Delete an experiment (only by the user who created it)
        
        Args:
            exp_id: Experiment ID
            user_id: User ID (must match experiment creator)
            
        Returns:
            bool: True if deletion succeeded
        """
        try:
            # Verify user owns the experiment
            query = "SELECT user_id FROM experiments WHERE id = %s"
            experiment = self.execute_query(query, (exp_id,), fetch_one=True)
            
            if not experiment:
                logger.warning(f"Experiment {exp_id} not found for deletion")
                return False
            
            if experiment['user_id'] != user_id:
                logger.warning(f"User {user_id} attempted to delete experiment {exp_id} owned by {experiment['user_id']}")
                return False
            
            # Delete experiment
            delete_query = "DELETE FROM experiments WHERE id = %s"
            self.execute_query(delete_query, (exp_id,), fetch_all=False)
            
            logger.info(f"Deleted experiment {exp_id} by user {user_id}")
            return True
            
        except Exception as e:
            self.handle_db_error("delete_experiment", e, {'exp_id': exp_id, 'user_id': user_id})
            return False

    def get_experiment_files(self, exp_id: str) -> List[Dict[str, Any]]:
        """
        Get list of files associated with an experiment
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            List of file information dicts
        """
        try:
            query = """
                SELECT ef.*, f.original_name, f.file_size, f.mime_type
                FROM experiment_files ef
                JOIN files f ON ef.file_id = f.id
                WHERE ef.experiment_id = %s
                ORDER BY ef.uploaded_at DESC
            """
            return self.execute_query(query, (exp_id,))
            
        except Exception as e:
            self.handle_db_error("get_experiment_files", e, {'exp_id': exp_id})
            return []

    def update_experiment_metadata(self, exp_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update experiment metadata
        
        Args:
            exp_id: Experiment ID
            metadata: Metadata dictionary to update
            
        Returns:
            bool: True if update succeeded
        """
        try:
            query = "UPDATE experiments SET metadata = %s WHERE id = %s"
            self.execute_query(query, (psycopg2.extras.Json(metadata), exp_id), fetch_all=False)
            
            logger.info(f"Updated metadata for experiment {exp_id}")
            return True
            
        except Exception as e:
            self.handle_db_error("update_experiment_metadata", e, {'exp_id': exp_id})
            return False