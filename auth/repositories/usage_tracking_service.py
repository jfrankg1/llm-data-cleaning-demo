"""
Usage Tracking Service
Handles all usage tracking and metrics database operations
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)

class UsageTrackingService(BaseRepository):
    """Service for usage tracking database operations"""
    
    def __init__(self):
        """Initialize usage tracking service"""
        super().__init__()

    def track_usage(self, org_id: str, user_id: str, metric_type: str, value: int = 1, 
                   metadata: Dict[str, Any] = None) -> bool:
        """
        Track usage metrics for an organization and user
        
        Args:
            org_id: Organization ID
            user_id: User ID
            metric_type: Type of metric ('experiment', 'api_call', 'storage', 'file_upload')
            value: Metric value to record
            metadata: Optional additional metadata
            
        Returns:
            bool: True if tracking succeeded
        """
        try:
            query = """
                INSERT INTO usage_tracking (organization_id, user_id, metric_type, metric_value, metadata) 
                VALUES (%s, %s, %s, %s, %s)
            """
            params = (org_id, user_id, metric_type, value, metadata)
            
            self.execute_query(query, params, fetch_all=False)
            
            logger.debug(f"Tracked {metric_type}: {value} for user {user_id} in org {org_id}")
            return True
            
        except Exception as e:
            self.log_error("track_usage", e, {'org_id': org_id, 'user_id': user_id, 'metric_type': metric_type})
            return False

    def get_usage_summary(self, org_id: str, period_days: int = 30) -> Dict[str, Any]:
        """
        Get usage summary for an organization over a period
        
        Args:
            org_id: Organization ID
            period_days: Number of days to include in summary
            
        Returns:
            Dict with usage summary
        """
        try:
            cutoff_date = self.get_current_timestamp() - timedelta(days=period_days)
            
            query = """
                SELECT 
                    metric_type,
                    COUNT(*) as event_count,
                    SUM(metric_value) as total_value,
                    AVG(metric_value) as avg_value,
                    MAX(metric_value) as max_value,
                    MIN(timestamp) as first_event,
                    MAX(timestamp) as last_event
                FROM usage_tracking 
                WHERE organization_id = %s 
                AND timestamp >= %s
                GROUP BY metric_type
                ORDER BY total_value DESC
            """
            
            results = self.execute_query(query, (org_id, cutoff_date))
            
            # Organize results by metric type
            summary = {}
            for row in results:
                summary[row['metric_type']] = {
                    'event_count': row['event_count'],
                    'total_value': row['total_value'],
                    'avg_value': float(row['avg_value']) if row['avg_value'] else 0,
                    'max_value': row['max_value'],
                    'first_event': row['first_event'],
                    'last_event': row['last_event']
                }
            
            return summary
            
        except Exception as e:
            self.handle_db_error("get_usage_summary", e, {'org_id': org_id, 'period_days': period_days})
            return {}

    def get_daily_usage(self, org_id: str, metric_type: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get daily usage breakdown for a specific metric
        
        Args:
            org_id: Organization ID
            metric_type: Type of metric to analyze
            days: Number of days to include
            
        Returns:
            List of daily usage data
        """
        try:
            cutoff_date = self.get_current_timestamp() - timedelta(days=days)
            
            query = """
                SELECT 
                    DATE(timestamp) as usage_date,
                    COUNT(*) as event_count,
                    SUM(metric_value) as total_value,
                    COUNT(DISTINCT user_id) as unique_users
                FROM usage_tracking 
                WHERE organization_id = %s 
                AND metric_type = %s
                AND timestamp >= %s
                GROUP BY DATE(timestamp)
                ORDER BY usage_date DESC
            """
            
            return self.execute_query(query, (org_id, metric_type, cutoff_date))
            
        except Exception as e:
            self.handle_db_error("get_daily_usage", e, {'org_id': org_id, 'metric_type': metric_type})
            return []

    def get_user_usage(self, user_id: str, period_days: int = 30) -> Dict[str, Any]:
        """
        Get usage summary for a specific user
        
        Args:
            user_id: User ID
            period_days: Number of days to include
            
        Returns:
            Dict with user usage summary
        """
        try:
            cutoff_date = self.get_current_timestamp() - timedelta(days=period_days)
            
            query = """
                SELECT 
                    metric_type,
                    COUNT(*) as event_count,
                    SUM(metric_value) as total_value,
                    DATE(MIN(timestamp)) as first_activity,
                    DATE(MAX(timestamp)) as last_activity
                FROM usage_tracking 
                WHERE user_id = %s 
                AND timestamp >= %s
                GROUP BY metric_type
                ORDER BY total_value DESC
            """
            
            results = self.execute_query(query, (user_id, cutoff_date))
            
            # Organize results by metric type
            summary = {}
            for row in results:
                summary[row['metric_type']] = {
                    'event_count': row['event_count'],
                    'total_value': row['total_value'],
                    'first_activity': row['first_activity'],
                    'last_activity': row['last_activity']
                }
            
            return summary
            
        except Exception as e:
            self.handle_db_error("get_user_usage", e, {'user_id': user_id})
            return {}

    def get_top_users(self, org_id: str, metric_type: str, limit: int = 10, 
                     period_days: int = 30) -> List[Dict[str, Any]]:
        """
        Get top users by usage metric
        
        Args:
            org_id: Organization ID
            metric_type: Type of metric to rank by
            limit: Maximum number of users to return
            period_days: Number of days to include
            
        Returns:
            List of top users with usage data
        """
        try:
            cutoff_date = self.get_current_timestamp() - timedelta(days=period_days)
            
            query = """
                SELECT 
                    ut.user_id,
                    u.email,
                    u.username,
                    COUNT(*) as event_count,
                    SUM(ut.metric_value) as total_value,
                    MAX(ut.timestamp) as last_activity
                FROM usage_tracking ut
                JOIN users u ON ut.user_id = u.id
                WHERE ut.organization_id = %s 
                AND ut.metric_type = %s
                AND ut.timestamp >= %s
                GROUP BY ut.user_id, u.email, u.username
                ORDER BY total_value DESC
                LIMIT %s
            """
            
            return self.execute_query(query, (org_id, metric_type, cutoff_date, limit))
            
        except Exception as e:
            self.handle_db_error("get_top_users", e, {'org_id': org_id, 'metric_type': metric_type})
            return []

    def get_usage_trends(self, org_id: str, metric_type: str, days: int = 7) -> Dict[str, Any]:
        """
        Get usage trends and patterns
        
        Args:
            org_id: Organization ID
            metric_type: Type of metric to analyze
            days: Number of days to analyze
            
        Returns:
            Dict with trend analysis
        """
        try:
            cutoff_date = self.get_current_timestamp() - timedelta(days=days)
            
            # Get daily totals
            daily_query = """
                SELECT 
                    DATE(timestamp) as usage_date,
                    SUM(metric_value) as daily_total
                FROM usage_tracking 
                WHERE organization_id = %s 
                AND metric_type = %s
                AND timestamp >= %s
                GROUP BY DATE(timestamp)
                ORDER BY usage_date
            """
            
            daily_data = self.execute_query(daily_query, (org_id, metric_type, cutoff_date))
            
            # Calculate trend
            if len(daily_data) >= 2:
                first_half = daily_data[:len(daily_data)//2]
                second_half = daily_data[len(daily_data)//2:]
                
                first_avg = sum(row['daily_total'] for row in first_half) / len(first_half)
                second_avg = sum(row['daily_total'] for row in second_half) / len(second_half)
                
                trend_direction = 'increasing' if second_avg > first_avg else 'decreasing'
                trend_percentage = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
            else:
                trend_direction = 'stable'
                trend_percentage = 0
            
            # Get hourly patterns
            hourly_query = """
                SELECT 
                    EXTRACT(HOUR FROM timestamp) as hour,
                    AVG(metric_value) as avg_value
                FROM usage_tracking 
                WHERE organization_id = %s 
                AND metric_type = %s
                AND timestamp >= %s
                GROUP BY EXTRACT(HOUR FROM timestamp)
                ORDER BY hour
            """
            
            hourly_data = self.execute_query(hourly_query, (org_id, metric_type, cutoff_date))
            
            return {
                'daily_data': daily_data,
                'trend_direction': trend_direction,
                'trend_percentage': round(trend_percentage, 2),
                'hourly_patterns': hourly_data,
                'analysis_period_days': days,
                'total_events': len(daily_data)
            }
            
        except Exception as e:
            self.handle_db_error("get_usage_trends", e, {'org_id': org_id, 'metric_type': metric_type})
            return {}

    def get_current_usage(self, org_id: str) -> Dict[str, Any]:
        """
        Get current usage levels for an organization
        
        Args:
            org_id: Organization ID
            
        Returns:
            Dict with current usage metrics
        """
        try:
            today = datetime.now().date()
            this_month_start = today.replace(day=1)
            
            # Get today's usage
            today_query = """
                SELECT 
                    metric_type,
                    SUM(metric_value) as today_total
                FROM usage_tracking 
                WHERE organization_id = %s 
                AND DATE(timestamp) = %s
                GROUP BY metric_type
            """
            
            today_results = self.execute_query(today_query, (org_id, today))
            today_usage = {row['metric_type']: row['today_total'] for row in today_results}
            
            # Get this month's usage
            month_query = """
                SELECT 
                    metric_type,
                    SUM(metric_value) as month_total
                FROM usage_tracking 
                WHERE organization_id = %s 
                AND DATE(timestamp) >= %s
                GROUP BY metric_type
            """
            
            month_results = self.execute_query(month_query, (org_id, this_month_start))
            month_usage = {row['metric_type']: row['month_total'] for row in month_results}
            
            return {
                'today': today_usage,
                'this_month': month_usage,
                'as_of': self.get_current_timestamp()
            }
            
        except Exception as e:
            self.handle_db_error("get_current_usage", e, {'org_id': org_id})
            return {'today': {}, 'this_month': {}}

    def cleanup_old_usage_data(self, days_to_keep: int = 365) -> int:
        """
        Clean up old usage tracking data
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            int: Number of records deleted
        """
        try:
            cutoff_date = self.get_current_timestamp() - timedelta(days=days_to_keep)
            
            # Count records to be deleted
            count_query = "SELECT COUNT(*) as count FROM usage_tracking WHERE timestamp < %s"
            count_result = self.execute_query(count_query, (cutoff_date,), fetch_one=True)
            records_to_delete = count_result['count'] if count_result else 0
            
            # Delete old records
            delete_query = "DELETE FROM usage_tracking WHERE timestamp < %s"
            self.execute_query(delete_query, (cutoff_date,), fetch_all=False)
            
            logger.info(f"Cleaned up {records_to_delete} usage tracking records older than {days_to_keep} days")
            return records_to_delete
            
        except Exception as e:
            self.handle_db_error("cleanup_old_usage_data", e, {'days_to_keep': days_to_keep})
            return 0