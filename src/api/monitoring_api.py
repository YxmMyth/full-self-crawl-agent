"""
API 模块 - 供 Orchestrator 查询状态
"""
import json
from typing import Dict, Any, Optional
from datetime import datetime


class MonitoringAPI:
    """
    监控API - 提供简单的查询接口供 Orchestrator 使用
    """

    def __init__(self, state_manager, metrics_collector, progress_tracker):
        self.state_manager = state_manager
        self.metrics_collector = metrics_collector
        self.progress_tracker = progress_tracker

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取健康状态
        """
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime': getattr(self, '_start_time', datetime.now().isoformat())
        }

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取特定任务状态
        """
        # 从进度追踪器获取进度信息
        progress_info = self.progress_tracker.get_task_progress(task_id)

        # 从状态管理器获取状态信息
        state_info = self.state_manager.get_state()

        if progress_info:
            return {
                'task_id': task_id,
                'progress': progress_info,
                'state_snapshot': {
                    'current_stage': state_info.get('stage', 'unknown'),
                    'iteration': state_info.get('iteration', 0),
                    'total_extracted': state_info.get('total_extracted', 0),
                    'quality_score': state_info.get('quality_score', 0.0)
                },
                'timestamp': datetime.now().isoformat()
            }
        return None

    def get_all_task_progress(self) -> Dict[str, Any]:
        """
        获取所有任务进度
        """
        return self.progress_tracker.get_all_progress()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        获取指标摘要
        """
        return self.metrics_collector.get_metrics_summary()

    def get_current_state(self) -> Dict[str, Any]:
        """
        获取当前状态快照
        """
        state = self.state_manager.get_state()
        return {
            'state': state,
            'timestamp': datetime.now().isoformat(),
            'progress_data': self.progress_tracker.get_all_progress(),
            'metrics_summary': self.metrics_collector.get_metrics_summary()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        """
        return {
            'metrics_summary': self.get_metrics_summary(),
            'health_status': self.get_health_status(),
            'current_state': self.get_current_state(),
            'timestamp': datetime.now().isoformat()
        }