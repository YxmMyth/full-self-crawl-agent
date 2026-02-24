"""
风险监控器 - 管理层
监控系统运行风险
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAlert:
    """风险告警"""
    level: RiskLevel
    type: str
    message: str
    timestamp: datetime
    details: Dict[str, Any]


class RiskMonitor:
    """
    风险监控器

    监控项：
    - 代码安全风险
    - 资源使用风险
    - 执行时间风险
    - 错误率风险
    - 递归深度风险

    告警机制：
    - 实时监控
    - 阈值告警
    - 趋势预测
    """

    def __init__(self):
        self.alerts: List[RiskAlert] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.thresholds = self._load_default_thresholds()

    def _load_default_thresholds(self) -> Dict[str, Any]:
        """加载默认阈值"""
        return {
            # 资源阈值
            'max_memory_mb': 512,
            'max_cpu_percent': 80,
            'max_execution_time_seconds': 300,

            # 错误率阈值
            'max_error_rate': 0.3,
            'max_consecutive_errors': 5,

            # 递归深度阈值
            'max_recursion_depth': 5,
            'max_iteration_count': 100,

            # 网络阈值
            'max_request_rate': 10,  # 每秒请求数
            'max_concurrent_requests': 5,

            # 时间阈值
            'max_task_duration_hours': 24,
        }

    def check_metrics(self, metrics: Dict[str, Any]) -> List[RiskAlert]:
        """
        检查指标并生成告警

        Args:
            metrics: 指标数据

        Returns:
            告警列表
        """
        new_alerts: List[RiskAlert] = []

        # 1. 检查资源使用
        resource_alerts = self._check_resource_usage(metrics)
        new_alerts.extend(resource_alerts)

        # 2. 检查错误率
        error_alerts = self._check_error_rate(metrics)
        new_alerts.extend(error_alerts)

        # 3. 检查执行时间
        time_alerts = self._check_execution_time(metrics)
        new_alerts.extend(time_alerts)

        # 4. 检查递归深度
        recursion_alerts = self._check_recursion(metrics)
        new_alerts.extend(recursion_alerts)

        # 5. 检查网络请求
        network_alerts = self._check_network(metrics)
        new_alerts.extend(network_alerts)

        # 保存告警
        self.alerts.extend(new_alerts)

        # 保存指标历史
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })

        # 限制历史记录数量
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)

        return new_alerts

    def _check_resource_usage(self, metrics: Dict[str, Any]) -> List[RiskAlert]:
        """检查资源使用"""
        alerts = []

        # 检查内存
        memory_mb = metrics.get('memory_mb', 0)
        if memory_mb > self.thresholds['max_memory_mb'] * 0.8:
            level = RiskLevel.HIGH if memory_mb > self.thresholds['max_memory_mb'] else RiskLevel.MEDIUM
            alerts.append(RiskAlert(
                level=level,
                type='memory_usage',
                message=f'内存使用过高: {memory_mb}MB',
                timestamp=datetime.now(),
                details={'memory_mb': memory_mb, 'threshold': self.thresholds['max_memory_mb']}
            ))

        # 检查CPU
        cpu_percent = metrics.get('cpu_percent', 0)
        if cpu_percent > self.thresholds['max_cpu_percent'] * 0.8:
            level = RiskLevel.HIGH if cpu_percent > self.thresholds['max_cpu_percent'] else RiskLevel.MEDIUM
            alerts.append(RiskAlert(
                level=level,
                type='cpu_usage',
                message=f'CPU使用率过高: {cpu_percent:.1f}%',
                timestamp=datetime.now(),
                details={'cpu_percent': cpu_percent, 'threshold': self.thresholds['max_cpu_percent']}
            ))

        return alerts

    def _check_error_rate(self, metrics: Dict[str, Any]) -> List[RiskAlert]:
        """检查错误率"""
        alerts = []

        total = metrics.get('total_items', 0)
        failed = metrics.get('failed_items', 0)

        if total > 0:
            error_rate = failed / total

            if error_rate > self.thresholds['max_error_rate'] * 0.8:
                level = RiskLevel.HIGH if error_rate > self.thresholds['max_error_rate'] else RiskLevel.MEDIUM
                alerts.append(RiskAlert(
                    level=level,
                    type='error_rate',
                    message=f'错误率过高: {error_rate:.1%}',
                    timestamp=datetime.now(),
                    details={
                        'error_rate': error_rate,
                        'failed': failed,
                        'total': total,
                        'threshold': self.thresholds['max_error_rate']
                    }
                ))

        # 检查连续错误
        consecutive_errors = metrics.get('consecutive_errors', 0)
        if consecutive_errors >= self.thresholds['max_consecutive_errors'] * 0.8:
            level = RiskLevel.CRITICAL if consecutive_errors >= self.thresholds['max_consecutive_errors'] else RiskLevel.HIGH
            alerts.append(RiskAlert(
                level=level,
                type='consecutive_errors',
                message=f'连续错误次数过多: {consecutive_errors}',
                timestamp=datetime.now(),
                details={
                    'consecutive_errors': consecutive_errors,
                    'threshold': self.thresholds['max_consecutive_errors']
                }
            ))

        return alerts

    def _check_execution_time(self, metrics: Dict[str, Any]) -> List[RiskAlert]:
        """检查执行时间"""
        alerts = []

        execution_time = metrics.get('execution_time_seconds', 0)
        if execution_time > self.thresholds['max_execution_time_seconds'] * 0.8:
            level = RiskLevel.HIGH if execution_time > self.thresholds['max_execution_time_seconds'] else RiskLevel.MEDIUM
            alerts.append(RiskAlert(
                level=level,
                type='execution_time',
                message=f'执行时间过长: {execution_time:.1f}秒',
                timestamp=datetime.now(),
                details={
                    'execution_time_seconds': execution_time,
                    'threshold': self.thresholds['max_execution_time_seconds']
                }
            ))

        return alerts

    def _check_recursion(self, metrics: Dict[str, Any]) -> List[RiskAlert]:
        """检查递归深度和迭代次数"""
        alerts = []

        # 检查递归深度
        depth = metrics.get('recursion_depth', 0)
        if depth > self.thresholds['max_recursion_depth'] * 0.8:
            level = RiskLevel.CRITICAL if depth >= self.thresholds['max_recursion_depth'] else RiskLevel.HIGH
            alerts.append(RiskAlert(
                level=level,
                type='recursion_depth',
                message=f'递归深度过大: {depth}',
                timestamp=datetime.now(),
                details={
                    'depth': depth,
                    'threshold': self.thresholds['max_recursion_depth']
                }
            ))

        # 检查迭代次数
        iterations = metrics.get('iteration_count', 0)
        if iterations > self.thresholds['max_iteration_count'] * 0.8:
            level = RiskLevel.HIGH if iterations >= self.thresholds['max_iteration_count'] else RiskLevel.MEDIUM
            alerts.append(RiskAlert(
                level=level,
                type='iteration_count',
                message=f'迭代次数过多: {iterations}',
                timestamp=datetime.now(),
                details={
                    'iterations': iterations,
                    'threshold': self.thresholds['max_iteration_count']
                }
            ))

        return alerts

    def _check_network(self, metrics: Dict[str, Any]) -> List[RiskAlert]:
        """检查网络请求"""
        alerts = []

        # 检查请求速率
        request_rate = metrics.get('request_rate', 0)
        if request_rate > self.thresholds['max_request_rate']:
            alerts.append(RiskAlert(
                level=RiskLevel.MEDIUM,
                type='request_rate',
                message=f'请求速率过高: {request_rate}/秒',
                timestamp=datetime.now(),
                details={
                    'request_rate': request_rate,
                    'threshold': self.thresholds['max_request_rate']
                }
            ))

        # 检查并发请求数
        concurrent_requests = metrics.get('concurrent_requests', 0)
        if concurrent_requests > self.thresholds['max_concurrent_requests']:
            alerts.append(RiskAlert(
                level=RiskLevel.MEDIUM,
                type='concurrent_requests',
                message=f'并发请求数过多: {concurrent_requests}',
                timestamp=datetime.now(),
                details={
                    'concurrent_requests': concurrent_requests,
                    'threshold': self.thresholds['max_concurrent_requests']
                }
            ))

        return alerts

    def get_active_alerts(self) -> List[RiskAlert]:
        """获取活跃告警"""
        return [alert for alert in self.alerts if self._is_alert_active(alert)]

    def _is_alert_active(self, alert: RiskAlert) -> bool:
        """判断告警是否活跃"""
        # 5分钟内的告警视为活跃
        return (datetime.now() - alert.timestamp) < timedelta(minutes=5)

    def get_high_risk_alerts(self) -> List[RiskAlert]:
        """获取高风险告警"""
        return [
            alert for alert in self.alerts
            if alert.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            and self._is_alert_active(alert)
        ]

    def clear_alerts(self) -> None:
        """清除所有告警"""
        self.alerts.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        active_alerts = self.get_active_alerts()
        high_risk = [a for a in active_alerts if a.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]

        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'high_risk_alerts': len(high_risk),
            'alerts_by_level': {
                'low': len([a for a in active_alerts if a.level == RiskLevel.LOW]),
                'medium': len([a for a in active_alerts if a.level == RiskLevel.MEDIUM]),
                'high': len([a for a in active_alerts if a.level == RiskLevel.HIGH]),
                'critical': len([a for a in active_alerts if a.level == RiskLevel.CRITICAL]),
            }
        }

    def has_critical_risk(self) -> bool:
        """是否有严重风险"""
        return len(self.get_high_risk_alerts()) > 0
