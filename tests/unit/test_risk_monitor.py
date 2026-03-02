"""RiskMonitor 单元测试"""

import pytest
from datetime import datetime, timedelta
from src.core.risk_monitor import RiskMonitor, RiskLevel, RiskAlert


@pytest.fixture
def monitor():
    return RiskMonitor()


# ── 默认阈值 ──

class TestDefaults:
    def test_default_thresholds_loaded(self, monitor):
        assert monitor.thresholds['max_memory_mb'] == 512
        assert monitor.thresholds['max_cpu_percent'] == 80
        assert monitor.thresholds['max_error_rate'] == 0.3
        assert monitor.thresholds['max_consecutive_errors'] == 5
        assert monitor.thresholds['max_execution_time_seconds'] == 300

    def test_initial_state_empty(self, monitor):
        assert monitor.alerts == []
        assert monitor.metrics_history == []


# ── 资源检查 ──

class TestResourceUsage:
    def test_memory_high_warning(self, monitor):
        alerts = monitor.check_metrics({'memory_mb': 420})  # > 512*0.8=409.6
        mem_alerts = [a for a in alerts if a.type == 'memory_usage']
        assert len(mem_alerts) == 1
        assert mem_alerts[0].level == RiskLevel.MEDIUM

    def test_memory_critical(self, monitor):
        alerts = monitor.check_metrics({'memory_mb': 600})  # > 512
        mem_alerts = [a for a in alerts if a.type == 'memory_usage']
        assert len(mem_alerts) == 1
        assert mem_alerts[0].level == RiskLevel.HIGH

    def test_memory_normal_no_alert(self, monitor):
        alerts = monitor.check_metrics({'memory_mb': 200})
        mem_alerts = [a for a in alerts if a.type == 'memory_usage']
        assert len(mem_alerts) == 0

    def test_cpu_high_warning(self, monitor):
        alerts = monitor.check_metrics({'cpu_percent': 70})  # > 80*0.8=64
        cpu_alerts = [a for a in alerts if a.type == 'cpu_usage']
        assert len(cpu_alerts) == 1
        assert cpu_alerts[0].level == RiskLevel.MEDIUM

    def test_cpu_critical(self, monitor):
        alerts = monitor.check_metrics({'cpu_percent': 95})
        cpu_alerts = [a for a in alerts if a.type == 'cpu_usage']
        assert len(cpu_alerts) == 1
        assert cpu_alerts[0].level == RiskLevel.HIGH


# ── 错误率检查 ──

class TestErrorRate:
    def test_error_rate_warning(self, monitor):
        alerts = monitor.check_metrics({'total_items': 100, 'failed_items': 26})  # 0.26 > 0.3*0.8=0.24
        err_alerts = [a for a in alerts if a.type == 'error_rate']
        assert len(err_alerts) == 1
        assert err_alerts[0].level == RiskLevel.MEDIUM

    def test_error_rate_high(self, monitor):
        alerts = monitor.check_metrics({'total_items': 100, 'failed_items': 40})  # 0.4 > 0.3
        err_alerts = [a for a in alerts if a.type == 'error_rate']
        assert len(err_alerts) == 1
        assert err_alerts[0].level == RiskLevel.HIGH

    def test_zero_total_no_alert(self, monitor):
        alerts = monitor.check_metrics({'total_items': 0, 'failed_items': 0})
        err_alerts = [a for a in alerts if a.type == 'error_rate']
        assert len(err_alerts) == 0

    def test_consecutive_errors_critical(self, monitor):
        alerts = monitor.check_metrics({'consecutive_errors': 5})
        cons_alerts = [a for a in alerts if a.type == 'consecutive_errors']
        assert len(cons_alerts) == 1
        assert cons_alerts[0].level == RiskLevel.CRITICAL

    def test_consecutive_errors_high(self, monitor):
        alerts = monitor.check_metrics({'consecutive_errors': 4})  # >= 5*0.8=4
        cons_alerts = [a for a in alerts if a.type == 'consecutive_errors']
        assert len(cons_alerts) == 1
        assert cons_alerts[0].level == RiskLevel.HIGH


# ── 执行时间检查 ──

class TestExecutionTime:
    def test_execution_time_warning(self, monitor):
        alerts = monitor.check_metrics({'execution_time_seconds': 250})  # > 300*0.8=240
        time_alerts = [a for a in alerts if a.type == 'execution_time']
        assert len(time_alerts) == 1
        assert time_alerts[0].level == RiskLevel.MEDIUM

    def test_execution_time_critical(self, monitor):
        alerts = monitor.check_metrics({'execution_time_seconds': 350})
        time_alerts = [a for a in alerts if a.type == 'execution_time']
        assert len(time_alerts) == 1
        assert time_alerts[0].level == RiskLevel.HIGH


# ── 递归检查 ──

class TestRecursion:
    def test_recursion_depth_critical(self, monitor):
        alerts = monitor.check_metrics({'recursion_depth': 5})
        dep_alerts = [a for a in alerts if a.type == 'recursion_depth']
        assert len(dep_alerts) == 1
        assert dep_alerts[0].level == RiskLevel.CRITICAL

    def test_iteration_count_high(self, monitor):
        alerts = monitor.check_metrics({'iteration_count': 100})
        iter_alerts = [a for a in alerts if a.type == 'iteration_count']
        assert len(iter_alerts) == 1
        assert iter_alerts[0].level == RiskLevel.HIGH


# ── 网络检查 ──

class TestNetwork:
    def test_request_rate_alert(self, monitor):
        alerts = monitor.check_metrics({'request_rate': 15})
        rate_alerts = [a for a in alerts if a.type == 'request_rate']
        assert len(rate_alerts) == 1

    def test_concurrent_requests_alert(self, monitor):
        alerts = monitor.check_metrics({'concurrent_requests': 10})
        conc_alerts = [a for a in alerts if a.type == 'concurrent_requests']
        assert len(conc_alerts) == 1


# ── 告警生命周期 ──

class TestAlertLifecycle:
    def test_alerts_accumulated(self, monitor):
        monitor.check_metrics({'memory_mb': 600})
        monitor.check_metrics({'cpu_percent': 95})
        assert len(monitor.alerts) >= 2

    def test_active_alerts_within_5min(self, monitor):
        monitor.check_metrics({'memory_mb': 600})
        active = monitor.get_active_alerts()
        assert len(active) >= 1

    def test_old_alerts_expire(self, monitor):
        old_alert = RiskAlert(
            level=RiskLevel.HIGH, type='memory_usage',
            message='old', timestamp=datetime.now() - timedelta(minutes=10),
            details={},
        )
        monitor.alerts.append(old_alert)
        active = monitor.get_active_alerts()
        assert old_alert not in active

    def test_clear_alerts(self, monitor):
        monitor.check_metrics({'memory_mb': 600})
        monitor.clear_alerts()
        assert monitor.alerts == []

    def test_high_risk_alerts_filter(self, monitor):
        monitor.check_metrics({'memory_mb': 600, 'consecutive_errors': 5})
        high_risk = monitor.get_high_risk_alerts()
        assert all(a.level in [RiskLevel.HIGH, RiskLevel.CRITICAL] for a in high_risk)


# ── 统计与状态 ──

class TestStats:
    def test_has_critical_risk(self, monitor):
        assert monitor.has_critical_risk() is False
        monitor.check_metrics({'memory_mb': 600})
        assert monitor.has_critical_risk() is True

    def test_get_stats_structure(self, monitor):
        monitor.check_metrics({'memory_mb': 200})
        stats = monitor.get_stats()
        assert 'total_alerts' in stats
        assert 'active_alerts' in stats
        assert 'high_risk_alerts' in stats
        assert 'alerts_by_level' in stats

    def test_metrics_history_stored(self, monitor):
        monitor.check_metrics({'memory_mb': 100})
        monitor.check_metrics({'memory_mb': 200})
        assert len(monitor.metrics_history) == 2

    def test_metrics_history_limit(self, monitor):
        for i in range(1010):
            monitor.check_metrics({'memory_mb': i})
        assert len(monitor.metrics_history) <= 1001
