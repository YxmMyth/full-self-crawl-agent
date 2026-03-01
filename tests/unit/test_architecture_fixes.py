"""
架构缺陷修复的单元测试

涵盖以下修复：
1. CompletionGate._evaluate() 对未知门禁条件返回 False 而非抛出 ValueError
2. StateManager.add_error_sync() 替代 main.py 中未被 await 的 add_error()
3. RiskMonitor 集成到主循环（通过 check_metrics 调用）
4. ProgressiveExplorer.explore() 完整实现（不再是空桩函数）
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ===========================================================================
# 1. CompletionGate - 未知门禁条件返回 False
# ===========================================================================

class TestCompletionGateUnknownCondition:
    """CompletionGate._evaluate() 对未知条件应返回 False 而非 raise ValueError"""

    def setup_method(self):
        from src.core.completion_gate import CompletionGate
        self.gate = CompletionGate()

    def test_unknown_condition_returns_false(self):
        """未知门禁条件应安静地返回 False，不抛出异常"""
        state = {'quality_score': 0.9}
        # 不应抛出 ValueError
        result = self.gate._evaluate('totally_unknown_condition', state)
        assert result is False

    def test_unknown_condition_does_not_crash_check(self):
        """包含未知条件的 spec 不应导致 check() 崩溃"""
        spec = {
            'completion_gate': [
                'quality_score >= 0.5',
                'an_unknown_gate_condition',   # 未知条件
            ]
        }
        state = {'quality_score': 0.9, 'sample_data': [1, 2, 3]}
        # 不应抛出任何异常；未知条件算作失败
        result = self.gate.check(state, spec)
        assert result is False
        assert 'an_unknown_gate_condition' in self.gate.get_failed_gates()

    def test_known_conditions_still_work(self):
        """已知条件的评估行为不受影响"""
        spec = {'completion_gate': ['quality_score >= 0.5']}
        state = {'quality_score': 0.8}
        assert self.gate.check(state, spec) is True

    def test_empty_gate_conditions_defaults_pass(self):
        """空 gate 条件且 completion_criteria 也为空时，应通过（无条件）"""
        spec = {}
        state = {'quality_score': 0.8, 'sample_data': [1]}
        # _build_gates_from_criteria 会生成 sample_count >= 1 和 quality_score >= 0.5
        result = self.gate.check(state, spec)
        assert result is True


# ===========================================================================
# 2. StateManager - add_error_sync 同步调用
# ===========================================================================

class TestStateManagerAddErrorSync:
    """add_error_sync 应同步记录错误，无需 await"""

    def setup_method(self):
        from src.core.state_manager import StateManager
        self.sm = StateManager()
        self.sm.create_initial_state_sync('task-001', {})

    def test_add_error_sync_records_error(self):
        """add_error_sync 应立即将错误写入状态"""
        self.sm.add_error_sync('test error message')
        state = self.sm.get_state()
        assert state.get('last_error') == 'test error message'
        assert any(
            'test error message' in str(e.get('message', ''))
            for e in state.get('errors', [])
        )

    def test_add_error_sync_is_not_a_coroutine(self):
        """add_error_sync 必须是普通函数，不是协程"""
        import inspect
        from src.core.state_manager import StateManager
        assert not inspect.iscoroutinefunction(StateManager.add_error_sync)

    @pytest.mark.asyncio
    async def test_add_error_async_still_works(self):
        """异步版本 add_error 仍可正常使用"""
        await self.sm.add_error('async error')
        state = self.sm.get_state()
        assert state.get('last_error') == 'async error'


# ===========================================================================
# 3. RiskMonitor - check_metrics 基本功能
# ===========================================================================

class TestRiskMonitorCheckMetrics:
    """RiskMonitor.check_metrics() 应基于迭代指标产生正确告警"""

    def setup_method(self):
        from src.core.risk_monitor import RiskMonitor
        self.monitor = RiskMonitor()

    def test_no_alerts_for_normal_metrics(self):
        """正常指标不应触发告警"""
        alerts = self.monitor.check_metrics({
            'iteration_count': 3,
            'consecutive_errors': 0,
            'total_items': 10,
            'failed_items': 0,
        })
        assert alerts == []

    def test_high_consecutive_errors_triggers_alert(self):
        """连续错误次数超过阈值的 80% 时应触发告警"""
        # 默认 max_consecutive_errors = 5，80% = 4
        alerts = self.monitor.check_metrics({'consecutive_errors': 4})
        assert len(alerts) >= 1
        types = [a.type for a in alerts]
        assert 'consecutive_errors' in types

    def test_high_iteration_count_triggers_alert(self):
        """迭代次数接近上限时应触发告警"""
        # 默认 max_iteration_count = 100，80% = 80
        alerts = self.monitor.check_metrics({'iteration_count': 85})
        assert len(alerts) >= 1
        types = [a.type for a in alerts]
        assert 'iteration_count' in types

    def test_has_critical_risk_returns_true_on_critical_alerts(self):
        """存在 CRITICAL 告警时 has_critical_risk() 应返回 True"""
        from src.core.risk_monitor import RiskLevel
        # 触发 CRITICAL 级别：连续错误 >= max_consecutive_errors
        alerts = self.monitor.check_metrics({'consecutive_errors': 6})
        critical = [a for a in alerts if a.level in (RiskLevel.HIGH, RiskLevel.CRITICAL)]
        assert len(critical) >= 1
        assert self.monitor.has_critical_risk() is True

    def test_get_stats_reflects_alerts(self):
        """get_stats() 应反映已记录的告警数量"""
        self.monitor.check_metrics({'consecutive_errors': 6})
        stats = self.monitor.get_stats()
        assert stats['total_alerts'] >= 1


# ===========================================================================
# 4. ProgressiveExplorer - explore() 不再是空桩函数
# ===========================================================================

class TestProgressiveExplorer:
    """ProgressiveExplorer.explore() 应实际评估策略并返回有意义的结果"""

    @pytest.mark.asyncio
    async def test_explore_returns_dict(self):
        """explore() 应返回字典，不论成功与否"""
        from src.core.smart_router import ProgressiveExplorer
        explorer = ProgressiveExplorer()
        result = await explorer.explore('https://example.com', '爬取标题')
        assert isinstance(result, dict)
        assert 'success' in result

    @pytest.mark.asyncio
    async def test_explore_with_simple_html_succeeds(self):
        """简单静态页面应被 direct_crawl 策略匹配并返回成功"""
        from src.core.smart_router import ProgressiveExplorer
        explorer = ProgressiveExplorer()
        simple_html = '<html><body><h1>Title</h1><p>Content</p></body></html>'
        result = await explorer.explore(
            'https://example.com', '爬取标题',
            html=simple_html,
            # 只测试 direct_crawl，不涉及 LLM
            strategies=[('direct_crawl', 0.5)]
        )
        assert result.get('success') is True
        assert result.get('strategy') == 'direct_crawl'

    @pytest.mark.asyncio
    async def test_explore_pagination_skipped_without_pagination_html(self):
        """没有分页特征的 HTML 应跳过 pagination_crawl 策略"""
        from src.core.smart_router import ProgressiveExplorer
        explorer = ProgressiveExplorer()
        no_pagination_html = '<html><body><h1>Single page</h1></body></html>'
        result = await explorer.explore(
            'https://example.com', '爬取数据',
            html=no_pagination_html,
            strategies=[('pagination_crawl', 0.1)]  # 极低阈值，确保只测试跳过逻辑
        )
        # pagination_crawl 被跳过，没有任何策略可用，应返回失败
        assert result.get('success') is False

    @pytest.mark.asyncio
    async def test_explore_pagination_applied_with_pagination_html(self):
        """包含分页特征的 HTML 应使 pagination_crawl 策略生效"""
        from src.core.smart_router import ProgressiveExplorer
        explorer = ProgressiveExplorer()
        paginated_html = (
            '<html><body>'
            '<a href="?page=2">下一页</a>'
            '</body></html>'
        )
        result = await explorer.explore(
            'https://example.com/list', '爬取列表',
            html=paginated_html,
            strategies=[('pagination_crawl', 0.5)]
        )
        assert result.get('success') is True
        assert result.get('strategy') == 'pagination_crawl'

    def test_is_applicable_direct_crawl_always_true(self):
        """direct_crawl 对任意页面都应适用"""
        from src.core.smart_router import ProgressiveExplorer
        explorer = ProgressiveExplorer()
        assert explorer._is_applicable('direct_crawl', 'https://x.com', None) is True
        assert explorer._is_applicable('direct_crawl', 'https://x.com', '<html/>') is True

    def test_is_applicable_login_required_no_html(self):
        """没有 HTML 时，login_required 不应被标记为可用"""
        from src.core.smart_router import ProgressiveExplorer
        explorer = ProgressiveExplorer()
        assert explorer._is_applicable('login_required', 'https://x.com', None) is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
