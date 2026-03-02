"""MetaController 单元测试"""

import pytest
from src.core.meta_controller import (
    MetaController, EscalationLevel, PageOutcome, StrategyAdjustment,
)


@pytest.fixture
def mc():
    return MetaController(window_size=5, quality_floor=0.3, success_rate_floor=0.4)


def _record(mc, url, success, quality=0.7, items=5, method="css", retries=1, error=None):
    """辅助方法：记录一页结果"""
    mc.record_outcome(url, {
        'success': success,
        'metrics': {
            'data_quality_score': quality,
            'total_items_extracted': items,
            'extraction_method': method,
            'retry_attempts': retries,
        },
        'error': error,
    })


# ── 基本记录 ──

class TestRecordOutcome:
    def test_record_adds_outcome(self, mc):
        _record(mc, "https://example.com/1", True)
        assert len(mc.outcomes) == 1
        assert mc.outcomes[0].success is True

    def test_record_updates_domain_stats(self, mc):
        _record(mc, "https://example.com/1", True, quality=0.8)
        _record(mc, "https://example.com/2", False, quality=0.0)
        stats = mc.domain_stats.get("example.com")
        assert stats is not None
        assert stats['total'] == 2
        assert stats['successes'] == 1
        assert stats['success_rate'] == 0.5

    def test_effective_selectors_tracked(self, mc):
        mc.record_outcome("https://x.com/1", {
            'success': True,
            'metrics': {'data_quality_score': 0.8, 'total_items_extracted': 5,
                        'extraction_method': 'css', 'retry_attempts': 1},
            'selectors': {'main': 'div.product-card'},
        })
        assert mc.effective_selectors.get('div.product-card', 0) >= 1

    def test_low_quality_selectors_not_tracked(self, mc):
        mc.record_outcome("https://x.com/1", {
            'success': True,
            'metrics': {'data_quality_score': 0.3, 'total_items_extracted': 1,
                        'extraction_method': 'css', 'retry_attempts': 1},
            'selectors': {'main': 'div.bad'},
        })
        assert 'div.bad' not in mc.effective_selectors


# ── 评估与升级 ──

class TestEvaluate:
    def test_no_adjustment_when_few_outcomes(self, mc):
        _record(mc, "https://x.com/1", False)
        assert mc.evaluate() is None

    def test_no_adjustment_when_all_good(self, mc):
        for i in range(5):
            _record(mc, f"https://x.com/{i}", True, quality=0.9, items=10)
        assert mc.evaluate() is None

    def test_tune_params_on_2_consecutive_failures(self, mc):
        for i in range(3):
            _record(mc, f"https://x.com/{i}", True, quality=0.8)
        _record(mc, "https://x.com/4", False)
        _record(mc, "https://x.com/5", False)
        adj = mc.evaluate()
        assert adj is not None
        assert adj.level == EscalationLevel.TUNE_PARAMS
        assert adj.action == 'increase_timeouts'

    def test_change_extraction_on_low_quality_and_items(self, mc):
        for i in range(5):
            _record(mc, f"https://x.com/{i}", True, quality=0.2, items=1)
        adj = mc.evaluate()
        assert adj is not None
        assert adj.level == EscalationLevel.CHANGE_EXTRACTION

    def test_degrade_graceful_on_very_low_success(self, mc):
        _record(mc, "https://x.com/0", True, quality=0.5)
        for i in range(1, 5):
            _record(mc, f"https://x.com/{i}", False)
        adj = mc.evaluate()
        assert adj is not None
        assert adj.level.value >= EscalationLevel.DEGRADE_GRACEFUL.value

    def test_abort_domain_on_all_failures(self, mc):
        for i in range(5):
            _record(mc, f"https://x.com/{i}", False)
        adj = mc.evaluate()
        assert adj is not None
        assert adj.level == EscalationLevel.ABORT_DOMAIN
        assert adj.action == 'skip_url_pattern'

    def test_adjustments_history(self, mc):
        for i in range(5):
            _record(mc, f"https://x.com/{i}", False)
        mc.evaluate()
        assert len(mc.adjustments) >= 1


# ── 策略覆盖 ──

class TestOverrides:
    def test_tune_params_overrides(self, mc):
        for i in range(3):
            _record(mc, f"https://x.com/{i}", True)
        _record(mc, "https://x.com/4", False)
        _record(mc, "https://x.com/5", False)
        mc.evaluate()
        overrides = mc.get_context_overrides()
        assert 'timeout_multiplier' in overrides or 'max_page_retries' in overrides

    def test_abort_adds_failed_pattern(self, mc):
        for i in range(5):
            _record(mc, f"https://x.com/products/{i}", False)
        mc.evaluate()
        assert len(mc.failed_url_patterns) >= 1

    def test_should_skip_url(self, mc):
        mc.failed_url_patterns.append("x.com/products")
        assert mc.should_skip_url("https://x.com/products/123") is True
        assert mc.should_skip_url("https://y.com/items/1") is False


# ── 重置升级 ──

class TestResetEscalation:
    def test_reset_on_recovery(self, mc):
        # 先触发升级
        for i in range(5):
            _record(mc, f"https://x.com/{i}", False)
        mc.evaluate()
        assert mc.current_level != EscalationLevel.NONE

        # 恢复正常
        for i in range(3):
            _record(mc, f"https://x.com/ok{i}", True, quality=0.8)
        mc.reset_escalation()
        assert mc.current_level == EscalationLevel.NONE
        assert mc.active_overrides == {}

    def test_no_reset_when_still_bad(self, mc):
        for i in range(5):
            _record(mc, f"https://x.com/{i}", False)
        mc.evaluate()
        _record(mc, "https://x.com/ok1", True, quality=0.8)
        _record(mc, "https://x.com/fail", False, quality=0.0)
        _record(mc, "https://x.com/ok2", True, quality=0.8)
        mc.reset_escalation()
        assert mc.current_level != EscalationLevel.NONE


# ── 统计 ──

class TestStats:
    def test_get_stats_structure(self, mc):
        _record(mc, "https://x.com/1", True, quality=0.8, items=5)
        _record(mc, "https://x.com/2", False, quality=0.0, items=0)
        stats = mc.get_stats()
        assert stats['total_pages'] == 2
        assert stats['success_rate'] == 0.5
        assert 'current_escalation_level' in stats
        assert 'total_adjustments' in stats

    def test_empty_stats(self, mc):
        stats = mc.get_stats()
        assert stats['total_pages'] == 0
        assert stats['success_rate'] == 0


# ── 内部方法 ──

class TestInternals:
    def test_consecutive_tail_failures(self, mc):
        _record(mc, "https://x.com/1", True)
        _record(mc, "https://x.com/2", False)
        _record(mc, "https://x.com/3", False)
        assert mc._count_consecutive_tail_failures() == 2

    def test_quality_declining(self, mc):
        outcomes = []
        for q in [0.9, 0.85, 0.5, 0.4, 0.3]:
            _record(mc, f"https://x.com/{q}", True, quality=q)
        window = mc.outcomes[-5:]
        assert mc._is_quality_declining(window) is True

    def test_quality_not_declining(self, mc):
        for q in [0.5, 0.6, 0.7, 0.8, 0.9]:
            _record(mc, f"https://x.com/{q}", True, quality=q)
        window = mc.outcomes[-5:]
        assert mc._is_quality_declining(window) is False

    def test_extract_url_pattern(self, mc):
        pattern = mc._extract_url_pattern("https://example.com/products/123/reviews")
        assert "example.com" in pattern
        assert "/*/" in pattern or "/products/" in pattern

    def test_get_top_selectors(self, mc):
        mc.effective_selectors = {"a": 10, "b": 5, "c": 20, "d": 1}
        top = mc._get_top_selectors(2)
        assert top == ["c", "a"]
