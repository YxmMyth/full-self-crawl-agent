"""
Meta-Controller — 全自主外循环控制器

职责：
- 监控流水线跨页面的整体表现（成功率、质量趋势）
- 当连续失败或质量下滑时，自主决策切换策略
- 无人工介入：通过多层策略自动升级（调参→换选择器→换提取方式→降级）
- 汇总跨页面知识（哪些选择器有效、哪些 URL 模式产出高）

设计原则：
- 纯自主决策，零人工干预
- 可观测：所有决策有原因日志
- 渐进升级：从微调到大幅策略变更
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EscalationLevel(int, Enum):
    """自主升级级别（从轻到重）"""
    NONE = 0          # 无需干预
    TUNE_PARAMS = 1   # 微调参数（超时、重试次数）
    CHANGE_SELECTORS = 2  # 换选择器策略
    CHANGE_EXTRACTION = 3  # 换提取方式（CSS→XPath→LLM）
    DEGRADE_GRACEFUL = 4   # 优雅降级（降低质量要求，收集部分数据）
    ABORT_DOMAIN = 5       # 放弃当前域名/路径模式，跳到下一区域


@dataclass
class PageOutcome:
    """单页处理结果摘要"""
    url: str
    success: bool
    quality_score: float
    item_count: int
    extraction_method: str
    retry_attempts: int
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyAdjustment:
    """策略调整指令"""
    level: EscalationLevel
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class MetaController:
    """
    全自主外循环控制器

    使用滑动窗口监控最近 N 页的表现，
    当检测到系统性问题（非偶发故障）时自主升级策略。
    """

    def __init__(self, window_size: int = 5, quality_floor: float = 0.3,
                 success_rate_floor: float = 0.4):
        self.window_size = window_size
        self.quality_floor = quality_floor
        self.success_rate_floor = success_rate_floor

        # 历史记录
        self.outcomes: List[PageOutcome] = []
        self.adjustments: List[StrategyAdjustment] = []

        # 当前策略状态
        self.current_level = EscalationLevel.NONE
        self.active_overrides: Dict[str, Any] = {}

        # 跨页面知识
        self.effective_selectors: Dict[str, int] = {}   # selector → success_count
        self.failed_url_patterns: List[str] = []
        self.domain_stats: Dict[str, Dict[str, float]] = {}  # domain → {success_rate, avg_quality}

    def record_outcome(self, url: str, page_result: Dict[str, Any]) -> None:
        """记录一页的处理结果"""
        metrics = page_result.get('metrics', {})
        outcome = PageOutcome(
            url=url,
            success=page_result.get('success', False),
            quality_score=metrics.get('data_quality_score', 0.0),
            item_count=metrics.get('total_items_extracted', 0),
            extraction_method=metrics.get('extraction_method', 'unknown'),
            retry_attempts=metrics.get('retry_attempts', 1),
            error=page_result.get('error'),
        )
        self.outcomes.append(outcome)

        # 更新有效选择器知识
        if outcome.success and outcome.quality_score >= 0.6:
            selectors = page_result.get('selectors', {})
            for key, sel in selectors.items():
                if isinstance(sel, str):
                    self.effective_selectors[sel] = self.effective_selectors.get(sel, 0) + 1

        # 更新域名统计
        self._update_domain_stats(url, outcome)

    def evaluate(self) -> Optional[StrategyAdjustment]:
        """
        评估最近 window_size 页的表现，决定是否需要策略调整。
        返回 None 表示无需调整，否则返回调整指令。
        """
        window = self.outcomes[-self.window_size:]
        if len(window) < 2:
            return None

        success_rate = sum(1 for o in window if o.success) / len(window)
        avg_quality = sum(o.quality_score for o in window) / len(window)
        avg_items = sum(o.item_count for o in window) / len(window)
        consecutive_failures = self._count_consecutive_tail_failures()

        # 趋势检测：质量是否在下滑？
        quality_declining = self._is_quality_declining(window)

        # 决策矩阵（从最严重到最轻微）
        adjustment = None

        if consecutive_failures >= self.window_size:
            # 连续全部失败 → 放弃当前区域
            adjustment = StrategyAdjustment(
                level=EscalationLevel.ABORT_DOMAIN,
                action='skip_url_pattern',
                params={'skip_pattern': self._extract_url_pattern(window[-1].url)},
                reason=f"连续 {consecutive_failures} 页全部失败，放弃当前 URL 模式",
            )
        elif success_rate < self.success_rate_floor and consecutive_failures >= 3:
            # 成功率极低 → 优雅降级
            adjustment = StrategyAdjustment(
                level=EscalationLevel.DEGRADE_GRACEFUL,
                action='lower_quality_threshold',
                params={
                    'quality_floor': max(0.1, self.quality_floor - 0.15),
                    'accept_partial': True,
                },
                reason=f"成功率 {success_rate:.0%} 低于阈值，降级为部分数据收集",
            )
        elif avg_quality < self.quality_floor and avg_items < 3:
            # 质量和数量都低 → 换提取方式
            adjustment = StrategyAdjustment(
                level=EscalationLevel.CHANGE_EXTRACTION,
                action='switch_extraction_method',
                params={
                    'prefer_method': 'llm_direct',
                    'fallback_to_regex': True,
                },
                reason=f"平均质量 {avg_quality:.2f}、平均条数 {avg_items:.1f} 均低，切换提取方式",
            )
        elif quality_declining and avg_quality < 0.6:
            # 质量持续下滑 → 换选择器
            adjustment = StrategyAdjustment(
                level=EscalationLevel.CHANGE_SELECTORS,
                action='refresh_selectors',
                params={
                    'use_effective_selectors': self._get_top_selectors(5),
                    'force_llm_plan': True,
                },
                reason=f"质量连续下滑（当前 {avg_quality:.2f}），刷新选择器策略",
            )
        elif consecutive_failures >= 2:
            # 轻微连续失败 → 微调参数
            adjustment = StrategyAdjustment(
                level=EscalationLevel.TUNE_PARAMS,
                action='increase_timeouts',
                params={
                    'timeout_multiplier': 1.5,
                    'max_page_retries': 5,
                    'scroll_delay_increase': 0.5,
                },
                reason=f"连续 {consecutive_failures} 页失败，微调超时和重试参数",
            )

        if adjustment:
            self.adjustments.append(adjustment)
            self.current_level = adjustment.level
            self._apply_overrides(adjustment)
            logger.info(f"[MetaController] 升级到 L{adjustment.level.value}: "
                        f"{adjustment.action} — {adjustment.reason}")

        return adjustment

    def get_context_overrides(self) -> Dict[str, Any]:
        """获取当前策略覆盖参数，注入到 pipeline context 中"""
        return dict(self.active_overrides)

    def get_stats(self) -> Dict[str, Any]:
        """获取控制器统计信息"""
        total = len(self.outcomes)
        successes = sum(1 for o in self.outcomes if o.success)
        return {
            'total_pages': total,
            'success_rate': successes / total if total else 0,
            'avg_quality': sum(o.quality_score for o in self.outcomes) / total if total else 0,
            'avg_items': sum(o.item_count for o in self.outcomes) / total if total else 0,
            'current_escalation_level': self.current_level.value,
            'total_adjustments': len(self.adjustments),
            'active_overrides': dict(self.active_overrides),
            'effective_selectors_count': len(self.effective_selectors),
            'failed_url_patterns': list(self.failed_url_patterns),
        }

    def should_skip_url(self, url: str) -> bool:
        """判断 URL 是否匹配已知的失败模式，应跳过"""
        for pattern in self.failed_url_patterns:
            if pattern in url:
                return True
        return False

    def reset_escalation(self) -> None:
        """当策略调整后表现改善时，重置升级级别"""
        recent = self.outcomes[-3:] if len(self.outcomes) >= 3 else self.outcomes
        if all(o.success and o.quality_score >= 0.6 for o in recent):
            if self.current_level != EscalationLevel.NONE:
                logger.info("[MetaController] 表现恢复，重置升级级别")
                self.current_level = EscalationLevel.NONE
                self.active_overrides.clear()

    # ── 内部方法 ──

    def _count_consecutive_tail_failures(self) -> int:
        """从最新一页倒数连续失败数"""
        count = 0
        for outcome in reversed(self.outcomes):
            if not outcome.success:
                count += 1
            else:
                break
        return count

    def _is_quality_declining(self, window: List[PageOutcome]) -> bool:
        """检测质量是否持续下滑（至少 3 个点形成下降趋势）"""
        scores = [o.quality_score for o in window if o.success]
        if len(scores) < 3:
            return False
        # 简单线性趋势：后半段平均 < 前半段平均
        mid = len(scores) // 2
        first_half = sum(scores[:mid]) / mid
        second_half = sum(scores[mid:]) / (len(scores) - mid)
        return second_half < first_half * 0.85

    def _extract_url_pattern(self, url: str) -> str:
        """从 URL 提取模式（去掉具体 ID/页码）"""
        from urllib.parse import urlparse
        import re
        parsed = urlparse(url)
        # 将数字 ID 替换为通配符
        path_pattern = re.sub(r'/\d+', '/*', parsed.path)
        return f"{parsed.netloc}{path_pattern}"

    def _get_top_selectors(self, n: int) -> List[str]:
        """获取历史上最有效的 N 个选择器"""
        sorted_sels = sorted(self.effective_selectors.items(), key=lambda x: x[1], reverse=True)
        return [sel for sel, _ in sorted_sels[:n]]

    def _apply_overrides(self, adjustment: StrategyAdjustment) -> None:
        """将策略调整转化为 context 覆盖参数"""
        params = adjustment.params

        if adjustment.action == 'increase_timeouts':
            self.active_overrides['timeout_multiplier'] = params.get('timeout_multiplier', 1.5)
            self.active_overrides['max_page_retries'] = params.get('max_page_retries', 5)

        elif adjustment.action == 'refresh_selectors':
            self.active_overrides['force_llm_plan'] = True
            effective = params.get('use_effective_selectors', [])
            if effective:
                self.active_overrides['hint_selectors'] = effective

        elif adjustment.action == 'switch_extraction_method':
            self.active_overrides['prefer_extraction_method'] = params.get('prefer_method', 'llm_direct')
            self.active_overrides['fallback_to_regex'] = params.get('fallback_to_regex', True)

        elif adjustment.action == 'lower_quality_threshold':
            self.active_overrides['quality_floor'] = params.get('quality_floor', 0.15)
            self.active_overrides['accept_partial'] = True

        elif adjustment.action == 'skip_url_pattern':
            pattern = params.get('skip_pattern', '')
            if pattern and pattern not in self.failed_url_patterns:
                self.failed_url_patterns.append(pattern)

    def _update_domain_stats(self, url: str, outcome: PageOutcome) -> None:
        """更新域名级别统计"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        if domain not in self.domain_stats:
            self.domain_stats[domain] = {'total': 0, 'successes': 0, 'quality_sum': 0.0}
        stats = self.domain_stats[domain]
        stats['total'] += 1
        if outcome.success:
            stats['successes'] += 1
        stats['quality_sum'] += outcome.quality_score
        stats['success_rate'] = stats['successes'] / stats['total']
        stats['avg_quality'] = stats['quality_sum'] / stats['total']
