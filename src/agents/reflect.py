"""
反思智能体 - ReflectAgent
反思和优化策略
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import json
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from .base import _safe_parse_json, DegradationTracker, AgentInterface


class ReflectAgent(AgentInterface):
    """反思智能体 - 反思和优化策略"""

    def __init__(self, llm_client=None, degradation_tracker: Optional[DegradationTracker] = None):
        super().__init__("ReflectAgent", "reflect")
        self.llm_client = llm_client
        self.degradation_tracker = degradation_tracker or DegradationTracker()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        execution_history = context.get('execution_history', [])
        errors = context.get('errors', [])
        quality_score = context.get('quality_score', 0)
        spec = context.get('spec')
        llm_client = context.get('llm_client') or self.llm_client
        degradation_info = None

        # 获取提取指标（包含失败的选择器）
        extraction_metrics = context.get('extraction_metrics', {})
        failed_selectors = extraction_metrics.get('failed_selectors', {})
        required_missing = extraction_metrics.get('required_missing', {})

        # 获取 HTML 样本
        html_sample = context.get('html_snapshot', '') or context.get('html', '')[:6000]

        # 1. 分析错误模式（增强版：包含选择器失败详情）
        error_analysis = self._analyze_errors_enhanced(
            errors, failed_selectors, required_missing
        )

        # 2. 分析执行历史
        history_analysis = self._analyze_history(execution_history)

        # 3. 生成改进建议
        if llm_client:
            try:
                improvements = await self._llm_reflect(
                    error_analysis, history_analysis, spec, llm_client, html_sample, failed_selectors
                )
            except Exception as e:
                error_msg = str(e)
                print(f"LLM 反思失败: {error_msg}")
                improvements = self._fallback_improvements(error_analysis)
                # 记录降级
                degradation_info = self.degradation_tracker.record_degradation(
                    self.name, 'llm_reflect', error_msg
                )
                if degradation_info.get('should_warn'):
                    print(f"警告: {degradation_info['message']}")
        else:
            improvements = self._fallback_improvements(error_analysis)

        # 4. Docker 模式：执行验证脚本测试新选择器
        verified = False
        from src.utils.runtime import is_docker
        if is_docker() and improvements.get('selectors') and html_sample:
            verified = await self._verify_selectors_with_script(
                improvements['selectors'], html_sample
            )
            if verified:
                logger.info("ReflectAgent: 验证脚本确认新选择器有效 ✓")
            else:
                logger.info("ReflectAgent: 验证脚本未能确认新选择器，建议 script 策略")
                improvements.setdefault('strategy_type', 'script')

        result = {
            'success': True,
            'analysis': {
                'error_patterns': error_analysis,
                'history_summary': history_analysis
            },
            'improvements': improvements,
            'suggested_action': improvements.get('action', 'retry'),
            'new_selectors': improvements.get('selectors', {}),
            'new_strategy': improvements.get('strategy', None),
            'verified': verified,
        }

        # 添加降级信息
        if degradation_info:
            result['degradation'] = degradation_info

        return result

    def _analyze_errors(self, errors: List[str]) -> Dict[str, Any]:
        """分析错误模式"""
        patterns = {}
        for error in errors:
            # 分类错误
            error_str = str(error).lower()
            if 'selector' in error_str or 'element' in error_str:
                key = 'selector_error'
            elif 'timeout' in error_str:
                key = 'timeout_error'
            elif 'network' in error_str or 'connection' in error_str:
                key = 'network_error'
            elif 'anti' in error_str or 'captcha' in error_str or 'block' in error_str:
                key = 'anti_bot_error'
            else:
                key = 'unknown_error'

            patterns[key] = patterns.get(key, 0) + 1

        return {
            'patterns': patterns,
            'most_common': max(patterns.items(), key=lambda x: x[1])[0] if patterns else 'none',
            'total_errors': len(errors)
        }

    def _analyze_history(self, history: List) -> Dict[str, Any]:
        """分析执行历史"""
        if not history:
            return {'summary': '无历史记录'}

        # 处理 StateSnapshot 对象或普通字典
        def get_item(item, key):
            if hasattr(item, 'state'):
                # StateSnapshot 对象
                return item.state.get(key)
            else:
                # 普通字典
                return item.get(key)

        return {
            'total_attempts': len(history),
            'stages': [get_item(h, 'stage') for h in history if get_item(h, 'stage')],
            'last_stage': get_item(history[-1], 'stage') if history else None,
            'quality_trend': [get_item(h, 'quality_score') for h in history if get_item(h, 'quality_score')]
        }

    def _analyze_errors_enhanced(self, errors: List[str], failed_selectors: Dict,
                                  required_missing: Dict) -> Dict[str, Any]:
        """
        增强的错误分析：包含选择器失败详情

        这是改进反思深度的关键：让 LLM 知道哪些选择器失败了
        """
        # 基础错误模式分析
        patterns = {}
        for error in errors:
            error_str = str(error).lower()
            if 'selector' in error_str or 'element' in error_str:
                key = 'selector_error'
            elif 'timeout' in error_str:
                key = 'timeout_error'
            elif 'network' in error_str or 'connection' in error_str:
                key = 'network_error'
            elif 'anti' in error_str or 'captcha' in error_str or 'block' in error_str:
                key = 'anti_bot_error'
            else:
                key = 'unknown_error'
            patterns[key] = patterns.get(key, 0) + 1

        # 如果有选择器失败，优先标记为选择器错误
        if failed_selectors:
            patterns['selector_error'] = patterns.get('selector_error', 0) + sum(failed_selectors.values())

        # 构建详细的选择器失败报告
        selector_failure_report = []
        for selector, count in failed_selectors.items():
            selector_failure_report.append({
                'selector': selector,
                'failure_count': count,
                'reason': '元素未找到或选择器无效'
            })

        # 构建必填字段缺失报告
        required_missing_report = []
        for field, count in required_missing.items():
            required_missing_report.append({
                'field': field,
                'missing_count': count
            })

        return {
            'patterns': patterns,
            'most_common': max(patterns.items(), key=lambda x: x[1])[0] if patterns else 'none',
            'total_errors': len(errors),
            'failed_selectors': selector_failure_report,  # 新增：失败选择器详情
            'required_fields_missing': required_missing_report  # 新增：必填字段缺失
        }

    async def _llm_reflect(self, error_analysis, history_analysis, spec, llm_client,
                           html_sample: str = '', failed_selectors: Dict = None) -> Dict:
        """
        使用 LLM 进行反思 - 推理任务，使用 DeepSeek

        改进：现在包含失败的选择器详情和 HTML 样本
        """
        goal = spec.get('goal', '未知') if spec else '未知'
        failed_selectors = failed_selectors or {}

        # 格式化失败选择器信息
        failed_selectors_text = ""
        if failed_selectors:
            failed_selectors_text = "\n失败的选择器详情：\n"
            for selector, count in failed_selectors.items():
                failed_selectors_text += f"  - '{selector}': 匹配失败 {count} 次\n"

        # 提取 HTML 上下文
        html_context = self._extract_html_context_for_reflection(html_sample)

        prompt = f"""分析爬取任务失败的原因并生成具体改进建议：

目标：{goal}

错误模式统计：{json.dumps(error_analysis.get('patterns', {}), ensure_ascii=False)}
{failed_selectors_text}

质量趋势：{json.dumps(history_analysis.get('quality_trend', []), ensure_ascii=False)}

HTML 样本（关键元素）：
```
{html_context}
```

请基于以上信息，特别是失败的选择器和 HTML 结构，生成具体的改进建议。

输出 JSON 格式：
{{
    "action": "retry|change_strategy|abort",
    "reasoning": "基于 HTML 分析的具体原因",
    "selectors": {{"field_name": "新的 CSS 选择器（基于 HTML 分析生成）"}},
    "container_selector": "新的容器选择器",
    "strategy": "具体策略描述",
    "precautions": ["注意事项"],
    "alternative_approaches": ["备选方案"]
}}"""

        try:
            # 推理任务 - 使用 reason() 方法（DeepSeek）
            if hasattr(llm_client, 'reason'):
                response = await llm_client.reason(prompt)
            else:
                response = await llm_client.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=2048
                )
            return _safe_parse_json(response, "LLM 反思分析")
        except Exception as e:
            print(f"LLM 反思失败: {e}")
            return self._fallback_improvements(error_analysis)

    async def _verify_selectors_with_script(self, selectors: Dict[str, str],
                                             html: str) -> bool:
        """
        在 Docker 容器内执行验证脚本，测试建议的选择器是否真的能匹配到数据。

        Returns:
            True if at least one selector matches data
        """
        if not selectors or not html:
            return False

        import json as _json
        selectors_json = _json.dumps(selectors, ensure_ascii=False)

        verify_code = f'''import sys, json
from bs4 import BeautifulSoup

html = sys.stdin.read()
soup = BeautifulSoup(html, 'html.parser')
selectors = {selectors_json}

results = {{}}
for name, sel in selectors.items():
    try:
        matches = soup.select(sel)
        texts = [m.get_text(strip=True)[:100] for m in matches[:5]]
        results[name] = {{"count": len(matches), "samples": texts}}
    except Exception as e:
        results[name] = {{"count": 0, "error": str(e)}}

print(json.dumps(results, ensure_ascii=False))
'''

        try:
            from src.executors.executor import Sandbox
            sandbox = Sandbox(strict_mode=False)
            result = await sandbox.execute(verify_code, stdin_data=html, timeout=15)

            if result['success'] and result['stdout']:
                verification = _json.loads(result['stdout'])
                total_matches = sum(v.get('count', 0) for v in verification.values())
                logger.debug(f"选择器验证结果: {verification}")
                return total_matches > 0
        except Exception as e:
            logger.debug(f"验证脚本执行失败: {e}")

        return False

    def _extract_html_context_for_reflection(self, html: str) -> str:
        """
        从 HTML 中提取对反思有用的上下文

        专注于可能包含目标数据的元素结构
        """
        if not html:
            return "(无 HTML 内容)"

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        context_parts = []

        # 1. 提取常见数据容器结构
        container_patterns = [
            'article', '.article', '.post', '.item', '.card',
            '.news-item', '.product', 'tr', 'li'
        ]
        for pattern in container_patterns:
            try:
                elements = soup.select(pattern)[:2]
                for elem in elements:
                    prettified = elem.prettify()[:400]
                    context_parts.append(f"数据容器 ({pattern}):\n{prettified}")
            except:
                pass

        # 2. 提取标题元素
        for tag in ['h1', 'h2', 'h3']:
            elements = soup.find_all(tag)[:2]
            for elem in elements:
                context_parts.append(f"标题元素 ({tag}):\n{elem.prettify()[:200]}")

        # 3. 提取链接模式
        links = soup.select('a[href]')[:5]
        for link in links:
            href = link.get('href', '')
            text = link.get_text(strip=True)[:50]
            if text:
                context_parts.append(f"链接: href='{href[:50]}' text='{text}'")

        result = '\n\n'.join(context_parts[:12])
        return result[:4000] if result else html[:1500]

    def _fallback_improvements(self, error_analysis: Dict) -> Dict:
        """降级改进建议"""
        most_common = error_analysis.get('most_common', 'unknown')

        suggestions = {
            'selector_error': {
                'action': 'retry',
                'reasoning': '选择器可能已更新，建议重新分析页面',
                'selectors': {},
                'strategy': 'reanalyze'
            },
            'timeout_error': {
                'action': 'retry',
                'reasoning': '增加等待时间',
                'selectors': {},
                'strategy': 'increase_timeout'
            },
            'network_error': {
                'action': 'retry',
                'reasoning': '网络问题，稍后重试',
                'selectors': {},
                'strategy': 'delay_retry'
            },
            'anti_bot_error': {
                'action': 'change_strategy',
                'reasoning': '检测到反爬机制，需要更换策略',
                'selectors': {},
                'strategy': 'use_stealth'
            },
            'unknown_error': {
                'action': 'abort',
                'reasoning': '未知错误，建议人工介入',
                'selectors': {},
                'strategy': None
            }
        }

        return suggestions.get(most_common, suggestions['unknown_error'])

    def get_description(self) -> str:
        return "分析失败原因，生成优化建议"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        # 可以在没有错误时也工作（用于分析改进）
        return True