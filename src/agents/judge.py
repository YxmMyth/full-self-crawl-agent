"""
评判智能体 - JudgeAgent
基于质量分数和迭代次数做出决策
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import json
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from .base import _safe_parse_json, DegradationTracker, AgentInterface


class JudgeAgent(AgentInterface):
    """评判智能体 - 评估当前状态并决定下一步行动"""

    def __init__(self, llm_client=None, degradation_tracker: Optional[DegradationTracker] = None):
        super().__init__("JudgeAgent", "judge")
        self.llm_client = llm_client
        self.degradation_tracker = degradation_tracker or DegradationTracker()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行评判任务
        """
        quality_score = context.get('quality_score', 0)
        iteration = context.get('iteration', 0)
        max_iterations = context.get('max_iterations', 10)
        errors = context.get('errors', [])
        spec = context.get('spec', {})
        llm_client = context.get('llm_client') or self.llm_client
        extracted_data = context.get('extracted_data', [])
        degradation_info = None

        try:
            # 1. 基于规则的决策
            rule_decision = self._rule_based_decision(
                quality_score, iteration, max_iterations, len(errors), len(extracted_data)
            )

            # 2. LLM 增强决策（推理任务，使用 DeepSeek）
            llm_decision = None
            if llm_client:
                try:
                    llm_decision = await self._llm_judge(context, llm_client)
                except Exception as e:
                    error_msg = str(e)
                    print(f"LLM 评判失败: {error_msg}")
                    # 记录降级
                    degradation_info = self.degradation_tracker.record_degradation(
                        self.name, 'llm_judge', error_msg
                    )
                    if degradation_info.get('should_warn'):
                        print(f"警告: {degradation_info['message']}")

            # 3. 综合决策
            final_decision = self._combine_decisions(rule_decision, llm_decision)

            result = {
                'success': True,
                'decision': final_decision[0],
                'reasoning': final_decision[1],
                'suggestions': self._get_suggestions(final_decision[0], context),
                'degradation_warning': degradation_info.get('message') if degradation_info else None
            }

            return result

        except Exception as e:
            error_msg = str(e)
            print(f"评判失败: {error_msg}")
            # 记录降级
            degradation_info = self.degradation_tracker.record_degradation(
                self.name, 'execute', error_msg
            )
            return {
                'success': False,
                'decision': 'reflect_and_retry',
                'reasoning': f'评判过程中出现错误: {error_msg}',
                'suggestions': ['检查错误日志', '重新尝试'],
                'degradation': degradation_info
            }

    def _rule_based_decision(self, quality_score: float, iteration: int, max_iterations: int,
                           error_count: int, data_count: int) -> tuple:
        """基于规则的决策"""
        # 1. 质量分数高，数据量充足 - 完成
        if quality_score >= 0.8 and data_count > 5:
            return 'complete', f"质量分数达标 {quality_score:.2f}，数据量充足 {data_count} 条"

        # 2. 错误过多，提前终止
        if error_count >= 5 and iteration > max_iterations * 0.5:
            return 'terminate', f"错误过多 {error_count}，迭代进度 {iteration}/{max_iterations}"

        # 3. 可提升
        if quality_score >= 0.3:
            return 'reflect_and_retry', f"质量分数 {quality_score:.2f} 可提升，继续迭代 {iteration + 1}/{max_iterations}"

        return 'terminate', f"质量分数过低 {quality_score:.2f}"

    async def _llm_judge(self, context: Dict, llm_client) -> Optional[Dict]:
        """使用 LLM 增强决策 - 推理任务，使用 DeepSeek"""
        prompt = f"""分析爬取任务的执行情况，决定下一步行动：

质量分数：{context.get('quality_score', 0)}
迭代次数：{context.get('iteration', 0)}/{context.get('max_iterations', 10)}
错误列表：{context.get('errors', [])[:5]}
目标：{context.get('spec', {}).get('goal', '未知') if context.get('spec') else '未知'}
数据量：{len(context.get('extracted_data', []))}

可选决策：
- complete: 任务完成，质量达标
- reflect_and_retry: 反思并重试
- terminate: 终止任务

请输出 JSON：
{{"decision": "complete|reflect_and_retry|terminate", "reasoning": "原因"}}"""

        try:
            # 推理任务 - 使用 reason() 方法（DeepSeek）
            if hasattr(llm_client, 'reason'):
                response = await llm_client.reason(prompt)
            else:
                response = await llm_client.chat([{"role": "user", "content": prompt}])
            return _safe_parse_json(response, "LLM 决策分析")
        except Exception as e:
            print(f"LLM 决策失败: {e}")
            return None

    def _combine_decisions(self, rule_decision: tuple, llm_decision: Optional[Dict]) -> tuple:
        """综合规则和LLM决策"""
        if llm_decision and 'decision' in llm_decision:
            # 如果LLM建议完成而规则建议重试，需要权衡
            if llm_decision['decision'] == 'complete' and rule_decision[0] != 'complete':
                # 检查LLM的质量分数和数据量
                # 这里我们暂时采用规则决策为主，LLM决策为辅
                return rule_decision
            else:
                return llm_decision.get('decision', rule_decision[0]), llm_decision.get('reasoning', rule_decision[1])
        else:
            return rule_decision

    def _get_suggestions(self, decision: str, context: Dict) -> List[str]:
        """获取改进建议"""
        if decision == 'complete':
            return ["任务已完成，可以结束"]

        suggestions = []
        errors = context.get('errors', [])
        quality_score = context.get('quality_score', 0)

        if any('selector' in str(e).lower() for e in errors):
            suggestions.append("考虑重新分析页面结构，更新选择器")
        if any('timeout' in str(e).lower() for e in errors):
            suggestions.append("增加等待时间或使用更稳定的选择器")
        if quality_score < 0.5:
            suggestions.append("质量分数过低，检查目标字段是否正确")
        if len(context.get('extracted_data', [])) < 5:
            suggestions.append("数据量不足，检查分页或数据容器选择器")

        return suggestions if suggestions else ["继续优化"]

    def get_description(self) -> str:
        return "在多个选项间做出最优决策"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'quality_score' in context or 'iteration' in context