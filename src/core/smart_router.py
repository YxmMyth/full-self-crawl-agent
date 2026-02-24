"""
智能路由 - 管理层
三层决策模式：程序→规则→LLM
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import re


class RouteDecision(str, Enum):
    """路由决策"""
    PROGRAM = "program"  # 程序判断（确定性）
    RULE = "rule"  # 规则判断（启发式）
    LLM = "llm"  # LLM 判断（模糊场景）
    DELEGATE = "delegate"  # 委托给其他组件


@dataclass
class RouteResult:
    """路由结果"""
    decision: RouteDecision  # 决策类型
    target: str  # 目标组件
    confidence: float  # 置信度 (0-1)
    reasoning: str  # 推理过程
    timestamp: datetime


class SmartRouter:
    """
    智能路由核心

    三层决策模式：
    1. 程序层：确定性判断（高效、准确）
    2. 规则层：启发式规则（平衡效率与灵活性）
    3. LLM层：语义理解（处理模糊场景）

    路由目标：
    - 7种智能体：Sense/Plan/Act/Verify/Judge/Explore/Reflect
    - 验证层组件
    - 执行层组件
    - 工具层组件
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self._load_program_rules()
        self._load_heuristic_rules()

    def route(self, context: Dict[str, Any], task_type: str) -> RouteResult:
        """
        执行路由决策

        路由流程：
        1. 程序判断 → 2. 规则判断 → 3. LLM 判断

        Args:
            context: 路由上下文
            task_type: 任务类型

        Returns:
            RouteResult: 路由结果
        """
        # 第1层：程序判断
        program_result = self._program_route(context, task_type)
        if program_result:
            return program_result

        # 第2层：规则判断
        rule_result = self._rule_route(context, task_type)
        if rule_result:
            return rule_result

        # 第3层：LLM 判断
        llm_result = self._llm_route(context, task_type)
        return llm_result

    def _program_route(self, context: Dict[str, Any], task_type: str) -> Optional[RouteResult]:
        """第1层：程序判断 - 确定性场景"""
        # 场景1：页面结构清晰，选择器明确
        if self._is_clear_structure(context):
            return RouteResult(
                decision=RouteDecision.PROGRAM,
                target="ActAgent",
                confidence=0.95,
                reasoning="页面结构清晰，可直接执行提取",
                timestamp=datetime.now()
            )

        # 场景2：需要感知页面结构
        if self._need_sense_page(context):
            return RouteResult(
                decision=RouteDecision.PROGRAM,
                target="SenseAgent",
                confidence=0.95,
                reasoning="需要先感知页面结构",
                timestamp=datetime.now()
            )

        # 场景3：需要探索页面
        if self._need_explore(context):
            return RouteResult(
                decision=RouteDecision.PROGRAM,
                target="ExploreAgent",
                confidence=0.90,
                reasoning="需要探索页面链接和结构",
                timestamp=datetime.now()
            )

        # 场景4：数据已提取，需要验证
        if self._need_verify(context):
            return RouteResult(
                decision=RouteDecision.PROGRAM,
                target="VerifyAgent",
                confidence=0.95,
                reasoning="数据已提取，需要验证质量",
                timestamp=datetime.now()
            )

        return None

    def _rule_route(self, context: Dict[str, Any], task_type: str) -> Optional[RouteResult]:
        """第2层：规则判断 - 启发式场景"""
        # 规则1：错误次数过多，需要反思
        if context.get('error_count', 0) > 3:
            return RouteResult(
                decision=RouteDecision.RULE,
                target="ReflectAgent",
                confidence=0.85,
                reasoning="错误次数过多，需要反思策略",
                timestamp=datetime.now()
            )

        # 规则2：提取数据量不足，需要调整策略
        extracted_count = context.get('extracted_count', 0)
        target_count = context.get('target_count', 10)
        if extracted_count < target_count * 0.5:
            return RouteResult(
                decision=RouteDecision.RULE,
                target="PlanAgent",
                confidence=0.80,
                reasoning="数据量不足，需要重新规划",
                timestamp=datetime.now()
            )

        # 规则3：页面结构复杂，需要智能规划
        if self._is_complex_structure(context):
            return RouteResult(
                decision=RouteDecision.RULE,
                target="PlanAgent",
                confidence=0.75,
                reasoning="页面结构复杂，需要智能规划",
                timestamp=datetime.now()
            )

        # 规则4：需要做出决策判断
        if self._need_judgment(context):
            return RouteResult(
                decision=RouteDecision.RULE,
                target="JudgeAgent",
                confidence=0.80,
                reasoning="需要做出关键决策",
                timestamp=datetime.now()
            )

        return None

    def _llm_route(self, context: Dict[str, Any], task_type: str) -> RouteResult:
        """第3层：LLM 判断 - 模糊场景"""
        if not self.llm_client:
            # 如果没有 LLM 客户端，降级到默认策略
            return RouteResult(
                decision=RouteDecision.LLM,
                target="SenseAgent",
                confidence=0.60,
                reasoning="降级到默认策略：先感知页面",
                timestamp=datetime.now()
            )

        # 构建 LLM 提示
        prompt = self._build_llm_prompt(context, task_type)

        # 调用 LLM
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            )
            return self._parse_llm_response(response)
        except Exception as e:
            # LLM 调用失败，降级处理
            return RouteResult(
                decision=RouteDecision.LLM,
                target="SenseAgent",
                confidence=0.50,
                reasoning=f"LLM 调用失败: {str(e)}，降级到默认策略",
                timestamp=datetime.now()
            )

    def _build_llm_prompt(self, context: Dict[str, Any], task_type: str) -> str:
        """构建 LLM 提示"""
        return f"""你是一个智能路由决策器。根据当前状态选择最合适的处理组件。

当前上下文:
- 任务类型: {task_type}
- 当前状态: {context.get('current_state', 'unknown')}
- 已提取数据: {context.get('extracted_count', 0)} 条
- 错误次数: {context.get('error_count', 0)}
- 页面类型: {context.get('page_type', 'unknown')}

可选组件:
1. SenseAgent - 感知页面结构和特征
2. PlanAgent - 规划提取策略
3. ActAgent - 执行提取操作
4. VerifyAgent - 验证数据质量
5. JudgeAgent - 做出决策判断
6. ExploreAgent - 探索页面链接
7. ReflectAgent - 反思和优化策略

请根据当前情况选择最合适的组件，并说明理由。

格式:
组件: <组件名>
置信度: <0-1>
理由: <简短说明>
"""

    def _parse_llm_response(self, response: str) -> RouteResult:
        """解析 LLM 响应"""
        # 简单解析（实际应用中应该使用更复杂的解析器）
        try:
            lines = response.strip().split('\n')
            target = "SenseAgent"
            confidence = 0.7
            reasoning = "LLM 推荐"

            for line in lines:
                if line.startswith('组件:'):
                    target = line.split(':')[1].strip()
                elif line.startswith('置信度:'):
                    confidence = float(line.split(':')[1].strip())
                elif line.startswith('理由:'):
                    reasoning = line.split(':', 1)[1].strip()

            # 映射到实际组件名
            target_map = {
                '感知': 'SenseAgent',
                '规划': 'PlanAgent',
                '执行': 'ActAgent',
                '验证': 'VerifyAgent',
                '判断': 'JudgeAgent',
                '探索': 'ExploreAgent',
                '反思': 'ReflectAgent'
            }
            target = target_map.get(target, target)

            return RouteResult(
                decision=RouteDecision.LLM,
                target=target,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
        except Exception:
            return RouteResult(
                decision=RouteDecision.LLM,
                target="SenseAgent",
                confidence=0.6,
                reasoning="LLM 响应解析失败",
                timestamp=datetime.now()
            )

    # ==================== 程序判断规则 ====================

    def _is_clear_structure(self, context: Dict[str, Any]) -> bool:
        """判断页面结构是否清晰"""
        # 如果已经有明确的选择器定义
        selectors = context.get('selectors', {})
        if selectors and len(selectors) > 0:
            return True

        # 如果页面类型已知且简单
        page_type = context.get('page_type')
        simple_types = ['list', 'table', 'card']
        if page_type in simple_types:
            return True

        # 如果之前已经成功提取过
        success_count = context.get('success_count', 0)
        if success_count > 0:
            return True

        return False

    def _need_sense_page(self, context: Dict[str, Any]) -> bool:
        """判断是否需要感知页面"""
        # 如果是新页面
        if not context.get('page_analyzed', False):
            return True

        # 如果页面结构未知
        if not context.get('page_structure'):
            return True

        # 如果选择器为空
        selectors = context.get('selectors', {})
        if not selectors:
            return True

        return False

    def _need_explore(self, context: Dict[str, Any]) -> bool:
        """判断是否需要探索页面"""
        # 如果需要多页提取
        task_type = context.get('task_type')
        if task_type in ['multi_page', 'pagination', 'infinite_scroll']:
            extracted_count = context.get('extracted_count', 0)
            target_count = context.get('target_count', 10)
            if extracted_count < target_count:
                return True

        # 如果有未访问的链接
        if context.get('has_more_links', False):
            return True

        return False

    def _need_verify(self, context: Dict[str, Any]) -> bool:
        """判断是否需要验证"""
        # 如果已提取数据
        extracted_count = context.get('extracted_count', 0)
        if extracted_count > 0:
            # 如果还没有验证过
            if not context.get('verified', False):
                return True

        return False

    # ==================== 规则判断 ====================

    def _is_complex_structure(self, context: Dict[str, Any]) -> bool:
        """判断页面结构是否复杂"""
        # 动态内容
        if context.get('has_dynamic_content', False):
            return True

        # 嵌套层级深
        depth = context.get('structure_depth', 0)
        if depth > 3:
            return True

        # 字段数量多
        field_count = context.get('field_count', 0)
        if field_count > 10:
            return True

        # 有特殊结构（表格、树形等）
        special_structures = context.get('special_structures', [])
        if special_structures:
            return True

        return False

    def _need_judgment(self, context: Dict[str, Any]) -> bool:
        """判断是否需要做出判断"""
        # 多个选择器候选
        selector_candidates = context.get('selector_candidates', [])
        if len(selector_candidates) > 1:
            return True

        # 数据质量存疑
        quality_score = context.get('quality_score', 1.0)
        if quality_score < 0.7:
            return True

        # 有冲突的证据
        has_conflict = context.get('has_conflicting_evidence', False)
        if has_conflict:
            return True

        return False

    # ==================== 辅助方法 ====================

    def _load_program_rules(self):
        """加载程序判断规则"""
        # 实际应用中可以从配置文件加载
        pass

    def _load_heuristic_rules(self):
        """加载启发式规则"""
        # 实际应用中可以从配置文件加载
        pass

    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        return {
            'program_decisions': 0,
            'rule_decisions': 0,
            'llm_decisions': 0,
            'delegate_decisions': 0
        }
