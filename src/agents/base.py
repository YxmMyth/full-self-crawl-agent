"""
Agent 能力定义
执行层的 7 种智能体能力
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum


class AgentCapability(str, Enum):
    """智能体能力"""
    SENSE = "sense"  # 感知页面结构和特征
    PLAN = "plan"  # 规划提取策略
    ACT = "act"  # 执行提取操作
    VERIFY = "verify"  # 验证数据质量
    JUDGE = "judge"  # 做出决策判断
    EXPLORE = "explore"  # 探索页面链接
    REFLECT = "reflect"  # 反思和优化策略


class AgentInterface:
    """智能体接口"""

    def __init__(self, name: str, capability: AgentCapability):
        self.name = name
        self.capability = capability

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行智能体

        Args:
            context: 执行上下文

        Returns:
            执行结果
        """
        raise NotImplementedError()

    def get_description(self) -> str:
        """获取智能体描述"""
        raise NotImplementedError()

    def can_handle(self, context: Dict[str, Any]) -> bool:
        """判断是否能处理"""
        raise NotImplementedError()


# ==================== 感知智能体 ====================

class SenseAgent(AgentInterface):
    """感知智能体 - 分析页面结构"""

    def __init__(self):
        super().__init__("SenseAgent", AgentCapability.SENSE)

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        browser = context.get('browser')
        html = await browser.get_html()
        screenshot = await browser.take_screenshot()

        # 分析页面结构
        structure = self._analyze_structure(html)
        features = self._detect_features(html)
        anti_bot = self._detect_anti_bot(html)

        return {
            'success': True,
            'structure': structure,
            'features': features,
            'anti_bot_detected': anti_bot,
            'html_snapshot': html,
            'screenshot': screenshot
        }

    def _analyze_structure(self, html: str) -> Dict[str, Any]:
        """分析页面结构"""
        # TODO: 实现页面结构分析
        return {
            'type': 'unknown',
            'complexity': 'medium',
            'has_dynamic_content': False
        }

    def _detect_features(self, html: str) -> Dict[str, Any]:
        """检测页面特征"""
        # TODO: 实现特征检测
        return {
            'has_table': False,
            'has_list': True,
            'has_pagination': False
        }

    def _detect_anti_bot(self, html: str) -> bool:
        """检测反爬机制"""
        # TODO: 实现反爬检测
        anti_bot_keywords = ['cloudflare', 'recaptcha', 'challenge']
        return any(keyword in html.lower() for keyword in anti_bot_keywords)

    def get_description(self) -> str:
        return "感知页面结构和特征，识别页面类型和反爬机制"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'browser' in context


# ==================== 规划智能体 ====================

class PlanAgent(AgentInterface):
    """规划智能体 - 生成提取策略"""

    def __init__(self, llm_client=None):
        super().__init__("PlanAgent", AgentCapability.PLAN)
        self.llm_client = llm_client

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        page_structure = context.get('page_structure', {})
        spec = context.get('spec')

        # 基于页面结构和契约生成策略
        strategy = self._generate_strategy(page_structure, spec)

        return {
            'success': True,
            'strategy': strategy,
            'selectors': strategy.get('selectors', {})
        }

    def _generate_strategy(self, structure: Dict[str, Any], spec: Any) -> Dict[str, Any]:
        """生成提取策略"""
        if not self.llm_client:
            return self._fallback_strategy(structure, spec)

        # 使用 LLM 生成策略
        prompt = self._build_prompt(structure, spec)
        response = self.llm_client.generate(prompt)

        return self._parse_llm_response(response)

    def _fallback_strategy(self, structure: Dict[str, Any], spec: Any) -> Dict[str, Any]:
        """降级策略"""
        selectors = {}
        for target in spec.targets:
            for field in target.fields:
                selectors[field.name] = field.selector

        return {
            'type': 'simple',
            'selectors': selectors,
            'approach': 'direct_extraction'
        }

    def _build_prompt(self, structure: Dict[str, Any], spec: Any) -> str:
        """构建 LLM 提示"""
        return f"""基于页面结构和提取目标，生成最佳提取策略。

页面结构: {structure}
提取目标: {spec.targets}

请生成包含以下内容的策略：
1. 提取方法（直接选择器/分页/滚动等）
2. 每个字段的选择器
3. 特殊处理逻辑

输出格式：JSON
"""

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 响应"""
        import json
        try:
            return json.loads(response)
        except:
            return self._fallback_strategy({}, None)

    def get_description(self) -> str:
        return "基于页面结构和契约生成智能提取策略"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'spec' in context


# ==================== 执行智能体 ====================

class ActAgent(AgentInterface):
    """执行智能体 - 执行提取操作"""

    def __init__(self):
        super().__init__("ActAgent", AgentCapability.ACT)

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        browser = context.get('browser')
        selectors = context.get('selectors', {})
        strategy = context.get('strategy', {})

        # 执行提取
        extracted_data = await self._extract_data(browser, selectors, strategy)

        return {
            'success': True,
            'extracted_data': extracted_data,
            'count': len(extracted_data)
        }

    async def _extract_data(self, browser, selectors: Dict[str, str],
                           strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行数据提取"""
        # 获取页面内容
        html = await browser.get_html()

        # 使用解析器提取数据
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        data = []
        # TODO: 实现实际的提取逻辑
        # 根据选择器和策略提取数据

        return data

    def get_description(self) -> str:
        return "执行实际的数据提取操作"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'browser' in context and 'selectors' in context


# ==================== 验证智能体 ====================

class VerifyAgent(AgentInterface):
    """验证智能体 - 验证数据质量"""

    def __init__(self, verifier=None):
        super().__init__("VerifyAgent", AgentCapability.VERIFY)
        self.verifier = verifier

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        extracted_data = context.get('extracted_data', [])
        spec = context.get('spec')

        # 验证数据
        verification_result = self.verifier.verify(extracted_data, context)

        return {
            'success': True,
            'verification_result': verification_result,
            'valid_items': verification_result.get('valid_items', 0),
            'total_items': verification_result.get('total_items', 0)
        }

    def get_description(self) -> str:
        return "验证提取数据的质量和完整性"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'extracted_data' in context


# ==================== 决策智能体 ====================

class JudgeAgent(AgentInterface):
    """决策智能体 - 做出关键决策"""

    def __init__(self, llm_client=None):
        super().__init__("JudgeAgent", AgentCapability.JUDGE)
        self.llm_client = llm_client

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        options = context.get('options', [])
        criteria = context.get('criteria', {})

        # 评估选项
        decision = self._make_decision(options, criteria)

        return {
            'success': True,
            'decision': decision,
            'reasoning': decision.get('reasoning', '')
        }

    def _make_decision(self, options: List[Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """做出决策"""
        if not self.llm_client:
            return {'option': options[0], 'confidence': 0.5, 'reasoning': '默认选择'}

        # 使用 LLM 做决策
        prompt = self._build_decision_prompt(options, criteria)
        response = self.llm_client.generate(prompt)

        return self._parse_decision(response)

    def _build_decision_prompt(self, options: List[Any], criteria: Dict[str, Any]) -> str:
        """构建决策提示"""
        return f"""请基于以下标准做出决策：

选项: {options}
标准: {criteria}

请选择最佳选项并说明理由。

输出格式：
选项: <选项>
置信度: <0-1>
理由: <说明>
"""

    def _parse_decision(self, response: str) -> Dict[str, Any]:
        """解析决策结果"""
        # 简单解析
        return {
            'option': 'option_1',
            'confidence': 0.7,
            'reasoning': '基于分析做出的决策'
        }

    def get_description(self) -> str:
        return "在多个选项间做出最优决策"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'options' in context


# ==================== 探索智能体 ====================

class ExploreAgent(AgentInterface):
    """探索智能体 - 探索页面链接和结构"""

    def __init__(self):
        super().__init__("ExploreAgent", AgentCapability.EXPLORE)

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        browser = context.get('browser')
        current_url = context.get('current_url', '')
        depth = context.get('depth', 0)
        max_depth = context.get('max_depth', 2)

        if depth >= max_depth:
            return {'success': True, 'links': [], 'message': '已达到最大探索深度'}

        # 获取页面中的所有链接
        links = await self._extract_links(browser)

        return {
            'success': True,
            'links': links,
            'count': len(links),
            'next_depth': depth + 1
        }

    async def _extract_links(self, browser) -> List[str]:
        """提取页面链接"""
        html = await browser.get_html()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        links = []
        for a in soup.find_all('a', href=True):
            links.append(a['href'])

        return links

    def get_description(self) -> str:
        return "探索页面链接，发现新的数据源"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'browser' in context


# ==================== 反思智能体 ====================

class ReflectAgent(AgentInterface):
    """反思智能体 - 反思和优化策略"""

    def __init__(self, llm_client=None):
        super().__init__("ReflectAgent", AgentCapability.REFLECT)
        self.llm_client = llm_client

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        execution_history = context.get('execution_history', [])
        errors = context.get('errors', [])

        # 分析失败原因
        analysis = self._analyze_failures(execution_history, errors)

        # 生成优化建议
        improvements = self._generate_improvements(analysis)

        return {
            'success': True,
            'analysis': analysis,
            'improvements': improvements,
            'suggested_action': improvements.get('action', 'retry')
        }

    def _analyze_failures(self, history: List[Dict[str, Any]],
                         errors: List[str]) -> Dict[str, Any]:
        """分析失败原因"""
        if not self.llm_client:
            return self._fallback_analysis(errors)

        # 使用 LLM 分析
        prompt = self._build_analysis_prompt(history, errors)
        response = self.llm_client.generate(prompt)

        return self._parse_analysis(response)

    def _fallback_analysis(self, errors: List[str]) -> Dict[str, Any]:
        """降级分析"""
        common_errors = {}
        for error in errors:
            common_errors[error] = common_errors.get(error, 0) + 1

        return {
            'error_patterns': common_errors,
            'likely_cause': 'selector_issue',
            'confidence': 0.6
        }

    def _generate_improvements(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成改进方案"""
        if not self.llm_client:
            return {'action': 'retry', 'suggestions': []}

        prompt = self._build_improvement_prompt(analysis)
        response = self.llm_client.generate(prompt)

        return self._parse_improvements(response)

    def _build_analysis_prompt(self, history: List[Dict[str, Any]],
                              errors: List[str]) -> str:
        return f"""分析以下执行历史和错误，找出失败的根本原因：

执行历史: {history}
错误列表: {errors}

请分析：
1. 主要问题是什么？
2. 是选择器问题、页面结构问题还是其他问题？
3. 如何解决？

输出格式：JSON
"""

    def _build_improvement_prompt(self, analysis: Dict[str, Any]) -> str:
        return f"""基于以下分析，生成改进方案：

问题分析: {analysis}

请建议：
1. 具体的改进措施
2. 下一步应该采取什么行动

输出格式：JSON
"""

    def _parse_analysis(self, response: str) -> Dict[str, Any]:
        import json
        try:
            return json.loads(response)
        except:
            return self._fallback_analysis([])

    def _parse_improvements(self, response: str) -> Dict[str, Any]:
        import json
        try:
            return json.loads(response)
        except:
            return {'action': 'retry', 'suggestions': []}

    def get_description(self) -> str:
        return "分析失败原因，生成优化建议"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return len(context.get('errors', [])) > 0


# ==================== 智能体池 ====================

class AgentPool:
    """
    智能体池

    管理 7 种能力的智能体：
    - Sense: 感知页面结构
    - Plan: 规划提取策略
    - Act: 执行提取操作
    - Verify: 验证数据质量
    - Judge: 做出决策判断
    - Explore: 探索页面链接
    - Reflect: 反思和优化策略
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.agents = {
            AgentCapability.SENSE: SenseAgent(),
            AgentCapability.PLAN: PlanAgent(llm_client),
            AgentCapability.ACT: ActAgent(),
            AgentCapability.VERIFY: VerifyAgent(),
            AgentCapability.JUDGE: JudgeAgent(llm_client),
            AgentCapability.EXPLORE: ExploreAgent(),
            AgentCapability.REFLECT: ReflectAgent(llm_client)
        }

    def get_agent(self, capability: AgentCapability) -> Optional[AgentInterface]:
        """获取指定能力的智能体"""
        return self.agents.get(capability)

    def execute_capability(self, capability: AgentCapability,
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """执行指定能力"""
        agent = self.get_agent(capability)
        if not agent:
            return {'success': False, 'error': f'Unknown capability: {capability}'}

        if not agent.can_handle(context):
            return {'success': False, 'error': f'Agent cannot handle this context'}

        import asyncio
        return asyncio.run(agent.execute(context))

    def get_all_capabilities(self) -> List[AgentCapability]:
        """获取所有能力"""
        return list(self.agents.keys())

    def get_capability_description(self, capability: AgentCapability) -> str:
        """获取能力描述"""
        agent = self.get_agent(capability)
        return agent.get_description() if agent else ''
