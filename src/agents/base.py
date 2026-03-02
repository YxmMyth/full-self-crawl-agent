"""
AgentPool - 智能体池管理器
统一管理所有智能体类型
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import json
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """智能体能力枚举"""
    SENSE = "sense"
    PLAN = "plan"
    ACT = "act"
    VERIFY = "verify"
    JUDGE = "judge"
    EXPLORE = "explore"
    REFLECT = "reflect"
    SPA_HANDLE = "spa_handle"


class AgentInterface:
    """智能体接口"""

    def __init__(self, name: str, capability):
        self.name = name
        self.capability = capability

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行智能体任务"""
        raise NotImplementedError

    def can_handle(self, context: Dict[str, Any]) -> bool:
        """判断是否能处理当前上下文"""
        raise NotImplementedError

    def get_description(self) -> str:
        """获取智能体描述"""
        raise NotImplementedError


class DegradationTracker:
    """
    降级追踪器

    追踪 LLM 调用降级情况，提供警告和统计
    """

    def __init__(self, warning_threshold: int = 3):
        self.degradation_count = 0
        self.warning_threshold = warning_threshold
        self.degradation_history: List[Dict[str, Any]] = []

    def record_degradation(self, agent_name: str, operation: str, error: str) -> Dict[str, Any]:
        """
        记录降级事件

        Returns:
            包含 is_degraded 和 should_warn 的字典
        """
        self.degradation_count += 1
        event = {
            'agent': agent_name,
            'operation': operation,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.degradation_history.append(event)

        return {
            'is_degraded': True,
            'should_warn': self.degradation_count >= self.warning_threshold,
            'degradation_count': self.degradation_count,
            'message': f"LLM {operation} 降级 (总计: {self.degradation_count}次)"
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取降级统计"""
        return {
            'total_degradations': self.degradation_count,
            'warning_threshold': self.warning_threshold,
            'history': self.degradation_history[-10:]  # 最近10条
        }


def _safe_parse_json(response: str, context: str = "JSON解析") -> Dict:
    """
    安全解析 JSON 响应
    """
    if not response.strip():
        return {}

    # 尝试清理 Markdown 代码块
    if '```' in response:
        import re
        matches = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if matches:
            response = matches[0].strip()

    try:
        import json
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"{context} JSON解析失败: {e}")
        print(f"原始响应: {response[:500]}...")  # 仅打印前500字符
        return {}


class AgentPool:
    """
    智能体池

    管理 8 种能力的智能体：
    - Sense: 感知页面结构
    - Plan: 规划提取策略
    - Act: 执行提取操作
    - Verify: 验证数据质量
    - Judge: 做出决策判断
    - Explore: 探索页面链接
    - Reflect: 反思和优化策略
    - SpaHandle: 处理SPA页面
    """

    def __init__(self, llm_client=None, sandbox=None):
        from .sense import SenseAgent
        from .plan import PlanAgent
        from .act import ActAgent
        from .verify import VerifyAgent
        from .judge import JudgeAgent
        from .explore import ExploreAgent
        from .reflect import ReflectAgent

        # Store llm_client and degradation tracker to initialize SPAHandler later
        self.llm_client = llm_client
        # 共享的降级追踪器
        self.degradation_tracker = DegradationTracker()

        from .spa_handler import SPAHandler

        # Initialize agents
        self.agents = {
            AgentCapability.SENSE: SenseAgent(self.degradation_tracker),
            AgentCapability.PLAN: PlanAgent(llm_client, self.degradation_tracker),
            AgentCapability.ACT: ActAgent(sandbox=sandbox),
            AgentCapability.VERIFY: VerifyAgent(),
            AgentCapability.JUDGE: JudgeAgent(llm_client, self.degradation_tracker),
            AgentCapability.EXPLORE: ExploreAgent(),
            AgentCapability.REFLECT: ReflectAgent(llm_client, self.degradation_tracker),
            AgentCapability.SPA_HANDLE: SPAHandler(llm_client),
        }

    def get_agent(self, capability: AgentCapability) -> Optional[AgentInterface]:
        """获取指定能力的智能体"""
        return self.agents.get(capability)

    async def execute_capability(self, capability: AgentCapability,
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """执行指定能力（异步版本）"""
        agent = self.get_agent(capability)
        if not agent:
            return {'success': False, 'error': f'Unknown capability: {capability}'}

        if not agent.can_handle(context):
            return {'success': False, 'error': f'Agent cannot handle this context'}

        return await agent.execute(context)

    def get_all_capabilities(self) -> List[AgentCapability]:
        """获取所有能力"""
        return list(self.agents.keys())

    def get_capability_description(self, capability: AgentCapability) -> str:
        """获取能力描述"""
        agent = self.get_agent(capability)
        return agent.get_description() if agent else ''

    def set_verifier(self, verifier):
        """设置验证器"""
        from .verify import VerifyAgent
        self.agents[AgentCapability.VERIFY] = VerifyAgent(verifier)

    def get_degradation_stats(self) -> Dict[str, Any]:
        """获取降级统计"""
        return self.degradation_tracker.get_stats()

# Re-export agent classes for backward compatibility
from .sense import SenseAgent
from .plan import PlanAgent
from .act import ActAgent
from .explore import ExploreAgent
from .verify import VerifyAgent
from .judge import JudgeAgent


class ExtractionMetrics(dict):
    """兼容占位类型：历史代码从 base 导入 ExtractionMetrics。"""
    pass
