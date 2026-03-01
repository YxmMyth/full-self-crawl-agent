"""
智能体能力单元测试
"""

import pytest
import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_agent_pool_initialization():
    """测试智能体池初始化"""
    from src.agents.base import AgentPool
    
    pool = AgentPool()
    
    # 检查所有能力是否都已注册（含新增的 SPA_HANDLE）
    capabilities = pool.get_all_capabilities()
    assert len(capabilities) == 8
    
    # 检查具体能力
    from src.agents.base import AgentCapability
    assert AgentCapability.SENSE in capabilities
    assert AgentCapability.PLAN in capabilities
    assert AgentCapability.ACT in capabilities
    assert AgentCapability.VERIFY in capabilities
    assert AgentCapability.JUDGE in capabilities
    assert AgentCapability.EXPLORE in capabilities
    assert AgentCapability.REFLECT in capabilities
    assert AgentCapability.SPA_HANDLE in capabilities


def test_sense_agent_description():
    """测试感知智能体描述"""
    from src.agents.base import SenseAgent
    
    agent = SenseAgent()
    description = agent.get_description()
    
    assert '感知' in description or '分析' in description
    assert len(description) > 0


def test_plan_agent_description():
    """测试规划智能体描述"""
    from src.agents.base import PlanAgent
    
    agent = PlanAgent()
    description = agent.get_description()
    
    assert len(description) > 0


def test_act_agent_description():
    """测试执行智能体描述"""
    from src.agents.base import ActAgent
    
    agent = ActAgent()
    description = agent.get_description()
    
    assert len(description) > 0


def test_capability_descriptions():
    """测试所有能力的描述"""
    from src.agents.base import AgentPool
    
    pool = AgentPool()
    
    for capability in pool.get_all_capabilities():
        desc = pool.get_capability_description(capability)
        assert len(desc) > 0


def test_agent_can_handle():
    """测试智能体能否处理"""
    from src.agents.base import SenseAgent, ActAgent
    
    sense_agent = SenseAgent()
    act_agent = ActAgent()
    
    # Sense需要browser
    assert sense_agent.can_handle({'browser': 'test'}) is True
    assert sense_agent.can_handle({}) is False
    
    # Act需要browser和selectors
    assert act_agent.can_handle({'browser': 'test', 'selectors': {}}) is True
    assert act_agent.can_handle({'browser': 'test'}) is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
