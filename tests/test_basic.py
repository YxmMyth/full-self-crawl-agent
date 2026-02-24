"""Tests - 测试文件"""

import pytest
import asyncio
import json
from pathlib import Path


class TestConfig:
    """测试配置加载"""

    def test_spec_contract_creation(self):
        """测试 SpecContract 创建"""
        from src.config.contracts import SpecContract, ExtractionType

        spec = SpecContract(
            task_id='test_001',
            task_name='Test Task',
            created_at='2026-02-25T00:00:00',
            extraction_type=ExtractionType.SINGLE_PAGE
        )

        assert spec.task_id == 'test_001'
        assert spec.task_name == 'Test Task'

    def test_spec_freeze(self):
        """测试契约冻结"""
        from src.config.contracts import SpecContract

        spec = SpecContract(
            task_id='test',
            task_name='Test',
            created_at='2026-02-25T00:00:00'
        )

        # 契约应该被冻结
        with pytest.raises(PermissionError):
            spec.task_name = 'Modified'


class TestTools:
    """测试工具层"""

    @pytest.mark.asyncio
    async def test_browser_tool(self):
        """测试浏览器工具"""
        from src.tools.browser import BrowserTool

        browser = BrowserTool(headless=True)
        await browser.start()
        await browser.stop()

    async def test_llm_client(self):
        """测试 LLM 客户端"""
        from src.tools.llm_client import LLMClient

        # 跳过实际调用，只测试初始化
        client = LLMClient('test_key', model='glm-4')
        assert client.model == 'glm-4'


class TestCore:
    """测试核心组件"""

    def test_policy_manager(self):
        """测试策略管理器"""
        from src.core.policy_manager import PolicyManager

        policy = PolicyManager()
        result = policy.check_code('print("hello")')

        assert 'allowed' in result

    def test_smart_router(self):
        """测试智能路由"""
        from src.core.smart_router import SmartRouter

        router = SmartRouter()
        context = {
            'extracted_count': 5,
            'target_count': 10
        }

        result = router.route(context, 'crawl')
        assert result is not None


class TestAgents:
    """测试智能体"""

    def test_agent_pool(self):
        """测试智能体池"""
        from src.agents import AgentPool

        pool = AgentPool()
        assert pool.get_all_capabilities() is not None


class TestStorage:
    """测试存储"""

    def test_evidence_storage(self):
        """测试证据存储"""
        from src.tools.storage import EvidenceStorage

        storage = EvidenceStorage()
        task_dir = storage.create_task_dir('test_task')

        assert task_dir.exists()
        assert (task_dir / 'screenshots').exists()
        assert (task_dir / 'data').exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
