"""
端到端测试 - 测试完整迭代循环和错误恢复
"""

import pytest
import sys
import os
import asyncio
import tempfile
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class MockBrowserTool:
    """模拟浏览器工具"""

    def __init__(self, html_content: str = "<html><body>Test</body></html>"):
        self.html_content = html_content
        self.page = AsyncMock()
        self.page.url = "https://example.com"
        self.page.query_selector = AsyncMock(return_value=None)
        self.page.evaluate = AsyncMock(return_value=1000)
        self.page.wait_for_load_state = AsyncMock()
        self.page.wait_for_timeout = AsyncMock()
        self.is_started = False

    async def start(self):
        self.is_started = True

    async def stop(self):
        self.is_started = False

    async def navigate(self, url: str, **kwargs):
        """模拟导航"""
        pass

    async def get_html(self) -> str:
        return self.html_content

    async def take_screenshot(self, **kwargs) -> bytes:
        return b"mock_screenshot_data"

    async def scroll_to_bottom(self, delay: float = 0.5):
        pass

    async def get_current_url(self) -> str:
        return "https://example.com/page/1"

    async def wait_for_selector(self, selector: str, **kwargs) -> bool:
        return True

    async def wait_for_page_ready(self, **kwargs) -> bool:
        return True


class MockLLMClient:
    """模拟LLM客户端"""

    def __init__(self, responses: list = None):
        self.responses = responses or ["{'strategy_type': 'css'}"]
        self.response_index = 0
        self.call_count = 0
        self.total_tokens = 0

    async def chat(self, messages: list, **kwargs) -> str:
        self.call_count += 1
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        return "{}"

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.chat([{"role": "user", "content": prompt}])

    def get_stats(self) -> dict:
        return {
            'call_count': self.call_count,
            'total_tokens': self.total_tokens,
            'model': 'mock-model',
            'provider': 'mock'
        }

    async def close(self):
        pass


# ==================== 测试用例 ====================

@pytest.mark.asyncio
async def test_sense_agent_basic():
    """测试感知智能体基本功能"""
    from src.agents.base import SenseAgent, DegradationTracker

    # 创建模拟浏览器
    mock_html = """
    <html>
    <body>
        <div class="product-list">
            <div class="item">
                <h3 class="title">Product 1</h3>
                <span class="price">$99.99</span>
            </div>
        </div>
        <a class="next" href="/page/2">Next</a>
    </body>
    </html>
    """
    browser = MockBrowserTool(mock_html)

    # 创建感知智能体
    agent = SenseAgent()

    # 执行
    context = {
        'browser': browser,
        'spec': {'goal': '爬取产品列表'}
    }

    result = await agent.execute(context)

    # 验证
    assert result['success'] is True
    assert 'structure' in result
    assert 'features' in result
    assert result['structure']['type'] in ['list', 'detail', 'form', 'other']
    print(f"\n✅ 感知智能体测试通过")
    print(f"   页面类型: {result['structure']['type']}")
    print(f"   分页类型: {result['structure']['pagination_type']}")


@pytest.mark.asyncio
async def test_plan_agent_with_fallback():
    """测试规划智能体降级策略"""
    from src.agents.base import PlanAgent, DegradationTracker

    # 创建不带LLM的规划智能体（会使用降级策略）
    agent = PlanAgent()

    context = {
        'page_structure': {
            'type': 'list',
            'pagination_type': 'click',
            'main_content_selector': '.product-item',
            'estimated_items': 10
        },
        'spec': {
            'targets': [{
                'name': 'products',
                'fields': [
                    {'name': 'title', 'selector': '.title'},
                    {'name': 'price', 'selector': '.price'}
                ]
            }]
        }
    }

    result = await agent.execute(context)

    # 验证
    assert result['success'] is True
    assert 'strategy' in result
    assert 'selectors' in result
    assert 'generated_code' in result
    assert '.title' in result['selectors'] or 'title' in result['selectors']
    print(f"\n✅ 规划智能体降级策略测试通过")
    print(f"   策略类型: {result['strategy']['strategy_type']}")
    print(f"   选择器数量: {len(result['selectors'])}")


@pytest.mark.asyncio
async def test_act_agent_extraction_metrics():
    """测试执行智能体提取指标"""
    from src.agents.base import ActAgent, ExtractionMetrics

    # 创建模拟浏览器
    mock_html = """
    <html>
    <body>
        <div class="product-item">
            <h3 class="title">Product 1</h3>
            <span class="price">$99.99</span>
        </div>
        <div class="product-item">
            <h3 class="title">Product 2</h3>
            <span class="price">$199.99</span>
        </div>
        <div class="product-item">
            <h3 class="title">Product 3</h3>
            <span class="price"></span>
        </div>
    </body>
    </html>
    """
    browser = MockBrowserTool(mock_html)

    # 创建执行智能体
    agent = ActAgent()

    context = {
        'browser': browser,
        'selectors': {
            'title': '.title',
            'price': '.price'
        },
        'strategy': {
            'container_selector': '.product-item',
            'pagination_strategy': 'none'
        },
        'spec': {
            'targets': [{
                'name': 'products',
                'fields': [
                    {'name': 'title', 'selector': '.title', 'required': True},
                    {'name': 'price', 'selector': '.price', 'required': False}
                ]
            }]
        }
    }

    result = await agent.execute(context)

    # 验证
    assert result['success'] is True
    assert len(result['extracted_data']) == 3
    assert 'extraction_metrics' in result

    metrics = result['extraction_metrics']
    assert 'total_items' in metrics
    assert 'failed_selectors' in metrics
    print(f"\n✅ 执行智能体提取指标测试通过")
    print(f"   提取数量: {result['count']}")
    print(f"   成功率: {metrics['success_rate']:.1%}")


@pytest.mark.asyncio
async def test_verify_agent_quality_check():
    """测试验证智能体质量检查"""
    from src.agents.base import VerifyAgent

    agent = VerifyAgent()

    # 测试数据
    context = {
        'extracted_data': [
            {'title': 'Product 1', 'price': '$99.99'},
            {'title': 'Product 2', 'price': '$199.99'},
            {'title': '', 'price': '$299.99'},  # 缺少必填字段
            {'title': 'Product 4', 'price': ''},  # 价格为空
        ],
        'spec': {
            'targets': [{
                'name': 'products',
                'fields': [
                    {'name': 'title', 'required': True},
                    {'name': 'price', 'required': False}
                ]
            }]
        },
        'extraction_metrics': {
            'total_items': 4,
            'missing_fields': {'title': 1},
            'empty_fields': {'price': 1}
        }
    }

    result = await agent.execute(context)

    # 验证
    assert result['success'] is True
    assert 'quality_score' in result
    assert result['quality_score'] >= 0 and result['quality_score'] <= 1
    assert 'verification_result' in result
    print(f"\n✅ 验证智能体测试通过")
    print(f"   质量分数: {result['quality_score']:.2f}")
    print(f"   有效数据: {result['valid_items']}/{result['total_items']}")


@pytest.mark.asyncio
async def test_judge_agent_decision():
    """测试决策智能体决策逻辑"""
    from src.agents.base import JudgeAgent

    agent = JudgeAgent()

    # 测试场景1: 质量分数高，应该完成
    context = {
        'quality_score': 0.9,
        'iteration': 0,
        'max_iterations': 10,
        'errors': [],
        'spec': {},
        'extracted_data': [{'title': 'Product 1'} for _ in range(10)]
    }

    result = await agent.execute(context)

    assert result['success'] is True
    assert result['decision'] == 'complete'
    assert 'reasoning' in result
    print(f"\n✅ 决策智能体测试通过")
    print(f"   决策: {result['decision']}")
    print(f"   原因: {result['reasoning']}")

    # 测试场景2: 质量分数中等，应该继续迭代
    context2 = {
        'quality_score': 0.5,
        'iteration': 2,
        'max_iterations': 10,
        'errors': ['selector_error'],
        'spec': {},
        'extracted_data': [{'title': 'Product 1'} for _ in range(3)]
    }

    result2 = await agent.execute(context2)
    assert result2['decision'] == 'reflect_and_retry'
    print(f"   决策(中等质量): {result2['decision']}")


@pytest.mark.asyncio
async def test_degradation_tracker():
    """测试降级追踪器"""
    from src.agents.base import DegradationTracker

    tracker = DegradationTracker(warning_threshold=2)

    # 记录第一次降级
    info1 = tracker.record_degradation('SenseAgent', 'llm_analyze', 'Timeout')
    assert info1['is_degraded'] is True
    assert info1['should_warn'] is False

    # 记录第二次降级
    info2 = tracker.record_degradation('PlanAgent', 'generate_strategy', 'API Error')
    assert info2['should_warn'] is True  # 达到阈值

    # 获取统计
    stats = tracker.get_stats()
    assert stats['total_degradations'] == 2
    assert len(stats['history']) == 2

    print(f"\n✅ 降级追踪器测试通过")
    print(f"   总降级次数: {stats['total_degradations']}")
    print(f"   警告阈值: {stats['warning_threshold']}")


@pytest.mark.asyncio
async def test_browser_retry_mechanism():
    """测试浏览器重试机制"""
    from src.tools.browser import BrowserTool, with_retry

    # 模拟重试场景
    call_count = 0

    class FailingBrowser:
        async def failing_operation(self):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                from playwright.async_api import Error
                raise Error("Navigation failed")
            return "success"

    # 测试重试装饰器
    browser = FailingBrowser()

    # 应用装饰器
    @with_retry(max_retries=3, base_delay=0.1, max_delay=1.0)
    async def operation(self):
        return await self.failing_operation()

    result = await operation(browser)
    assert result == "success"
    assert call_count == 3  # 失败2次，成功1次

    print(f"\n✅ 浏览器重试机制测试通过")
    print(f"   重试次数: {call_count - 1}")


@pytest.mark.asyncio
async def test_state_manager_concurrency():
    """测试状态管理器并发安全"""
    from src.core.state_manager import StateManager

    manager = StateManager()
    await manager.create_initial_state('test_task', {'goal': 'test'})

    # 并发更新
    async def update_state(i):
        await manager.update_state({'iteration': i}, f"update_{i}")

    # 并发执行100次更新
    tasks = [update_state(i) for i in range(100)]
    await asyncio.gather(*tasks)

    # 验证更新计数
    assert manager.get_update_count() == 100

    print(f"\n✅ 状态管理器并发测试通过")
    print(f"   更新次数: {manager.get_update_count()}")


@pytest.mark.asyncio
async def test_llm_retry_mechanism():
    """测试LLM客户端重试机制"""
    from src.tools.llm_client import LLMClient, LLMException, ErrorType

    # 创建模拟客户端
    client = LLMClient(api_key='test_key', model='test_model')

    # 模拟API调用
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_response = Mock()
        if call_count < 3:
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
        else:
            mock_response.status_code = 200
            mock_response.json = lambda: {
                'choices': [{'message': {'content': 'success'}}],
                'usage': {'total_tokens': 100}
            }
        return mock_response

    client.client.post = mock_post

    # 测试重试
    result = await client.generate("test prompt", max_retries=3)

    assert call_count == 3
    assert result == 'success'
    assert client.retry_count == 2

    print(f"\n✅ LLM重试机制测试通过")
    print(f"   API调用次数: {call_count}")
    print(f"   重试次数: {client.retry_count}")


@pytest.mark.asyncio
async def test_full_iteration_loop():
    """测试完整迭代循环"""
    from src.agents.base import AgentPool, AgentCapability

    # 创建模拟浏览器
    mock_html = """
    <html>
    <body>
        <div class="product-item">
            <h3 class="title">Product 1</h3>
            <span class="price">$99.99</span>
        </div>
        <div class="product-item">
            <h3 class="title">Product 2</h3>
            <span class="price">$199.99</span>
        </div>
    </body>
    </html>
    """
    browser = MockBrowserTool(mock_html)

    # 创建智能体池
    pool = AgentPool()

    # 1. 感知
    sense_result = await pool.execute_capability(
        AgentCapability.SENSE,
        {'browser': browser, 'spec': {'goal': '爬取产品'}}
    )
    assert sense_result['success']

    # 2. 规划
    plan_result = await pool.execute_capability(
        AgentCapability.PLAN,
        {
            'page_structure': sense_result['structure'],
            'spec': {
                'targets': [{
                    'name': 'products',
                    'fields': [
                        {'name': 'title', 'selector': '.title'},
                        {'name': 'price', 'selector': '.price'}
                    ]
                }]
            }
        }
    )
    assert plan_result['success']

    # 3. 执行
    act_result = await pool.execute_capability(
        AgentCapability.ACT,
        {
            'browser': browser,
            'selectors': plan_result['selectors'],
            'strategy': plan_result['strategy']
        }
    )
    assert act_result['success']

    # 4. 验证
    verify_result = await pool.execute_capability(
        AgentCapability.VERIFY,
        {
            'extracted_data': act_result['extracted_data'],
            'spec': {'targets': [{'name': 'products', 'fields': []}]}
        }
    )
    assert verify_result['success']

    # 5. 决策
    judge_result = await pool.execute_capability(
        AgentCapability.JUDGE,
        {
            'quality_score': verify_result['quality_score'],
            'iteration': 0,
            'max_iterations': 5,
            'errors': [],
            'extracted_data': act_result['extracted_data']
        }
    )
    assert judge_result['success']
    assert judge_result['decision'] in ['complete', 'reflect_and_retry', 'terminate']

    print(f"\n✅ 完整迭代循环测试通过")
    print(f"   感知成功: {sense_result['success']}")
    print(f"   规划成功: {plan_result['success']}")
    print(f"   提取数量: {act_result['count']}")
    print(f"   质量分数: {verify_result['quality_score']:.2f}")
    print(f"   最终决策: {judge_result['decision']}")


@pytest.mark.asyncio
async def test_error_recovery_path():
    """测试错误恢复路径"""
    from src.agents.base import AgentPool, AgentCapability

    # 创建智能体池
    pool = AgentPool()

    # 模拟错误场景：can_handle 返回 False
    context = {'browser': None}

    # SenseAgent 的 can_handle 会返回 False (browser is None)
    agent = pool.get_agent(AgentCapability.SENSE)
    assert agent.can_handle(context) is False, "can_handle should return False when browser is None"

    # 执行会返回错误（因为 can_handle 返回 False）
    sense_result = await pool.execute_capability(
        AgentCapability.SENSE,
        context
    )

    # 应该返回失败
    assert sense_result['success'] is False
    assert 'error' in sense_result

    # 模拟反射智能体处理错误
    reflect_result = await pool.execute_capability(
        AgentCapability.REFLECT,
        {
            'execution_history': [{'stage': 'sense', 'error': 'browser not initialized'}],
            'errors': ['browser_error: Browser is None'],
            'quality_score': 0,
            'spec': {}
        }
    )

    assert reflect_result['success']
    assert 'improvements' in reflect_result
    assert 'suggested_action' in reflect_result

    print(f"\n✅ 错误恢复路径测试通过")
    print(f"   错误检测: {sense_result['error']}")
    print(f"   恢复建议: {reflect_result['suggested_action']}")


# ==================== 主函数 ====================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 运行端到端测试")
    print("=" * 60)

    asyncio.run(test_sense_agent_basic())
    asyncio.run(test_plan_agent_with_fallback())
    asyncio.run(test_act_agent_extraction_metrics())
    asyncio.run(test_verify_agent_quality_check())
    asyncio.run(test_judge_agent_decision())
    asyncio.run(test_degradation_tracker())
    asyncio.run(test_browser_retry_mechanism())
    asyncio.run(test_state_manager_concurrency())
    asyncio.run(test_llm_retry_mechanism())
    asyncio.run(test_full_iteration_loop())
    asyncio.run(test_error_recovery_path())

    print("\n" + "=" * 60)
    print("✅ 所有端到端测试通过!")
    print("=" * 60 + "\n")