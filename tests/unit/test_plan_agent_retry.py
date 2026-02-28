"""
测试 PlanAgent 的单元内重试架构
"""

import pytest
import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.mark.asyncio
async def test_plan_agent_retry_normal():
    """测试正常情况下的 PlanAgent"""
    from src.agents.base import PlanAgent

    agent = PlanAgent()

    # 模拟上下文
    context = {
        'page_structure': {
            'main_content_selector': 'body',
            'pagination_type': 'none'
        },
        'spec': {
            'targets': [
                {
                    'name': 'test_target',
                    'fields': [
                        {'name': 'title', 'selector': 'h1'},
                        {'name': 'price', 'selector': '.price'},
                    ]
                }
            ]
        },
        'html_snapshot': '''
            <html>
                <body>
                    <h1>Test Title</h1>
                    <div class="price">100</div>
                </body>
            </html>
        '''
    }

    result = await agent.execute(context)

    assert result['success'] == True
    assert 'title' in result['selectors']
    assert 'price' in result['selectors']
    print("[PASS] Normal generation test passed")


@pytest.mark.asyncio
async def test_plan_agent_conservative_strategy():
    """测试保守策略 - 当选择器无效时，应该使用更宽泛的选择器"""
    from src.agents.base import PlanAgent

    agent = PlanAgent()

    # 模拟无效的选择器情况
    context = {
        'page_structure': {
            'main_content_selector': 'body',
            'pagination_type': 'none'
        },
        'spec': {
            'targets': [
                {
                    'name': 'test_target',
                    'fields': [
                        {'name': 'title', 'selector': '.nonexistent-title'},
                        {'name': 'author', 'selector': '.nonexistent-author'},
                    ]
                }
            ]
        },
        'html_snapshot': '''
            <html>
                <body>
                    <h1>Real Title</h1>
                    <span class="real-author">John Doe</span>
                </body>
            </html>
        '''
    }

    result = await agent.execute(context)

    assert result['success'] == True
    # 应该至少有一个选择器
    assert len(result['selectors']) > 0
    print("[PASS] Conservative strategy test passed")


@pytest.mark.asyncio
async def test_plan_agent_aggressive_strategy():
    """测试激进策略 - 当前两次都失败时"""
    from src.agents.base import PlanAgent

    agent = PlanAgent()

    # 模拟极端情况：所有选择器都无效
    context = {
        'page_structure': {
            'main_content_selector': 'body',
            'pagination_type': 'none'
        },
        'spec': {
            'targets': [
                {
                    'name': 'test_target',
                    'fields': [
                        {'name': 'field1', 'selector': '.nonexistent'},
                        {'name': 'field2', 'selector': '#nonexistent'},
                        {'name': 'field3', 'selector': '[data-test="nonexistent"]'},
                    ]
                }
            ]
        },
        'html_snapshot': '''
            <html>
                <body>
                    <div class="container">
                        <div class="item">
                            <h1>标题</h1>
                            <span class="value">123</span>
                        </div>
                    </div>
                </body>
            </html>
        '''
    }

    result = await agent.execute(context)

    assert result['success'] == True
    # 激进策略应该仍然返回一些选择器
    assert len(result['selectors']) > 0
    print("[PASS] Aggressive strategy test passed")


def test_conservative_strategy_generation():
    """直接测试保守策略生成逻辑"""
    from src.agents.base import PlanAgent
    import asyncio

    agent = PlanAgent()

    structure = {'main_content_selector': 'body'}
    spec = {
        'targets': [
            {
                'fields': [
                    {'name': 'title'},
                    {'name': 'price'},
                    {'name': 'author'},
                ]
            }
        ]
    }
    html = '''
        <html>
            <body>
                <h1>Test</h1>
                <div class="real-price">100</div>
                <span class="author-name">John</span>
            </body>
        </html>
    '''

    # 使用 asyncio 运行异步方法
    async def run_test():
        result = await agent._generate_conservative(structure, spec, None, html)
        assert result['success'] == True
        assert result['strategy_type'] == 'conservative'
        assert len(result['selectors']) > 0
        print("[PASS] Conservative strategy direct test passed")

    asyncio.run(run_test())


def test_aggressive_strategy_generation():
    """直接测试激进策略生成逻辑"""
    from src.agents.base import PlanAgent
    import asyncio

    agent = PlanAgent()

    structure = {'main_content_selector': 'body'}
    spec = {
        'targets': [
            {
                'fields': [
                    {'name': 'title', 'description': '文章标题'},
                    {'name': 'price', 'description': '商品价格'},
                ]
            }
        ]
    }
    html = '''
        <html>
            <body>
                <article>
                    <h1>新闻标题</h1>
                    <div class="info">
                        <span class="cost">100</span>
                    </div>
                </article>
            </body>
        </html>
    '''

    async def run_test():
        result = await agent._generate_aggressive(structure, spec, None, html)
        assert result['success'] == True
        assert result['strategy_type'] == 'aggressive'
        assert len(result['selectors']) > 0
        print("[PASS] Aggressive strategy direct test passed")

    asyncio.run(run_test())


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '-s'])
