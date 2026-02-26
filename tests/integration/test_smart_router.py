"""
智能路由单元测试
"""

import pytest
import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_feature_detector():
    """测试特征检测器"""
    from src.core.smart_router import FeatureDetector
    
    detector = FeatureDetector()
    
    # 简单页面
    simple_html = """
    <html>
    <body>
        <div class="product">
            <h2 class="title">Product</h2>
            <span class="price">$99</span>
        </div>
    </body>
    </html>
    """
    
    features = detector.analyze(simple_html)
    
    assert 'page_type' in features
    assert 'complexity' in features
    assert features['complexity'] in ['simple', 'medium', 'complex']


def test_direct_crawl_strategy():
    """测试直接爬取策略"""
    from src.core.smart_router import SmartRouter
    
    router = SmartRouter()
    
    simple_html = """
    <html><body>
        <div class="item">
            <h2 class="title">Test</h2>
        </div>
    </body></html>
    """
    
    import asyncio
    decision = asyncio.run(router.route(
        url='https://example.com',
        goal='Test crawl',
        html=simple_html,
        use_llm=False
    ))
    
    assert 'strategy' in decision
    assert 'capabilities' in decision
    assert 'expected_success_rate' in decision


def test_routing_stats():
    """测试路由统计"""
    from src.core.smart_router import SmartRouter
    
    router = SmartRouter()
    
    simple_html = "<html><body><div class='test'>Test</div></body></html>"
    
    import asyncio
    for _ in range(3):
        asyncio.run(router.route(
            url='https://example.com',
            goal='Test',
            html=simple_html,
            use_llm=False
        ))
    
    stats = router.get_routing_stats()
    assert stats['total_decisions'] == 3
    assert stats['program_ratio'] >= 0


def test_compose_capabilities():
    """测试能力组合"""
    from src.core.smart_router import compose_capabilities
    
    # 简单任务
    simple_task = {'special_requirements': []}
    capabilities = compose_capabilities(simple_task)
    
    assert 'sense' in capabilities
    assert 'plan' in capabilities
    assert 'act' in capabilities
    assert 'verify' in capabilities


def test_compose_capabilities_with_login():
    """测试带登录需求的能力组合"""
    from src.core.smart_router import compose_capabilities
    
    task = {'special_requirements': ['login']}
    capabilities = compose_capabilities(task)
    
    assert 'handle_login' in capabilities
    assert capabilities.index('handle_login') < capabilities.index('sense')


def test_progressive_explorer():
    """测试渐进式探索"""
    from src.core.smart_router import ProgressiveExplorer
    
    explorer = ProgressiveExplorer()
    
    # 验证策略顺序
    assert len(explorer.STRATEGY_ORDER) > 0
    assert explorer.STRATEGY_ORDER[0][0] == 'direct_crawl'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
