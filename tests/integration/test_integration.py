"""
集成测试 - 使用用户提供的示例
"""

import pytest
import sys
import os
import json

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_ecommerce_spec_loading():
    """测试电商示例Spec加载"""
    from src.config.contracts import ContractValidator
    
    spec_path = os.path.join(os.path.dirname(__file__), '..', 'specs', 'example_ecommerce.json')
    
    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)
    
    # 验证Spec
    assert ContractValidator.validate_spec(spec) is True
    
    # 检查关键字段
    assert spec['task_name'] == '电商平台产品列表爬取'
    assert spec['goal'] == '爬取产品列表页面的产品名称、价格和图片链接'
    assert len(spec['targets']) == 1
    assert spec['targets'][0]['name'] == 'products'
    
    print(f"\n✅ 电商示例Spec加载成功")
    print(f"   任务名称: {spec['task_name']}")
    print(f"   目标: {spec['goal']}")
    print(f"   提取字段数: {len(spec['targets'][0]['fields'])}")


def test_news_spec_loading():
    """测试新闻示例Spec加载"""
    from src.config.contracts import ContractValidator
    
    spec_path = os.path.join(os.path.dirname(__file__), '..', 'specs', 'example_news.json')
    
    if not os.path.exists(spec_path):
        pytest.skip("新闻示例文件不存在")
    
    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)
    
    # 验证Spec
    assert ContractValidator.validate_spec(spec) is True
    
    # 检查关键字段
    assert spec['task_name'] == '新闻文章列表爬取'
    assert 'articles' in spec['targets'][0]['name']
    
    print(f"\n✅ 新闻示例Spec加载成功")
    print(f"   任务名称: {spec['task_name']}")
    print(f"   目标: {spec['goal']}")


def test_smart_router_with_ecommerce():
    """测试智能路由与电商示例"""
    from src.core.smart_router import SmartRouter
    
    router = SmartRouter()
    
    # 模拟电商页面HTML
    ecommerce_html = """
    <html>
    <body>
        <div class="product-list">
            <div class="product-item">
                <h3 class="product-title">iPhone 15</h3>
                <span class="product-price">¥5999</span>
                <img class="product-image" src="iphone.jpg">
            </div>
            <div class="product-item">
                <h3 class="product-title">MacBook Pro</h3>
                <span class="product-price">¥12999</span>
                <img class="product-image" src="macbook.jpg">
            </div>
        </div>
    </body>
    </html>
    """
    
    import asyncio
    decision = asyncio.run(router.route(
        url='https://example.com/products',
        goal='爬取产品列表',
        html=ecommerce_html,
        use_llm=False
    ))
    
    assert 'strategy' in decision
    assert 'capabilities' in decision
    assert 'expected_success_rate' in decision
    
    print(f"\n✅ 智能路由决策成功")
    print(f"   策略: {decision['strategy']}")
    print(f"   能力: {', '.join(decision['capabilities'])}")
    print(f"   预期成功率: {decision['expected_success_rate']:.1%}")


def test_evidence_collection():
    """测试证据收集"""
    from src.tools.storage import EvidenceStorage
    import tempfile
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = EvidenceStorage(base_dir=tmpdir)
        
        # 创建任务
        task_dir = storage.create_task_dir('test_ecommerce_001')
        
        # 模拟保存电商数据
        ecommerce_data = [
            {'title': 'iPhone 15', 'price': 5999, 'image_url': 'iphone.jpg'},
            {'title': 'MacBook Pro', 'price': 12999, 'image_url': 'macbook.jpg'},
            {'title': 'AirPods Pro', 'price': 1999, 'image_url': 'airpods.jpg'}
        ]
        
        data_path = storage.save_data(ecommerce_data, 'products.json')
        
        # 保存日志
        storage.save_log('开始爬取电商产品列表', 'info')
        storage.save_log('发现3个产品', 'info')
        
        # 验证
        summary = storage.get_task_summary('test_ecommerce_001')
        assert summary['data_files'] == 1
        
        print(f"\n✅ 证据收集测试成功")
        print(f"   任务目录: {task_dir}")
        print(f"   数据文件数: {summary['data_files']}")
        print(f"   提取产品数: {len(ecommerce_data)}")


def test_completion_gate_ecommerce():
    """测试完成门禁 - 电商场景"""
    from src.core.completion_gate import CompletionGate, GateDecision
    
    gate = CompletionGate()
    decision_maker = GateDecision()
    
    # 模拟电商爬取结果
    state = {
        'html_snapshot': '<html>...</html>',
        'extracted_data': [
            {'title': 'Product 1', 'price': 99.99},
            {'title': 'Product 2', 'price': 199.99},
            {'title': 'Product 3', 'price': 299.99},
            {'title': 'Product 4', 'price': 399.99},
            {'title': 'Product 5', 'price': 499.99}
        ],
        'quality_score': 0.85
    }
    
    spec = {
        'completion_gate': [
            'html_snapshot_exists',
            'execution_success',
            'quality_score >= 0.6',
            'sample_count >= 5'
        ]
    }
    
    # 检查门禁
    gate_passed = gate.check(state, spec)
    
    assert gate_passed is True
    assert len(gate.get_passed_gates()) == 4
    
    # 决策
    final_decision = decision_maker.decide(state, spec)
    assert final_decision == 'complete'
    
    print(f"\n✅ 完成门禁测试成功")
    print(f"   通过门禁: {len(gate.get_passed_gates())}个")
    print(f"   最终决策: {final_decision}")
    print(f"   提取数据: {len(state['extracted_data'])}条")
    print(f"   质量分数: {state['quality_score']:.2f}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 运行集成测试 - 用户示例")
    print("="*60)
    
    test_ecommerce_spec_loading()
    print("\n" + "-"*60)
    
    try:
        test_news_spec_loading()
        print("\n" + "-"*60)
    except pytest.skip.Exception:
        print("   ⚠️  跳过新闻示例测试")
    
    test_smart_router_with_ecommerce()
    print("\n" + "-"*60)
    
    test_evidence_collection()
    print("\n" + "-"*60)
    
    test_completion_gate_ecommerce()
    
    print("\n" + "="*60)
    print("✅ 所有集成测试通过!")
    print("="*60 + "\n")

