"""
基础流程测试 - 验证完整执行流程
"""

import pytest
import asyncio
from pathlib import Path


@pytest.mark.asyncio
async def test_basic_crawl_flow():
    """测试基础爬虫流程"""
    print("\n" + "="*60)
    print("🧪 测试: 基础爬虫流程")
    print("="*60)

    # 测试环境检查
    print("\n✅ 测试环境检查...")

    # 检查依赖
    try:
        import playwright
        print("  ✅ Playwright: 已安装")
    except ImportError:
        print("  ⚠️  Playwright: 未安装（可选）")

    try:
        import httpx
        print("  ✅ HTTPX: 已安装")
    except ImportError:
        assert False, "HTTPX 未安装"

    try:
        import bs4
        print("  ✅ BeautifulSoup: 已安装")
    except ImportError:
        print("  ⚠️  BeautifulSoup: 未安装（可选）")


@pytest.mark.asyncio
async def test_contract_loading():
    """测试契约加载"""
    print("\n" + "="*60)
    print("🧪 测试: 契约加载")
    print("="*60)

    # 从相对路径加载
    spec_path = Path('specs/example_ecommerce.json')

    if not spec_path.exists():
        print(f"  ⚠️  Spec文件不存在: {spec_path}")
        assert False, "测试用例文件缺失"

    print(f"  ✅ Spec文件存在: {spec_path}")

    # 加载并验证
    import json
    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)

    print(f"  ✅ Spec加载成功")
    print(f"  📋 任务名称: {spec.get('task_name', 'N/A')}")
    print(f"  🎯 目标: {spec.get('goal', 'N/A')}")
    print(f"  📦 提取目标数量: {len(spec.get('targets', []))}")

    assert 'version' in spec
    assert spec['freeze'] is True
    assert 'task_id' in spec
    assert 'targets' in spec
    print("  ✅ 契约验证通过")


@pytest.mark.asyncio
async def test_smart_router():
    """测试智能路由"""
    print("\n" + "="*60)
    print("🧪 测试: 智能路由")
    print("="*60)

    # 创建测试HTML
    test_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Page</title></head>
    <body>
        <div class="product">
            <h2 class="product-title">Test Product</h2>
            <span class="product-price">$99.99</span>
        </div>
    </body>
    </html>
    """

    # 从src导入
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

    from src.core.smart_router import SmartRouter

    router = SmartRouter()

    # 测试特征检测
    print("\n  🧪 测试特征检测...")
    from src.core.smart_router import FeatureDetector
    detector = FeatureDetector()
    features = detector.analyze(test_html)

    print(f"    检测结果: {features}")
    assert 'page_type' in features
    assert 'complexity' in features
    print("    ✅ 特征检测通过")

    # 测试路由决策
    print("\n  🧪 测试路由决策...")
    decision = await router.route(
        url='https://test.com',
        goal='Test crawl',
        html=test_html,
        use_llm=False
    )

    print(f"    策略: {decision['strategy']}")
    print(f"    能力: {decision['capabilities']}")
    print(f"    复杂度: {decision['complexity']}")

    assert 'strategy' in decision
    assert 'capabilities' in decision
    assert 'expected_success_rate' in decision
    print("    ✅ 路由决策通过")


@pytest.mark.asyncio
async def test_evidence_storage():
    """测试证据存储"""
    print("\n" + "="*60)
    print("🧪 测试: 证据存储")
    print("="*60)

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

    from src.tools.storage import EvidenceStorage
    import tempfile

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='test_evidence_')
    print(f"  📁 测试目录: {temp_dir}")

    # 测试存储
    storage = EvidenceStorage(base_dir=temp_dir)

    task_dir = storage.create_task_dir('test_task_001')
    print(f"  ✅ 创建任务目录: {task_dir}")

    # 保存HTML
    test_html = "<html><body><h1>Test</h1></body></html>"
    html_path = storage.save_html(test_html, 'test.html')
    print(f"  ✅ 保存HTML: {html_path}")

    # 保存数据
    test_data = [{"title": "Test", "price": 99.99}]
    data_path = storage.save_data(test_data, 'test_data.json')
    print(f"  ✅ 保存数据: {data_path}")

    # 保存日志
    log_path = storage.save_log('Test log message', 'info')
    print(f"  ✅ 保存日志: {log_path}")

    # 检查摘要
    summary = storage.get_task_summary('test_task_001')
    print(f"\n  📊 任务摘要:")
    print(f"    HTML快照: {summary.get('html_snapshots', 0)}")
    print(f"    数据文件: {summary.get('data_files', 0)}")

    assert summary['html_snapshots'] == 1
    assert summary['data_files'] == 1
    print("  ✅ 证据存储测试通过")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 运行基础测试套件")
    print("="*60)

    import asyncio
    asyncio.run(test_basic_crawl_flow())
    asyncio.run(test_contract_loading())
    asyncio.run(test_smart_router())
    asyncio.run(test_evidence_storage())

    print("\n" + "="*60)
    print("✅ 所有测试通过!")
    print("="*60)
