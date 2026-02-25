"""
工具层单元测试
"""

import pytest
import sys
import os
import tempfile
import json

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_html_parser():
    """测试HTML解析器"""
    from tools.parser import HTMLParser
    
    html = """
    <html>
    <body>
        <div class="product">
            <h2 class="title">Product 1</h2>
            <span class="price">$99.99</span>
        </div>
        <div class="product">
            <h2 class="title">Product 2</h2>
            <span class="price">$199.99</span>
        </div>
    </body>
    </html>
    """
    
    parser = HTMLParser(html)
    
    # 测试CSS选择器
    products = parser.select('.product')
    assert len(products) == 2
    
    # 测试文本提取
    titles = [p.select_one('.title').get_text() for p in products]
    assert 'Product 1' in titles
    assert 'Product 2' in titles


def test_evidence_storage():
    """测试证据存储"""
    from tools.storage import EvidenceStorage
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = EvidenceStorage(base_dir=tmpdir)
        
        # 创建任务目录
        task_dir = storage.create_task_dir('test_task')
        assert task_dir.exists()
        
        # 保存HTML
        html = "<html><body><h1>Test</h1></body></html>"
        html_path = storage.save_html(html, 'test.html')
        assert os.path.exists(html_path)
        
        # 保存数据
        data = [{'title': 'Test', 'price': 99.99}]
        data_path = storage.save_data(data, 'test.json')
        assert os.path.exists(data_path)
        
        # 保存日志
        log_path = storage.save_log('Test message', 'info')
        assert os.path.exists(log_path)
        
        # 检查摘要
        summary = storage.get_task_summary('test_task')
        assert summary['html_snapshots'] == 1
        assert summary['data_files'] == 1


def test_data_export_json():
    """测试JSON导出"""
    from tools.storage import DataExport
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.json')
        data = [
            {'title': 'Product 1', 'price': 99.99},
            {'title': 'Product 2', 'price': 199.99}
        ]
        
        DataExport.to_json(data, filepath)
        
        # 验证
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        assert len(loaded) == 2
        assert loaded[0]['title'] == 'Product 1'


def test_data_export_csv():
    """测试CSV导出"""
    from tools.storage import DataExport
    import os
    import csv
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.csv')
        data = [
            {'title': 'Product 1', 'price': 99.99},
            {'title': 'Product 2', 'price': 199.99}
        ]
        
        DataExport.to_csv(data, filepath)
        
        # 验证
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2
        assert rows[0]['title'] == 'Product 1'


def test_selector_builder():
    """测试选择器构建器"""
    from tools.parser import SelectorBuilder
    
    # 类选择器
    assert SelectorBuilder.class_name('product') == '.product'
    
    # ID选择器
    assert SelectorBuilder.id_name('main') == '#main'
    
    # 组合选择器
    combined = SelectorBuilder.combine('.product', '.title')
    assert combined == '.product .title'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
