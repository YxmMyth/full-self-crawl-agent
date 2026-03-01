"""
目标驱动模式测试

验证 Spec 字段可以没有 selector，只需要 description
"""

import pytest
import tempfile
import os
from pathlib import Path
import io

from src.config.loader import SpecLoader
from src.agents.base import PlanAgent


class TestGoalDrivenSpec:
    """目标驱动模式测试"""

    def test_spec_with_description_only(self):
        """测试只有 description 没有 selector 的字段可以正常加载"""
        loader = SpecLoader("specs")

        spec_content = """
task_id: "test_001"
task_name: "Test Goal Driven"
targets:
  - name: "articles"
    fields:
      - name: "title"
        description: "article title"
        required: true
      - name: "author"
        description: "author name"
        examples: ["John", "Jane"]
"""
        # 写入临时文件，使用 utf-8 编码
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            f.write(spec_content)
            temp_path = f.name

        try:
            spec = loader.load_spec(temp_path, validate=True)
            assert spec['task_name'] == "Test Goal Driven"
            fields = spec['targets'][0]['fields']
            assert len(fields) == 2
            assert fields[0]['name'] == 'title'
            assert fields[0]['description'] == 'article title'
            assert 'selector' not in fields[0]  # 没有 selector
        finally:
            os.unlink(temp_path)

    def test_spec_with_both_selector_and_description(self):
        """测试同时有 selector 和 description 的字段"""
        loader = SpecLoader("specs")

        spec_content = """
task_id: "test_002"
task_name: "Test Hybrid Mode"
targets:
  - name: "products"
    fields:
      - name: "title"
        selector: ".product-title"
        description: "product title"
        required: true
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            f.write(spec_content)
            temp_path = f.name

        try:
            spec = loader.load_spec(temp_path, validate=True)
            field = spec['targets'][0]['fields'][0]
            assert field['selector'] == '.product-title'
            assert field['description'] == 'product title'
        finally:
            os.unlink(temp_path)

    def test_spec_missing_both_selector_and_description(self):
        """测试缺少 selector 和 description 的字段应该报错"""
        loader = SpecLoader("specs")

        spec_content = """
task_id: "test_003"
task_name: "Test Missing Fields"
targets:
  - name: "items"
    fields:
      - name: "title"
        required: true
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(spec_content)
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                loader.load_spec(temp_path, validate=True)
            assert "requires either 'selector' or 'description'" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_fallback_strategy_with_missing_selectors(self):
        """测试降级策略能为无 selector 字段生成默认选择器"""
        agent = PlanAgent()

        spec = {
            'targets': [{
                'name': 'articles',
                'fields': [
                    {'name': 'title', 'description': '文章标题'},
                    {'name': 'author', 'description': '作者'},
                    {'name': 'date', 'selector': '.publish-date', 'description': '发布日期'},
                ]
            }]
        }

        structure = {'main_content_selector': 'article'}

        result = agent._fallback_strategy(structure, spec)

        # 验证所有字段都有选择器
        assert 'title' in result['selectors']
        assert 'author' in result['selectors']
        assert 'date' in result['selectors']
        assert result['selectors']['date'] == '.publish-date'

        # 验证选择器来源信息
        assert result['selector_info']['date'] == 'from_spec'
        assert result['selector_info']['title'] == 'generated'
        assert result['selector_info']['author'] == 'generated'

    def test_default_selector_generation(self):
        """测试默认选择器生成逻辑"""
        agent = PlanAgent()

        # 测试常见字段
        assert 'h1, h2, h3' in agent._generate_default_selector('title', '')
        assert 'price' in agent._generate_default_selector('price', '')
        assert 'author' in agent._generate_default_selector('author', '')
        assert 'date' in agent._generate_default_selector('date', '')

        # 测试基于描述的推断
        assert 'h1' in agent._generate_default_selector('field1', '这是文章的标题')
        assert 'author' in agent._generate_default_selector('field2', '作者信息')

        # 测试通用选择器
        selector = agent._generate_default_selector('custom_field', '自定义字段')
        assert 'custom-field' in selector.lower()

    def test_semantic_search_in_html_context(self):
        """测试 HTML 上下文中的语义搜索功能"""
        agent = PlanAgent()

        html = '''
        <html>
        <body>
            <article>
                <h1 class="article-title">Deep Learning for NLP</h1>
                <div class="author-name">John Smith</div>
                <span class="publish-date">2024-01-15</span>
            </article>
        </body>
        </html>
        '''

        targets = [{
            'name': 'articles',
            'fields': [
                {'name': 'title', 'description': 'article title'},
                {'name': 'author', 'description': 'author name'},
            ]
        }]

        context = agent._extract_html_context(html, targets)

        # 验证语义搜索找到相关元素
        assert 'title' in context.lower()
        assert 'Deep Learning' in context

    def test_arxiv_spec_loading(self):
        """测试 arXiv Spec 加载（目标驱动模式）"""
        loader = SpecLoader("specs")

        # 加载修改后的 arXiv spec
        spec_path = Path("specs/test_sites/site_05_arxiv.yaml")
        if spec_path.exists():
            spec = loader.load_spec(spec_path, validate=True)

            # 验证目标驱动字段
            fields = spec['targets'][0]['fields']
            title_field = next(f for f in fields if f['name'] == 'title')
            authors_field = next(f for f in fields if f['name'] == 'authors')

            # title 和 authors 字段没有 selector
            assert 'selector' not in title_field or title_field.get('selector') is None
            assert 'selector' not in authors_field or authors_field.get('selector') is None

            # 但有 description
            assert 'description' in title_field
            assert 'description' in authors_field