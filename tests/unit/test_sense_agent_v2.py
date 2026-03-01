"""
SenseAgent v2 单元测试

验证：
1. SPA HTML 输入时 _wait_for_spa_render 被调用（mock browser）
2. 输出中只有 page_type，没有 type 字段冲突
3. pagination_type 来自 parser 集成
4. container_info 正确传递到 main_content_selector
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---- 辅助 mock ----

def make_browser(html: str, url: str = 'http://example.com'):
    """构造 mock browser"""
    browser = MagicMock()
    browser.get_html = AsyncMock(return_value=html)
    browser.take_screenshot = AsyncMock(return_value=b'')
    page = MagicMock()
    page.url = url
    page.wait_for_load_state = AsyncMock(return_value=None)
    browser.page = page
    return browser


SPA_HTML = """
<html>
<body>
  <div id="app"></div>
  <script>fetch('/api/data')</script>
</body>
</html>
"""

STATIC_LIST_HTML = """
<html>
<body>
  <ul>
    <li class="item"><a href="/page?p=1">item1</a></li>
    <li class="item"><a href="/page?p=2">item2</a></li>
  </ul>
  <a rel="next" href="/page?p=2">下一页</a>
</body>
</html>
"""

STATIC_PLAIN_HTML = """
<html>
<head><title>Test</title></head>
<body><p>Hello world</p></body>
</html>
"""


# ---- 测试 1：SPA 时 _wait_for_spa_render 被调用 ----

@pytest.mark.asyncio
async def test_spa_wait_called_for_spa_html():
    """SPA HTML 输入时 _wait_for_spa_render 被调用"""
    from src.agents.base import SenseAgent

    agent = SenseAgent()
    browser = make_browser(SPA_HTML)

    with patch.object(agent, '_wait_for_spa_render', new=AsyncMock(return_value=SPA_HTML)) as mock_wait:
        result = await agent.execute({'browser': browser})

    mock_wait.assert_called_once()
    assert result['success'] is True


@pytest.mark.asyncio
async def test_spa_wait_not_called_for_static_html():
    """非 SPA HTML 时 _wait_for_spa_render 不被调用"""
    from src.agents.base import SenseAgent

    agent = SenseAgent()
    browser = make_browser(STATIC_PLAIN_HTML)

    with patch.object(agent, '_wait_for_spa_render', new=AsyncMock(return_value=STATIC_PLAIN_HTML)) as mock_wait:
        result = await agent.execute({'browser': browser})

    mock_wait.assert_not_called()
    assert result['success'] is True


# ---- 测试 2：输出中只有 page_type，没有 type 字段冲突 ----

@pytest.mark.asyncio
async def test_output_has_page_type_not_type():
    """输出 structure 中包含 page_type，不存在旧的 type 字段"""
    from src.agents.base import SenseAgent

    agent = SenseAgent()
    browser = make_browser(STATIC_PLAIN_HTML)

    result = await agent.execute({'browser': browser})

    structure = result.get('structure', {})
    assert 'page_type' in structure, "structure 必须包含 page_type"
    assert 'type' not in structure, "structure 不应包含旧的 type 字段（_analyze_structure 已删除）"


# ---- 测试 3：pagination_type 来自 parser 集成 ----

@pytest.mark.asyncio
async def test_pagination_type_from_parser():
    """pagination_type 正确反映 parser 检测到的分页信息"""
    from src.agents.base import SenseAgent

    agent = SenseAgent()
    # STATIC_LIST_HTML 包含 <a rel="next"> 和 p=\d+ 链接
    browser = make_browser(STATIC_LIST_HTML)

    result = await agent.execute({'browser': browser})

    structure = result.get('structure', {})
    # parser 应能检测到 next_url（<a rel="next">），因此 pagination_type 应为 'url'
    assert structure.get('pagination_type') == 'url', (
        f"pagination_type 应为 url，实际为 {structure.get('pagination_type')}"
    )
    assert 'pagination_next_url' in structure


# ---- 测试 4：container_info 正确传递到 main_content_selector ----

@pytest.mark.asyncio
async def test_container_info_propagated_to_main_content_selector():
    """container_info 中的 container_selector 正确传递到 main_content_selector"""
    from src.agents.base import SenseAgent
    from src.core.smart_router import FeatureDetector

    agent = SenseAgent()
    browser = make_browser(STATIC_PLAIN_HTML)

    fake_container_info = {
        'container_selector': '.article-list',
        'item_selector': '.article-item',
        'estimated_items': 10,
    }

    # Mock FeatureDetector.analyze 返回带 container_info 的 features
    original_analyze = FeatureDetector.analyze

    def mock_analyze(self, html, url=None):
        result = original_analyze(self, html, url)
        result['container_info'] = fake_container_info
        return result

    with patch.object(FeatureDetector, 'analyze', mock_analyze):
        result = await agent.execute({'browser': browser})

    structure = result.get('structure', {})
    assert structure.get('main_content_selector') == '.article-list'
    assert structure.get('item_selector') == '.article-item'
    assert structure.get('estimated_items') == 10


# ---- 测试 5：_determine_pagination_type 逻辑 ----

def test_determine_pagination_type_with_next_url():
    """有 next_url 时返回 url"""
    from src.agents.base import SenseAgent
    agent = SenseAgent()
    result = agent._determine_pagination_type({'next_url': 'http://x.com/page=2', 'has_next': True}, {})
    assert result == 'url'


def test_determine_pagination_type_click():
    """无 next_url 但有 has_next 时返回 click"""
    from src.agents.base import SenseAgent
    agent = SenseAgent()
    result = agent._determine_pagination_type({'next_url': None, 'has_next': True}, {})
    assert result == 'click'


def test_determine_pagination_type_none():
    """无分页时返回 none"""
    from src.agents.base import SenseAgent
    agent = SenseAgent()
    result = agent._determine_pagination_type({'next_url': None, 'has_next': False}, {'has_pagination': False})
    assert result == 'none'


# ---- 测试 6：_wait_for_spa_render 稳定后返回 HTML ----

@pytest.mark.asyncio
async def test_wait_for_spa_render_returns_stable_html():
    """_wait_for_spa_render 在 DOM 稳定后返回 HTML"""
    from src.agents.base import SenseAgent

    agent = SenseAgent()
    browser = make_browser('<html><body>rendered</body></html>')

    # page.wait_for_load_state 已在 make_browser 中 mock
    result = await agent._wait_for_spa_render(browser, '<html></html>', max_wait=2.0)
    assert isinstance(result, str)
    assert len(result) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
