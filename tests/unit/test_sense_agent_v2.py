"""
SenseAgent v2 单元测试

覆盖：
- execute() 返回必需的顶层键
- structure 含 page_type 不含 type
- SPA 智能等待被调用（mock）
- pagination_type 字段被填充
- main_content_selector 从 container_info 派生
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_browser():
    browser = MagicMock()
    browser.page = MagicMock()
    browser.page.wait_for_load_state = AsyncMock()
    browser.page.wait_for_timeout = AsyncMock()
    browser.get_html = AsyncMock(return_value='<html><body><p>hello world</p></body></html>')
    browser.take_screenshot = AsyncMock(return_value=None)
    browser.get_current_url = AsyncMock(return_value='http://example.com/')
    return browser


@pytest.fixture
def sense_agent():
    from src.agents.base import SenseAgent
    return SenseAgent()


# ---------------------------------------------------------------------------
# 返回契约
# ---------------------------------------------------------------------------

async def test_execute_returns_required_keys(sense_agent, mock_browser):
    """execute() 必须包含全部规定的顶层键"""
    result = await sense_agent.execute({'browser': mock_browser})
    required = {'success', 'structure', 'features', 'anti_bot_detected',
                'anti_bot_info', 'html_snapshot', 'screenshot'}
    assert required.issubset(result.keys()), f"缺少键: {required - result.keys()}"


async def test_execute_no_browser_returns_failure(sense_agent):
    """无浏览器时应返回 success=False 并包含完整键"""
    result = await sense_agent.execute({})
    assert result['success'] is False
    required = {'success', 'structure', 'features', 'anti_bot_detected',
                'anti_bot_info', 'html_snapshot', 'screenshot'}
    assert required.issubset(result.keys())


# ---------------------------------------------------------------------------
# structure 字段校验
# ---------------------------------------------------------------------------

async def test_structure_has_page_type_not_type(sense_agent, mock_browser):
    """structure 应包含 'page_type' 但不含旧的 'type' 字段"""
    result = await sense_agent.execute({'browser': mock_browser})
    structure = result['structure']
    assert 'page_type' in structure, "structure 缺少 page_type 字段"
    assert 'type' not in structure, "structure 不应包含旧的 'type' 字段"


async def test_structure_pagination_type_present(sense_agent, mock_browser):
    """structure 中 pagination_type 字段应存在"""
    result = await sense_agent.execute({'browser': mock_browser})
    assert 'pagination_type' in result['structure']


async def test_pagination_type_click_for_rel_next(sense_agent, mock_browser):
    """rel=next 链接应使 pagination_type 变为 'click'"""
    html = '<html><body><a rel="next" href="/page/2">下一页</a></body></html>'
    mock_browser.get_html = AsyncMock(return_value=html)
    result = await sense_agent.execute({'browser': mock_browser})
    assert result['structure']['pagination_type'] == 'click'
    assert result['structure']['pagination_next_url'] is not None


async def test_pagination_type_none_for_simple_page(sense_agent, mock_browser):
    """无分页特征的页面 pagination_type 应为 'none'"""
    html = '<html><body><p>Simple content without pagination.</p></body></html>'
    mock_browser.get_html = AsyncMock(return_value=html)
    result = await sense_agent.execute({'browser': mock_browser})
    assert result['structure']['pagination_type'] == 'none'


# ---------------------------------------------------------------------------
# SPA 智能等待
# ---------------------------------------------------------------------------

async def test_spa_wait_called_for_spa_page(sense_agent, mock_browser):
    """检测到 SPA 时应调用 _wait_for_spa_render"""
    spa_html = """
    <html>
    <head>
      <script src="/js/vendor.chunk.js"></script>
      <script src="/js/app.chunk.js"></script>
    </head>
    <body><div id="root"></div></body>
    </html>
    """
    mock_browser.get_html = AsyncMock(return_value=spa_html)

    with patch.object(sense_agent, '_wait_for_spa_render',
                      AsyncMock(return_value=spa_html)) as mock_wait:
        await sense_agent.execute({'browser': mock_browser})
        mock_wait.assert_called_once()


async def test_spa_wait_not_called_for_non_spa(sense_agent, mock_browser):
    """非 SPA 页面不应调用 _wait_for_spa_render"""
    html = '<html><body><article><h1>Title</h1><p>A</p><p>B</p><p>C</p></article></body></html>'
    mock_browser.get_html = AsyncMock(return_value=html)

    with patch.object(sense_agent, '_wait_for_spa_render',
                      AsyncMock()) as mock_wait:
        await sense_agent.execute({'browser': mock_browser})
        mock_wait.assert_not_called()


# ---------------------------------------------------------------------------
# main_content_selector 从 container_info 派生
# ---------------------------------------------------------------------------

async def test_main_content_selector_from_list_page(sense_agent, mock_browser):
    """列表页的 main_content_selector 应从 container_info 派生"""
    list_html = """
    <html><body>
      <ul>
        <li><span>Item 1</span><span>Desc 1</span></li>
        <li><span>Item 2</span><span>Desc 2</span></li>
        <li><span>Item 3</span><span>Desc 3</span></li>
        <li><span>Item 4</span><span>Desc 4</span></li>
        <li><span>Item 5</span><span>Desc 5</span></li>
      </ul>
    </body></html>
    """
    mock_browser.get_html = AsyncMock(return_value=list_html)
    result = await sense_agent.execute({'browser': mock_browser})
    assert result['structure']['main_content_selector'] is not None
    assert result['structure']['estimated_items'] > 0


async def test_main_content_selector_none_for_simple_page(sense_agent, mock_browser):
    """无重复容器的简单页面 main_content_selector 应为 None"""
    html = '<html><body><p>Simple single paragraph page.</p></body></html>'
    mock_browser.get_html = AsyncMock(return_value=html)
    result = await sense_agent.execute({'browser': mock_browser})
    assert result['structure']['main_content_selector'] is None
    assert result['structure']['estimated_items'] == 0


# ---------------------------------------------------------------------------
# _determine_pagination_type
# ---------------------------------------------------------------------------

def test_determine_pagination_type_with_next_url(sense_agent):
    """parser 返回 next_url 时应输出 'click'"""
    result = sense_agent._determine_pagination_type(
        {'next_url': 'http://example.com/page/2', 'has_next': True},
        {'has_pagination': False}
    )
    assert result == 'click'


def test_determine_pagination_type_has_next(sense_agent):
    """parser 返回 has_next=True 时应输出 'click'"""
    result = sense_agent._determine_pagination_type(
        {'next_url': None, 'has_next': True},
        {'has_pagination': False}
    )
    assert result == 'click'


def test_determine_pagination_type_feature_detector(sense_agent):
    """FeatureDetector has_pagination=True 且无 next_url 时应输出 'url'"""
    result = sense_agent._determine_pagination_type(
        {'next_url': None, 'has_next': False},
        {'has_pagination': True}
    )
    assert result == 'url'


def test_determine_pagination_type_none(sense_agent):
    """无任何分页信号时应输出 'none'"""
    result = sense_agent._determine_pagination_type(
        {'next_url': None, 'has_next': False},
        {'has_pagination': False}
    )
    assert result == 'none'
