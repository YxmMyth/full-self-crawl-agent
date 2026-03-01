"""
ActAgent 分页改进单元测试

覆盖：
- _get_next_page_url 支持 page/p/pageNum/pageNo 等参数
- _get_next_page_url 支持 offset 参数（配合 size/limit）
- _get_next_page_url 支持 /page/N 路径格式
- _discover_next_url_pattern 兜底策略
- visited_urls 防循环机制（通过 mock）
- DOM next-url 提取（_extract_dom_next_url mock）
- 语义按钮检测（_find_semantic_next_button mock）
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.base import ActAgent


@pytest.fixture
def agent():
    return ActAgent()


# ---------------------------------------------------------------------------
# _get_next_page_url：常见分页参数
# ---------------------------------------------------------------------------

def test_get_next_page_url_page_param(agent):
    url = 'http://example.com/search?q=test&page=1'
    result = agent._get_next_page_url(url, 2)
    assert result is not None
    assert 'page=2' in result


def test_get_next_page_url_p_param(agent):
    url = 'http://example.com/items?p=1&size=20'
    result = agent._get_next_page_url(url, 2)
    assert result is not None
    assert 'p=2' in result


def test_get_next_page_url_pagenum_param(agent):
    url = 'http://example.com/list?pageNum=1'
    result = agent._get_next_page_url(url, 3)
    assert result is not None
    assert 'pageNum=3' in result


def test_get_next_page_url_pageno_param(agent):
    url = 'http://example.com/items?pageNo=2'
    result = agent._get_next_page_url(url, 3)
    assert result is not None
    assert 'pageNo=3' in result


def test_get_next_page_url_offset_param_with_size(agent):
    url = 'http://example.com/api/items?offset=0&size=10'
    result = agent._get_next_page_url(url, 2)
    assert result is not None
    assert 'offset=10' in result


def test_get_next_page_url_offset_param_with_limit(agent):
    url = 'http://example.com/api/data?offset=20&limit=20'
    result = agent._get_next_page_url(url, 3)
    assert result is not None
    assert 'offset=40' in result


def test_get_next_page_url_path_format(agent):
    url = 'http://example.com/articles/page/1'
    result = agent._get_next_page_url(url, 2)
    assert result is not None
    assert '/page/2' in result


def test_get_next_page_url_path_digit(agent):
    url = 'http://example.com/list/1'
    result = agent._get_next_page_url(url, 2)
    assert result is not None
    assert '/list/2' in result


def test_get_next_page_url_no_pattern_returns_none(agent):
    url = 'http://example.com/about'
    result = agent._get_next_page_url(url, 2)
    assert result is None


def test_get_next_page_url_preserves_other_params(agent):
    url = 'http://example.com/search?q=python&page=1&sort=desc'
    result = agent._get_next_page_url(url, 2)
    assert result is not None
    assert 'q=python' in result
    assert 'page=2' in result
    assert 'sort=desc' in result


# ---------------------------------------------------------------------------
# _discover_next_url_pattern: 兜底策略
# ---------------------------------------------------------------------------

def test_discover_next_url_pattern_query_param(agent):
    html = """
    <a href="/search?q=test&page=1">1</a>
    <a href="/search?q=test&page=2">2</a>
    <a href="/search?q=test&page=3">3</a>
    """
    current_url = 'http://example.com/search?q=test&page=1'
    result = agent._discover_next_url_pattern(html, current_url, 4)
    assert result is not None
    assert 'page=4' in result


def test_discover_next_url_pattern_path(agent):
    html = """
    <a href="/articles/page/1">1</a>
    <a href="/articles/page/2">2</a>
    <a href="/articles/page/3">3</a>
    """
    current_url = 'http://example.com/articles/page/1'
    result = agent._discover_next_url_pattern(html, current_url, 4)
    assert result is not None
    assert '/page/4' in result


def test_discover_next_url_pattern_no_match_returns_none(agent):
    html = "<a href='/about'>About</a><a href='/contact'>Contact</a>"
    result = agent._discover_next_url_pattern(html, 'http://example.com/', 2)
    assert result is None


# ---------------------------------------------------------------------------
# _extract_dom_next_url: DOM rel=next 提取（mock）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_dom_next_url_found(agent):
    """当 DOM 中存在 a[rel=next]，应返回绝对 URL"""
    mock_elem = MagicMock()
    mock_elem.get_attribute = AsyncMock(return_value='/page/2')

    browser = MagicMock()
    browser.page = MagicMock()
    browser.page.query_selector = AsyncMock(side_effect=lambda sel: mock_elem if 'next' in sel else None)
    browser.get_current_url = AsyncMock(return_value='http://example.com/page/1')

    result = await agent._extract_dom_next_url(browser)
    assert result == 'http://example.com/page/2'


@pytest.mark.asyncio
async def test_extract_dom_next_url_not_found(agent):
    """DOM 中无 rel=next 链接时应返回 None"""
    browser = MagicMock()
    browser.page = MagicMock()
    browser.page.query_selector = AsyncMock(return_value=None)
    browser.get_current_url = AsyncMock(return_value='http://example.com/')

    result = await agent._extract_dom_next_url(browser)
    assert result is None


# ---------------------------------------------------------------------------
# visited_urls 防循环：_handle_pagination with url type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_pagination_visited_urls_prevents_loop(agent):
    """url 类型分页时，相同 URL 不应被重复访问（防死循环）"""

    async def fake_get_current_url():
        return 'http://example.com/list?page=1'  # 始终返回同一 URL

    browser = MagicMock()
    browser.get_html = AsyncMock(return_value='<html><body></body></html>')
    browser.get_current_url = fake_get_current_url
    browser.navigate = AsyncMock()

    strategy = {
        'pagination_strategy': 'url',
        'max_pages': 5,
        'container_selector': '.item'
    }

    # 调用 _handle_pagination（_extract_dom_next_url 和 _get_next_page_url 全返回 None 时触发兜底）
    with patch.object(agent, '_extract_dom_next_url', new=AsyncMock(return_value=None)), \
         patch.object(agent, '_get_next_page_url', return_value=None), \
         patch.object(agent, '_discover_next_url_pattern', return_value=None), \
         patch.object(agent, '_extract_with_selectors', new=AsyncMock(return_value=[])):
        result = await agent._handle_pagination(
            browser, [{'title': 'item1'}], strategy, {}, None, set()
        )

    # 由于所有 next_url 均为 None，应在第一次迭代就 break
    browser.navigate.assert_not_called()
    assert result == [{'title': 'item1'}]


# ---------------------------------------------------------------------------
# click 分页：语义按钮回退
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_pagination_click_falls_back_to_semantic(agent):
    """click 分页时，若指定 selector 找不到，应尝试语义下一页按钮"""
    mock_next_btn = MagicMock()
    mock_next_btn.click = AsyncMock()

    browser = MagicMock()
    browser.get_html = AsyncMock(return_value='<html><body></body></html>')
    browser.page = MagicMock()
    # 第一次 query_selector（策略 selector）返回 None
    # 第二次（rel=next）返回 None
    # _find_semantic_next_button 返回 mock_next_btn
    browser.page.query_selector = AsyncMock(return_value=None)
    browser.page.wait_for_load_state = AsyncMock()
    browser.page.wait_for_timeout = AsyncMock()

    strategy = {
        'pagination_strategy': 'click',
        'max_pages': 2,
        'pagination_selector': 'a.next',
        'container_selector': '.item'
    }

    with patch.object(agent, '_find_semantic_next_button', new=AsyncMock(return_value=None)), \
         patch.object(agent, '_extract_with_selectors', new=AsyncMock(return_value=[])):
        result = await agent._handle_pagination(
            browser, [{'title': 'item1'}], strategy, {}, None, set()
        )

    # 语义按钮也为 None，应 break 不继续翻页
    assert result == [{'title': 'item1'}]
