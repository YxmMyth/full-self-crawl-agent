"""
ActAgent 分页单元测试

覆盖：
- _get_next_page_url: page, p, pageNum, pn, pg, offset, /page/N, 末尾数字
- _extract_next_url_from_dom: mock page.evaluate
- _discover_next_url_from_links: mock page.evaluate
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def act_agent():
    from src.agents.base import ActAgent
    return ActAgent()


# ---------------------------------------------------------------------------
# _get_next_page_url — 页码参数
# ---------------------------------------------------------------------------

class TestGetNextPageUrl:
    def test_page_param(self, act_agent):
        url = act_agent._get_next_page_url('http://example.com/list?page=1', 2)
        assert url == 'http://example.com/list?page=2'

    def test_page_param_from_3(self, act_agent):
        url = act_agent._get_next_page_url('http://example.com/list?page=3', 4)
        assert url == 'http://example.com/list?page=4'

    def test_p_param(self, act_agent):
        url = act_agent._get_next_page_url('http://example.com/list?p=1', 2)
        assert url == 'http://example.com/list?p=2'

    def test_pagenum_param(self, act_agent):
        url = act_agent._get_next_page_url('http://example.com/list?pageNum=2', 3)
        assert url == 'http://example.com/list?pageNum=3'

    def test_pn_param(self, act_agent):
        url = act_agent._get_next_page_url('http://example.com/list?pn=1', 2)
        assert url == 'http://example.com/list?pn=2'

    def test_pg_param(self, act_agent):
        url = act_agent._get_next_page_url('http://example.com/list?pg=1', 2)
        assert url == 'http://example.com/list?pg=2'

    def test_offset_param(self, act_agent):
        # next_page=3, current_page=2, current_offset=20 → page_size = 20//(2-1)=20, new_offset=20*2=40
        url = act_agent._get_next_page_url('http://example.com/list?offset=20', 3)
        assert 'offset=40' in url

    def test_start_param(self, act_agent):
        # next_page=3, current_page=2, current_offset=20 → page_size=20, new_offset=40
        url = act_agent._get_next_page_url('http://example.com/list?start=20', 3)
        assert 'start=40' in url

    def test_page_path_pattern(self, act_agent):
        url = act_agent._get_next_page_url('http://example.com/articles/page/1', 2)
        assert url == 'http://example.com/articles/page/2'

    def test_page_path_pattern_deep(self, act_agent):
        url = act_agent._get_next_page_url('http://example.com/news/page/5', 6)
        assert url == 'http://example.com/news/page/6'

    def test_numeric_path_ending(self, act_agent):
        url = act_agent._get_next_page_url('http://example.com/articles/1', 2)
        assert url == 'http://example.com/articles/2'

    def test_no_pagination_returns_none(self, act_agent):
        url = act_agent._get_next_page_url('http://example.com/about', 2)
        assert url is None

    def test_preserves_other_query_params(self, act_agent):
        url = act_agent._get_next_page_url(
            'http://example.com/list?q=python&page=1&sort=date', 2
        )
        assert 'page=2' in url
        assert 'q=python' in url
        assert 'sort=date' in url


# ---------------------------------------------------------------------------
# _extract_next_url_from_dom
# ---------------------------------------------------------------------------

async def test_extract_next_url_from_dom_with_rel_next(act_agent):
    """DOM 中存在 a[rel=next] 时应返回完整 URL"""
    mock_browser = MagicMock()
    mock_browser.page = MagicMock()
    mock_browser.page.evaluate = AsyncMock(return_value='/page/2')
    mock_browser.get_current_url = AsyncMock(return_value='http://example.com/page/1')

    result = await act_agent._extract_next_url_from_dom(mock_browser)
    assert result == 'http://example.com/page/2'


async def test_extract_next_url_from_dom_absolute_href(act_agent):
    """DOM 返回绝对 URL 时应直接使用"""
    mock_browser = MagicMock()
    mock_browser.page = MagicMock()
    mock_browser.page.evaluate = AsyncMock(
        return_value='http://example.com/page/3'
    )
    mock_browser.get_current_url = AsyncMock(return_value='http://example.com/page/2')

    result = await act_agent._extract_next_url_from_dom(mock_browser)
    assert result == 'http://example.com/page/3'


async def test_extract_next_url_from_dom_none(act_agent):
    """DOM 中无 rel=next 时应返回 None"""
    mock_browser = MagicMock()
    mock_browser.page = MagicMock()
    mock_browser.page.evaluate = AsyncMock(return_value=None)
    mock_browser.get_current_url = AsyncMock(return_value='http://example.com/')

    result = await act_agent._extract_next_url_from_dom(mock_browser)
    assert result is None


async def test_extract_next_url_from_dom_evaluate_exception(act_agent):
    """evaluate 抛出异常时应安全返回 None"""
    mock_browser = MagicMock()
    mock_browser.page = MagicMock()
    mock_browser.page.evaluate = AsyncMock(side_effect=Exception("page closed"))

    result = await act_agent._extract_next_url_from_dom(mock_browser)
    assert result is None


# ---------------------------------------------------------------------------
# _discover_next_url_from_links
# ---------------------------------------------------------------------------

async def test_discover_next_url_from_links_page_param(act_agent):
    """从页面链接中发现 page 参数规律并返回下一页 URL"""
    mock_browser = MagicMock()
    mock_browser.page = MagicMock()
    mock_browser.page.evaluate = AsyncMock(return_value=[
        '?page=1', '?page=2', '?page=3'
    ])

    result = await act_agent._discover_next_url_from_links(
        mock_browser, 'http://example.com/list?page=2', 3
    )
    assert result is not None
    assert 'page=3' in result


async def test_discover_next_url_from_links_no_match(act_agent):
    """无匹配链接时应返回 None"""
    mock_browser = MagicMock()
    mock_browser.page = MagicMock()
    mock_browser.page.evaluate = AsyncMock(return_value=[
        '/about', '/contact', '/home'
    ])

    result = await act_agent._discover_next_url_from_links(
        mock_browser, 'http://example.com/', 2
    )
    assert result is None


async def test_discover_next_url_from_links_evaluate_exception(act_agent):
    """evaluate 抛出异常时应安全返回 None"""
    mock_browser = MagicMock()
    mock_browser.page = MagicMock()
    mock_browser.page.evaluate = AsyncMock(side_effect=Exception("timeout"))

    result = await act_agent._discover_next_url_from_links(
        mock_browser, 'http://example.com/', 2
    )
    assert result is None
