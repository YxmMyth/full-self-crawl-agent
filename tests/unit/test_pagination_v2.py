"""
ActAgent 分页 v2 单元测试
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs
from src.agents.base import ActAgent


@pytest.fixture
def agent():
    return ActAgent()


# ─── _infer_next_page_url ──────────────────────────────────────────────────────

def test_infer_p_param(agent):
    """?p=1 → ?p=2"""
    result = agent._infer_next_page_url('https://example.com/list?p=1', 2)
    assert result is not None
    assert 'p=2' in result


def test_infer_offset_param(agent):
    """?offset=0&size=20 → ?offset=20&size=20"""
    result = agent._infer_next_page_url('https://example.com/list?offset=0&size=20', 2)
    assert result is not None
    assert 'offset=20' in result
    assert 'size=20' in result


def test_infer_start_param_default_size(agent):
    """?start=0 → ?start=20（默认 page_size=20）"""
    result = agent._infer_next_page_url('https://example.com/list?start=0', 2)
    assert result is not None
    assert 'start=20' in result


def test_infer_path_page(agent):
    """/page/1 → /page/2"""
    result = agent._infer_next_page_url('https://example.com/news/page/1', 2)
    assert result is not None
    assert '/page/2' in result


def test_infer_page_query_param(agent):
    """?page=1 → ?page=2（旧格式兼容）"""
    result = agent._infer_next_page_url('https://example.com/list?page=1', 2)
    assert result is not None
    assert 'page=2' in result


def test_infer_unrecognized_url_returns_none(agent):
    """无法识别的 URL 返回 None"""
    result = agent._infer_next_page_url('https://example.com/about', 2)
    assert result is None


# ─── _guess_page_size ──────────────────────────────────────────────────────────

def test_guess_page_size_from_size_param(agent):
    """从 ?size=10 中正确提取 page size"""
    params = parse_qs('size=10')
    assert agent._guess_page_size(params) == 10


def test_guess_page_size_default(agent):
    """无 size 参数时返回默认值 20"""
    assert agent._guess_page_size({}) == 20


# ─── _paginate_by_click ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_paginate_by_click_uses_rel_next_first(agent):
    """_paginate_by_click 优先使用 a[rel="next"] 选择器"""
    browser = MagicMock()
    page = AsyncMock()
    browser.page = page

    # a[rel="next"] 元素存在且可见
    mock_elem = AsyncMock()
    mock_elem.is_visible = AsyncMock(return_value=True)
    mock_elem.click = AsyncMock()

    called_selectors = []

    async def query_selector(selector):
        called_selectors.append(selector)
        if selector == 'a[rel="next"]':
            return mock_elem
        return None

    page.query_selector = query_selector
    page.wait_for_load_state = AsyncMock()
    page.wait_for_timeout = AsyncMock()

    result = await agent._paginate_by_click(browser)

    assert result is True
    assert called_selectors[0] == 'a[rel="next"]'
    mock_elem.click.assert_called_once()


@pytest.mark.asyncio
async def test_paginate_by_click_falls_back_to_text(agent):
    """_paginate_by_click 在所有 CSS 选择器失败后回退到文本匹配"""
    browser = MagicMock()
    page = AsyncMock()
    browser.page = page

    # 所有 CSS 选择器返回 None
    page.query_selector = AsyncMock(return_value=None)

    # 模拟文本定位器找到"下一页"
    mock_locator = AsyncMock()
    mock_locator.count = AsyncMock(return_value=1)
    mock_locator.first = AsyncMock()
    mock_locator.first.click = AsyncMock()

    page.locator = MagicMock(return_value=mock_locator)
    page.wait_for_load_state = AsyncMock()
    page.wait_for_timeout = AsyncMock()

    result = await agent._paginate_by_click(browser)

    assert result is True


@pytest.mark.asyncio
async def test_paginate_by_click_returns_false_when_nothing_found(agent):
    """_paginate_by_click 找不到任何元素时返回 False"""
    browser = MagicMock()
    page = AsyncMock()
    browser.page = page

    page.query_selector = AsyncMock(return_value=None)

    mock_locator = AsyncMock()
    mock_locator.count = AsyncMock(return_value=0)
    page.locator = MagicMock(return_value=mock_locator)

    result = await agent._paginate_by_click(browser)

    assert result is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
