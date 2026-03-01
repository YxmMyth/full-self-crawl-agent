"""
Tests for ActAgent pagination v2 — 三层策略 + 10+ URL 格式 + 语义化点击分页
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def make_agent():
    """Create an ActAgent instance without full initialization."""
    from src.agents.base import ActAgent
    agent = object.__new__(ActAgent)
    return agent


class TestInferNextPageUrl:
    """Tests for _infer_next_page_url supporting 10+ URL formats."""

    def test_p_param(self):
        """?p=1 → ?p=2"""
        agent = make_agent()
        result = agent._infer_next_page_url('https://example.com/list?p=1', 2)
        assert result is not None
        assert 'p=2' in result

    def test_offset_param(self):
        """?offset=0&size=20 → ?offset=20&size=20"""
        agent = make_agent()
        result = agent._infer_next_page_url('https://example.com/list?offset=0&size=20', 2)
        assert result is not None
        assert 'offset=20' in result
        assert 'size=20' in result

    def test_start_param_default_page_size(self):
        """?start=0 → ?start=20 (default page_size=20)"""
        agent = make_agent()
        result = agent._infer_next_page_url('https://example.com/list?start=0', 2)
        assert result is not None
        assert 'start=20' in result

    def test_path_page_n(self):
        """/page/1 → /page/2"""
        agent = make_agent()
        result = agent._infer_next_page_url('https://example.com/articles/page/1', 2)
        assert result is not None
        assert '/page/2' in result

    def test_page_query_param(self):
        """?page=1 → ?page=2 (backward compatibility)"""
        agent = make_agent()
        result = agent._infer_next_page_url('https://example.com/list?page=1', 2)
        assert result is not None
        assert 'page=2' in result

    def test_unrecognized_url_returns_none(self):
        """Unrecognized URL format should return None."""
        agent = make_agent()
        result = agent._infer_next_page_url('https://example.com/about', 2)
        assert result is None

    def test_pagenum_param(self):
        """?pageNum=1 → ?pageNum=2"""
        agent = make_agent()
        result = agent._infer_next_page_url('https://example.com/list?pageNum=1', 2)
        assert result is not None
        assert 'pageNum=2' in result

    def test_from_param(self):
        """?from=0&limit=10 → ?from=10&limit=10"""
        agent = make_agent()
        result = agent._infer_next_page_url('https://example.com/list?from=0&limit=10', 2)
        assert result is not None
        assert 'from=10' in result


class TestGuessPageSize:
    """Tests for _guess_page_size."""

    def test_size_param(self):
        """Extract page size from ?size=10."""
        agent = make_agent()
        result = agent._guess_page_size({'size': ['10']})
        assert result == 10

    def test_limit_param(self):
        """Extract page size from limit param."""
        agent = make_agent()
        result = agent._guess_page_size({'limit': ['25']})
        assert result == 25

    def test_default_when_missing(self):
        """Return default 20 when no size param."""
        agent = make_agent()
        result = agent._guess_page_size({})
        assert result == 20

    def test_per_page_param(self):
        """Extract page size from per_page param."""
        agent = make_agent()
        result = agent._guess_page_size({'per_page': ['50']})
        assert result == 50


class TestPaginateByClick:
    """Tests for _paginate_by_click selector priority."""

    @pytest.mark.asyncio
    async def test_rel_next_selector_first(self):
        """a[rel="next"] should be tried first."""
        agent = make_agent()

        mock_elem = AsyncMock()
        mock_elem.is_visible = AsyncMock(return_value=True)
        mock_elem.click = AsyncMock()

        mock_page = AsyncMock()
        mock_page.query_selector = AsyncMock(return_value=mock_elem)
        mock_page.wait_for_load_state = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()

        mock_browser = MagicMock()
        mock_browser.page = mock_page

        result = await agent._paginate_by_click(mock_browser)

        assert result is True
        # The first call should be with 'a[rel="next"]'
        first_call_args = mock_page.query_selector.call_args_list[0]
        assert first_call_args[0][0] == 'a[rel="next"]'

    @pytest.mark.asyncio
    async def test_returns_false_when_no_button_found(self):
        """Returns False when no next button is found."""
        agent = make_agent()

        mock_page = AsyncMock()
        mock_page.query_selector = AsyncMock(return_value=None)

        mock_locator = AsyncMock()
        mock_locator.count = AsyncMock(return_value=0)
        mock_page.locator = MagicMock(return_value=mock_locator)

        mock_browser = MagicMock()
        mock_browser.page = mock_page

        result = await agent._paginate_by_click(mock_browser)

        assert result is False

    @pytest.mark.asyncio
    async def test_text_match_fallback(self):
        """Falls back to text matching when selectors fail."""
        agent = make_agent()

        mock_page = AsyncMock()
        mock_page.query_selector = AsyncMock(return_value=None)

        mock_locator = AsyncMock()
        mock_locator.count = AsyncMock(return_value=1)
        mock_locator.first = AsyncMock()
        mock_locator.first.click = AsyncMock()
        mock_page.locator = MagicMock(return_value=mock_locator)
        mock_page.wait_for_load_state = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()

        mock_browser = MagicMock()
        mock_browser.page = mock_page

        result = await agent._paginate_by_click(mock_browser)

        assert result is True
