import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def act_agent():
    from src.agents.base import ActAgent
    return ActAgent()


@pytest.mark.asyncio
async def test_execute_uses_zip_fallback_when_primary_empty(act_agent):
    browser = MagicMock()
    browser.get_html = AsyncMock(return_value="<html></html>")
    browser.get_current_url = AsyncMock(return_value="http://example.com")

    act_agent._extract_simple = AsyncMock(return_value=[])
    act_agent._extract_from_zip_bundle = AsyncMock(return_value={
        "records": [{"record_type": "code_asset", "asset_type": "html", "content": "<h1>x</h1>"}],
        "metadata": {"attempted": True, "used": True, "errors": []}
    })

    result = await act_agent.execute({
        "browser": browser,
        "selectors": {"title": ".title"},
        "strategy": {},
        "crawl_mode": "single_page"
    })

    assert result["success"] is True
    assert len(result["extracted_data"]) == 1
    assert result["zip_fallback"]["used"] is True


@pytest.mark.asyncio
async def test_execute_keeps_existing_behavior_when_data_is_useful(act_agent):
    browser = MagicMock()
    browser.get_html = AsyncMock(return_value="<html></html>")
    browser.get_current_url = AsyncMock(return_value="http://example.com")

    act_agent._extract_simple = AsyncMock(return_value=[{"title": "t", "price": "$1"}])
    act_agent._extract_from_zip_bundle = AsyncMock()

    result = await act_agent.execute({
        "browser": browser,
        "selectors": {"title": ".title", "price": ".price"},
        "strategy": {},
        "crawl_mode": "single_page"
    })

    assert result["success"] is True
    assert result["extracted_data"] == [{"title": "t", "price": "$1"}]
    assert "zip_fallback" not in result
    act_agent._extract_from_zip_bundle.assert_not_called()


@pytest.mark.asyncio
async def test_zip_fallback_reports_no_link_error(act_agent):
    browser = MagicMock()
    browser.get_html = AsyncMock(return_value="<html><body><a href='/docs'>Docs</a></body></html>")
    browser.get_current_url = AsyncMock(return_value="http://example.com")

    result = await act_agent._extract_from_zip_bundle(browser)

    assert result["records"] == []
    assert result["metadata"]["attempted"] is True
    assert "no_zip_link_found" in result["metadata"]["errors"]


def test_extract_code_records_from_zip(act_agent, tmp_path):
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("index.html", "<html></html>")
        archive.writestr("style.css", "body {}")
        archive.writestr("app.js", "console.log('x')")
        archive.writestr("README.md", "ignore")

    records = act_agent._extract_code_records_from_zip(Path(zip_path), "http://example.com/bundle.zip")

    assert len(records) == 3
    assert {r["asset_type"] for r in records} == {"html", "css", "javascript"}
