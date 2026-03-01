"""
SPAHandler 单元测试

覆盖：
- _extract_list_from_json 从各种 JSON 信封结构中提取列表
- start_intercept + 拦截回调（mock response 对象）
- get_best_list_data 返回最优列表
- extract_from_dom 降级 DOM 提取
- execute() 在 API 路径和 DOM 降级路径均可工作
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# 辅助函数测试
# ---------------------------------------------------------------------------

def test_extract_list_from_json_direct_list():
    """顶层就是列表时直接返回"""
    from src.agents.spa_handler import _extract_list_from_json
    data = [{'a': 1}, {'a': 2}]
    assert _extract_list_from_json(data) == data


def test_extract_list_from_json_data_key():
    """data 信封键"""
    from src.agents.spa_handler import _extract_list_from_json
    obj = {'code': 0, 'data': [{'id': 1}, {'id': 2}]}
    result = _extract_list_from_json(obj)
    assert result == [{'id': 1}, {'id': 2}]


def test_extract_list_from_json_items_key():
    """items 信封键"""
    from src.agents.spa_handler import _extract_list_from_json
    obj = {'items': [{'name': 'x'}, {'name': 'y'}], 'total': 2}
    assert _extract_list_from_json(obj) == [{'name': 'x'}, {'name': 'y'}]


def test_extract_list_from_json_results_key():
    """results 信封键"""
    from src.agents.spa_handler import _extract_list_from_json
    obj = {'results': [{'url': 'http://a.com'}]}
    assert _extract_list_from_json(obj) == [{'url': 'http://a.com'}]


def test_extract_list_from_json_nested():
    """嵌套对象"""
    from src.agents.spa_handler import _extract_list_from_json
    obj = {'response': {'data': [{'k': 'v'}]}}
    result = _extract_list_from_json(obj)
    assert result == [{'k': 'v'}]


def test_extract_list_from_json_empty_list_returns_none():
    """空列表应返回 None"""
    from src.agents.spa_handler import _extract_list_from_json
    assert _extract_list_from_json({'data': []}) is None


def test_extract_list_from_json_non_dict_list_returns_none():
    """纯字符串列表应返回 None"""
    from src.agents.spa_handler import _extract_list_from_json
    assert _extract_list_from_json(['a', 'b', 'c']) is None


# ---------------------------------------------------------------------------
# _is_api_url 测试
# ---------------------------------------------------------------------------

def test_is_api_url_positive():
    from src.agents.spa_handler import _is_api_url
    assert _is_api_url('https://example.com/api/products')
    assert _is_api_url('https://example.com/v2/users')
    assert _is_api_url('https://example.com/data.json')
    assert _is_api_url('https://example.com/search?q=test')


def test_is_api_url_negative():
    from src.agents.spa_handler import _is_api_url
    assert not _is_api_url('https://example.com/about')
    assert not _is_api_url('https://example.com/static/main.css')


# ---------------------------------------------------------------------------
# 拦截逻辑测试（mock response）
# ---------------------------------------------------------------------------

@pytest.fixture
def spa_handler():
    from src.agents.spa_handler import SPAHandler
    return SPAHandler()


@pytest.fixture
def mock_page():
    page = MagicMock()
    page.on = MagicMock()  # 记录注册的回调
    page.wait_for_load_state = AsyncMock()
    page.evaluate = AsyncMock(return_value=None)
    return page


async def test_start_intercept_registers_listener(spa_handler, mock_page):
    """start_intercept 应调用 page.on('response', callback)"""
    spa_handler.start_intercept(mock_page)
    mock_page.on.assert_called_once()
    args = mock_page.on.call_args[0]
    assert args[0] == 'response'
    assert callable(args[1])


async def test_intercept_callback_parses_json_api(spa_handler, mock_page):
    """拦截回调应解析 JSON API 响应并存储列表数据"""
    spa_handler.start_intercept(mock_page)

    # 获取注册的回调
    callback = mock_page.on.call_args[0][1]

    # 构造 mock response
    mock_response = MagicMock()
    mock_response.url = 'https://example.com/api/products'
    mock_response.headers = {'content-type': 'application/json'}
    mock_response.text = AsyncMock(return_value=json.dumps(
        {'data': [{'id': 1, 'name': 'Product A'}, {'id': 2, 'name': 'Product B'}]}
    ))

    await callback(mock_response)

    intercepted = spa_handler.get_intercepted()
    assert len(intercepted) == 1
    assert intercepted[0]['url'] == 'https://example.com/api/products'
    assert intercepted[0]['list_data'] == [{'id': 1, 'name': 'Product A'}, {'id': 2, 'name': 'Product B'}]


async def test_intercept_callback_ignores_non_json(spa_handler, mock_page):
    """非 JSON content-type 应被忽略"""
    spa_handler.start_intercept(mock_page)
    callback = mock_page.on.call_args[0][1]

    mock_response = MagicMock()
    mock_response.url = 'https://example.com/api/data'
    mock_response.headers = {'content-type': 'text/html'}
    mock_response.text = AsyncMock(return_value='<html/>')

    await callback(mock_response)
    assert len(spa_handler.get_intercepted()) == 0


async def test_intercept_callback_ignores_non_api_url(spa_handler, mock_page):
    """非 API URL 应被忽略"""
    spa_handler.start_intercept(mock_page)
    callback = mock_page.on.call_args[0][1]

    mock_response = MagicMock()
    mock_response.url = 'https://example.com/about'
    mock_response.headers = {'content-type': 'application/json'}
    mock_response.text = AsyncMock(return_value='{}')

    await callback(mock_response)
    assert len(spa_handler.get_intercepted()) == 0


async def test_intercept_multiple_responses_best_list(spa_handler, mock_page):
    """get_best_list_data 应返回条目最多的列表"""
    spa_handler.start_intercept(mock_page)
    callback = mock_page.on.call_args[0][1]

    # 第一个 API：2 条
    r1 = MagicMock()
    r1.url = 'https://example.com/api/small'
    r1.headers = {'content-type': 'application/json'}
    r1.text = AsyncMock(return_value=json.dumps({'data': [{'a': 1}, {'a': 2}]}))

    # 第二个 API：5 条
    r2 = MagicMock()
    r2.url = 'https://example.com/api/large'
    r2.headers = {'content-type': 'application/json'}
    r2.text = AsyncMock(return_value=json.dumps(
        {'items': [{'x': i} for i in range(5)]}
    ))

    await callback(r1)
    await callback(r2)

    best = spa_handler.get_best_list_data()
    assert best is not None
    assert len(best) == 5


# ---------------------------------------------------------------------------
# extract_from_dom 测试
# ---------------------------------------------------------------------------

def test_extract_from_dom_ul_list(spa_handler):
    """应从 <ul> 中提取文本记录"""
    html = """
    <html><body>
      <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
      </ul>
    </body></html>
    """
    records = spa_handler.extract_from_dom(html)
    assert len(records) >= 3
    texts = [r['text'] for r in records]
    assert 'Item 1' in texts


def test_extract_from_dom_with_selectors(spa_handler):
    """提供字段定义时应按选择器提取"""
    html = """
    <html><body>
      <ul>
        <li><span class="name">Alice</span></li>
        <li><span class="name">Bob</span></li>
        <li><span class="name">Carol</span></li>
      </ul>
    </body></html>
    """
    target_fields = [{'name': 'person_name', 'selector': '.name'}]
    records = spa_handler.extract_from_dom(html, target_fields)
    assert len(records) >= 3
    names = [r.get('person_name') for r in records]
    assert 'Alice' in names


def test_extract_from_dom_no_container(spa_handler):
    """无重复容器时应返回空列表"""
    html = '<html><body><p>Hello world</p></body></html>'
    records = spa_handler.extract_from_dom(html)
    assert records == []


# ---------------------------------------------------------------------------
# execute() 集成测试
# ---------------------------------------------------------------------------

async def test_execute_no_browser(spa_handler):
    """无浏览器时应返回 success=False"""
    result = await spa_handler.execute({})
    assert result['success'] is False
    assert result['records'] == []


async def test_execute_dom_fallback(spa_handler):
    """当 API 拦截没有数据时，应降级到 DOM 提取"""
    html = """
    <html><body>
      <ul>
        <li>A</li><li>B</li><li>C</li>
      </ul>
    </body></html>
    """
    mock_browser = MagicMock()
    mock_browser.page = None  # 无 page 对象，跳过拦截
    mock_browser.get_html = AsyncMock(return_value=html)

    result = await spa_handler.execute({
        'browser': mock_browser,
        'current_url': 'https://example.com/',
    })

    assert result['method'] == 'dom_fallback'
    assert len(result['records']) >= 3


async def test_execute_api_intercept_path(spa_handler):
    """通过拦截 API 数据成功时，method 应为 api_intercept"""
    mock_page = MagicMock()
    mock_page.on = MagicMock()
    mock_page.wait_for_load_state = AsyncMock()

    mock_browser = MagicMock()
    mock_browser.page = mock_page
    mock_browser.get_html = AsyncMock(return_value='<html/>')

    # 在 start_intercept 后注入预设数据（模拟已拦截 API 响应）
    original_start = spa_handler.start_intercept

    def patched_start(page):
        original_start(page)
        spa_handler._intercepted = [{
            'url': 'https://example.com/api/products',
            'raw': {'data': [{'id': 1}, {'id': 2}]},
            'list_data': [{'id': 1}, {'id': 2}],
        }]
        spa_handler._candidate_urls = ['https://example.com/api/products']

    with patch.object(spa_handler, 'start_intercept', side_effect=patched_start):
        result = await spa_handler.execute({
            'browser': mock_browser,
            'current_url': 'https://example.com/',
        })

    assert result['success'] is True
    assert result['records'] == [{'id': 1}, {'id': 2}]
    assert result['method'] == 'api_intercept'


# ---------------------------------------------------------------------------
# AgentPool 集成
# ---------------------------------------------------------------------------

def test_agent_pool_has_spa_handle():
    """AgentPool 应包含 SPA_HANDLE 能力"""
    from src.agents.base import AgentPool, AgentCapability
    pool = AgentPool()
    assert AgentCapability.SPA_HANDLE in pool.agents
    desc = pool.get_capability_description(AgentCapability.SPA_HANDLE)
    assert len(desc) > 0
