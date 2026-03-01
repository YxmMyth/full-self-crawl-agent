"""
ExploreAgent v2 单元测试

覆盖：
- 基本链接提取和过滤
- 静态资源过滤
- URL 规范化（canonicalize）
- CrawlFrontier 集成（push_many 被调用）
- sitemap.xml 发现和解析（_parse_sitemap_xml）
- 深度限制
- 分类链接
- 返回键结构完整
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def explore_agent():
    from src.agents.base import ExploreAgent
    return ExploreAgent()


@pytest.fixture
def mock_browser():
    browser = MagicMock()
    browser.page = None
    return browser


def make_html(links):
    """生成包含指定链接的简单 HTML"""
    hrefs = '\n'.join(f'<a href="{link}">link</a>' for link in links)
    return f'<html><body>{hrefs}</body></html>'


# ---------------------------------------------------------------------------
# 基本链接提取
# ---------------------------------------------------------------------------

async def test_execute_returns_required_keys(explore_agent, mock_browser):
    """execute 应返回规定的顶层键"""
    mock_browser.get_html = AsyncMock(return_value='<html><body></body></html>')
    result = await explore_agent.execute({
        'browser': mock_browser,
        'current_url': 'https://example.com/',
        'base_url': 'https://example.com/',
    })
    required = {'success', 'links', 'categorized', 'count', 'next_depth',
                'sitemap_links', 'pushed_to_frontier'}
    assert required.issubset(result.keys())


async def test_execute_extracts_same_domain_links(explore_agent, mock_browser):
    """应提取同域名链接"""
    html = make_html([
        'https://example.com/page1',
        'https://example.com/page2',
        'https://other.com/external',  # 跨域，应被过滤
    ])
    mock_browser.get_html = AsyncMock(return_value=html)
    result = await explore_agent.execute({
        'browser': mock_browser,
        'current_url': 'https://example.com/',
        'base_url': 'https://example.com/',
    })
    assert 'https://example.com/page1' in result['links']
    assert 'https://example.com/page2' in result['links']
    # 跨域链接不应出现
    from urllib.parse import urlparse
    result_domains = {urlparse(lnk).netloc for lnk in result['links']}
    assert 'other.com' not in result_domains


async def test_execute_filters_static_resources(explore_agent, mock_browser):
    """静态资源链接应被过滤"""
    html = make_html([
        'https://example.com/page1',
        'https://example.com/style.css',
        'https://example.com/logo.png',
        'https://example.com/app.js',
    ])
    mock_browser.get_html = AsyncMock(return_value=html)
    result = await explore_agent.execute({
        'browser': mock_browser,
        'current_url': 'https://example.com/',
        'base_url': 'https://example.com/',
    })
    assert 'https://example.com/page1' in result['links']
    for static in ['style.css', 'logo.png', 'app.js']:
        assert not any(static in lnk for lnk in result['links'])


async def test_execute_filters_invalid_links(explore_agent, mock_browser):
    """javascript:、mailto:、#anchor 应被过滤"""
    html = (
        '<html><body>'
        '<a href="javascript:void(0)">JS</a>'
        '<a href="mailto:test@example.com">Mail</a>'
        '<a href="#section">Anchor</a>'
        '<a href="/valid">Valid</a>'
        '</body></html>'
    )
    mock_browser.get_html = AsyncMock(return_value=html)
    result = await explore_agent.execute({
        'browser': mock_browser,
        'current_url': 'https://example.com/',
        'base_url': 'https://example.com/',
    })
    assert len(result['links']) == 1
    assert result['links'][0].endswith('/valid')


async def test_execute_resolves_relative_links(explore_agent, mock_browser):
    """相对路径应被解析为绝对路径"""
    html = make_html(['/products', '/about', 'contact'])
    mock_browser.get_html = AsyncMock(return_value=html)
    result = await explore_agent.execute({
        'browser': mock_browser,
        'current_url': 'https://example.com/',
        'base_url': 'https://example.com/',
    })
    for link in result['links']:
        assert link.startswith('https://')


# ---------------------------------------------------------------------------
# 深度限制
# ---------------------------------------------------------------------------

async def test_depth_limit_stops_exploration(explore_agent, mock_browser):
    """达到 max_depth 时应返回空链接"""
    result = await explore_agent.execute({
        'browser': mock_browser,
        'current_url': 'https://example.com/',
        'depth': 3,
        'max_depth': 3,
    })
    assert result['links'] == []
    assert '最大探索深度' in result.get('message', '')


# ---------------------------------------------------------------------------
# CrawlFrontier 集成
# ---------------------------------------------------------------------------

async def test_execute_pushes_to_frontier(explore_agent, mock_browser):
    """提供 frontier 时应将链接推入队列"""
    from src.core.crawl_frontier import CrawlFrontier

    html = make_html([
        'https://example.com/page1',
        'https://example.com/page2',
    ])
    mock_browser.get_html = AsyncMock(return_value=html)

    frontier = CrawlFrontier(base_url='https://example.com', max_pages=50)
    result = await explore_agent.execute({
        'browser': mock_browser,
        'current_url': 'https://example.com/',
        'base_url': 'https://example.com/',
        'frontier': frontier,
    })

    assert result['pushed_to_frontier'] == 2
    assert frontier.queue_size() == 2


async def test_execute_without_frontier_pushed_zero(explore_agent, mock_browser):
    """未提供 frontier 时 pushed_to_frontier 应为 0"""
    mock_browser.get_html = AsyncMock(return_value='<html><body></body></html>')
    result = await explore_agent.execute({
        'browser': mock_browser,
        'current_url': 'https://example.com/',
    })
    assert result['pushed_to_frontier'] == 0


# ---------------------------------------------------------------------------
# sitemap.xml 解析
# ---------------------------------------------------------------------------

def test_parse_sitemap_xml(explore_agent):
    """_parse_sitemap_xml 应从 <loc> 标签提取 URL"""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://example.com/page1</loc></url>
        <url><loc>https://example.com/page2</loc></url>
        <url><loc>https://example.com/page3</loc></url>
    </urlset>"""
    urls = explore_agent._parse_sitemap_xml(xml)
    assert len(urls) == 3
    assert 'https://example.com/page1' in urls
    assert 'https://example.com/page3' in urls


def test_parse_sitemap_xml_empty(explore_agent):
    """无 <loc> 标签时应返回空列表"""
    assert explore_agent._parse_sitemap_xml('<xml/>') == []


# ---------------------------------------------------------------------------
# 链接分类
# ---------------------------------------------------------------------------

def test_categorize_detail_links(explore_agent):
    links = [
        'https://example.com/product/1',
        'https://example.com/article/news',
        'https://example.com/about',
    ]
    result = explore_agent._categorize_links(links, {})
    assert 'https://example.com/product/1' in result['detail']
    assert 'https://example.com/article/news' in result['detail']
    assert 'https://example.com/about' in result['other']


def test_categorize_list_links(explore_agent):
    links = ['https://example.com/category/books', 'https://example.com/list/items']
    result = explore_agent._categorize_links(links, {})
    assert 'https://example.com/category/books' in result['list']


# ---------------------------------------------------------------------------
# Spec 契约测试
# ---------------------------------------------------------------------------

def test_spec_contract_crawl_mode():
    """ContractFactory.create_spec 应支持 crawl_mode 参数"""
    from src.config.contracts import ContractFactory
    spec = ContractFactory.create_spec(
        goal='爬取所有产品',
        target_url='https://example.com',
        crawl_mode='full_site',
        max_pages=50,
        max_depth=3,
        url_patterns=[r'^/products/'],
    )
    assert spec['crawl_mode'] == 'full_site'
    assert spec['max_pages'] == 50
    assert spec['max_depth'] == 3
    assert spec['url_patterns'] == [r'^/products/']


def test_spec_contract_default_single_page():
    """默认应为 single_page 模式（向后兼容）"""
    from src.config.contracts import ContractFactory
    spec = ContractFactory.create_spec(goal='test', target_url='https://example.com')
    assert spec['crawl_mode'] == 'single_page'
    assert spec['max_pages'] == 1
    assert spec['max_depth'] == 0
