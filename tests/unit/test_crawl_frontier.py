"""
CrawlFrontier 单元测试

覆盖：
- push/pop 基本操作
- 已访问 URL 去重
- 优先级排序
- 静态资源过滤
- 深度限制
- 最大页面数限制
- URL 规范化
- 同域名过滤
- url_patterns 过滤
- push_many 批量入队
- get_stats 统计
- mark_visited / is_visited
"""

import pytest
from src.core.crawl_frontier import CrawlFrontier, CrawlItem, canonicalize_url, _is_static_resource


# ---------------------------------------------------------------------------
# canonicalize_url 测试
# ---------------------------------------------------------------------------

def test_canonicalize_removes_fragment():
    url = 'https://example.com/page#section'
    assert canonicalize_url(url) == 'https://example.com/page'


def test_canonicalize_lowercases_domain():
    url = 'https://EXAMPLE.COM/path'
    result = canonicalize_url(url)
    from urllib.parse import urlparse
    assert urlparse(result).netloc == 'example.com'


def test_canonicalize_sorts_query_params():
    url1 = 'https://example.com/search?z=1&a=2'
    url2 = 'https://example.com/search?a=2&z=1'
    assert canonicalize_url(url1) == canonicalize_url(url2)


def test_canonicalize_removes_fragment_preserves_query():
    url = 'https://example.com/search?q=test#anchor'
    result = canonicalize_url(url)
    assert 'q=test' in result
    assert '#' not in result


# ---------------------------------------------------------------------------
# _is_static_resource 测试
# ---------------------------------------------------------------------------

def test_is_static_css():
    assert _is_static_resource('https://example.com/style.css') is True


def test_is_static_image():
    assert _is_static_resource('https://example.com/logo.png') is True
    assert _is_static_resource('https://example.com/photo.jpg') is True


def test_is_static_js():
    assert _is_static_resource('https://example.com/app.js') is True


def test_is_not_static_html():
    assert _is_static_resource('https://example.com/page') is False
    assert _is_static_resource('https://example.com/articles/1') is False


# ---------------------------------------------------------------------------
# 基本 push/pop 测试
# ---------------------------------------------------------------------------

def test_push_and_pop_basic():
    frontier = CrawlFrontier(base_url='https://example.com')
    pushed = frontier.push('https://example.com/page1')
    assert pushed is True
    assert frontier.queue_size() == 1

    item = frontier.pop()
    assert item is not None
    assert item.url == 'https://example.com/page1'
    assert frontier.is_empty()


def test_pop_empty_returns_none():
    frontier = CrawlFrontier()
    assert frontier.pop() is None


def test_priority_ordering():
    """低优先级数值应先出队"""
    frontier = CrawlFrontier(base_url='https://example.com', same_domain_only=False)
    frontier.push('https://example.com/low', priority=10)
    frontier.push('https://example.com/high', priority=1)
    frontier.push('https://example.com/mid', priority=5)

    items = []
    while not frontier.is_empty():
        item = frontier.pop()
        if item:
            items.append(item)

    assert [i.url for i in items] == [
        'https://example.com/high',
        'https://example.com/mid',
        'https://example.com/low',
    ]


# ---------------------------------------------------------------------------
# 去重测试
# ---------------------------------------------------------------------------

def test_no_duplicate_push():
    frontier = CrawlFrontier(base_url='https://example.com')
    assert frontier.push('https://example.com/page') is True
    assert frontier.push('https://example.com/page') is False
    assert frontier.queue_size() == 1


def test_visited_url_not_re_queued():
    frontier = CrawlFrontier(base_url='https://example.com')
    frontier.push('https://example.com/page')
    item = frontier.pop()
    frontier.mark_visited(item.url)

    # 再次 push 同一 URL 应被拒绝
    assert frontier.push('https://example.com/page') is False


def test_canonicalized_url_deduplication():
    """fragment 不同但规范化后相同的 URL 应视为重复"""
    frontier = CrawlFrontier(base_url='https://example.com')
    frontier.push('https://example.com/page#sec1')
    # 规范化后 fragment 移除，与上面相同
    assert frontier.push('https://example.com/page#sec2') is False


# ---------------------------------------------------------------------------
# 过滤测试
# ---------------------------------------------------------------------------

def test_static_resource_filtered():
    frontier = CrawlFrontier(base_url='https://example.com')
    assert frontier.push('https://example.com/style.css') is False
    assert frontier.push('https://example.com/logo.png') is False
    assert frontier.push('https://example.com/app.js') is False
    assert frontier.queue_size() == 0


def test_depth_limit():
    frontier = CrawlFrontier(base_url='https://example.com', max_depth=1)
    assert frontier.push('https://example.com/page', depth=0) is True
    assert frontier.push('https://example.com/sub', depth=1) is True
    assert frontier.push('https://example.com/deep', depth=2) is False


def test_max_pages_limit():
    frontier = CrawlFrontier(base_url='https://example.com', max_pages=2)
    assert frontier.push('https://example.com/p1') is True
    assert frontier.push('https://example.com/p2') is True
    assert frontier.push('https://example.com/p3') is False


def test_same_domain_filter():
    frontier = CrawlFrontier(base_url='https://example.com', same_domain_only=True)
    assert frontier.push('https://other.com/page') is False
    assert frontier.push('https://example.com/page') is True


def test_url_patterns_filter():
    frontier = CrawlFrontier(
        base_url='https://example.com',
        url_patterns=[r'^/products/', r'^/articles/'],
    )
    assert frontier.push('https://example.com/products/1') is True
    assert frontier.push('https://example.com/articles/news') is True
    assert frontier.push('https://example.com/about') is False


# ---------------------------------------------------------------------------
# push_many 测试
# ---------------------------------------------------------------------------

def test_push_many_absolute():
    frontier = CrawlFrontier(base_url='https://example.com')
    pushed = frontier.push_many([
        'https://example.com/a',
        'https://example.com/b',
        'https://other.com/c',  # 跨域被过滤
    ])
    assert pushed == 2
    assert frontier.queue_size() == 2


def test_push_many_relative():
    frontier = CrawlFrontier(base_url='https://example.com')
    pushed = frontier.push_many(['/page1', '/page2'], base_url='https://example.com')
    assert pushed == 2


# ---------------------------------------------------------------------------
# mark_visited / is_visited 测试
# ---------------------------------------------------------------------------

def test_mark_visited_updates_stats():
    frontier = CrawlFrontier(base_url='https://example.com')
    frontier.push('https://example.com/page')
    item = frontier.pop()
    assert frontier.pages_crawled() == 0
    frontier.mark_visited(item.url)
    assert frontier.pages_crawled() == 1
    assert frontier.is_visited('https://example.com/page') is True


def test_mark_visited_canonicalizes():
    frontier = CrawlFrontier()
    frontier.mark_visited('https://example.com/page#frag')
    assert frontier.is_visited('https://example.com/page') is True


# ---------------------------------------------------------------------------
# get_stats 测试
# ---------------------------------------------------------------------------

def test_get_stats():
    frontier = CrawlFrontier(base_url='https://example.com', max_depth=2, max_pages=10)
    frontier.push('https://example.com/a')
    frontier.push('https://example.com/b')
    item = frontier.pop()
    frontier.mark_visited(item.url)

    stats = frontier.get_stats()
    assert stats['queue_size'] == 1
    assert stats['pages_crawled'] == 1
    assert stats['visited_count'] == 1
    assert stats['max_depth'] == 2
    assert stats['max_pages'] == 10


# ---------------------------------------------------------------------------
# reset 测试
# ---------------------------------------------------------------------------

def test_reset():
    frontier = CrawlFrontier(base_url='https://example.com')
    frontier.push('https://example.com/page')
    frontier.reset()
    assert frontier.is_empty()
    assert frontier.visited_count() == 0
    assert frontier.pages_crawled() == 0


# ---------------------------------------------------------------------------
# CrawlItem 测试
# ---------------------------------------------------------------------------

def test_crawl_item_comparison():
    a = CrawlItem(url='https://a.com', priority=1)
    b = CrawlItem(url='https://b.com', priority=2)
    assert a < b


def test_crawl_item_to_dict():
    item = CrawlItem(url='https://example.com/page', depth=1, priority=0)
    d = item.to_dict()
    assert d['url'] == 'https://example.com/page'
    assert d['depth'] == 1


# ---------------------------------------------------------------------------
# StateManager 集成测试
# ---------------------------------------------------------------------------

def test_state_manager_has_crawl_fields():
    """StateManager 初始状态应包含爬取追踪字段"""
    import asyncio
    from src.core.state_manager import StateManager

    sm = StateManager()
    state = sm.create_initial_state_sync('task_1', {})
    assert 'visited_urls' in state
    assert 'queue_size' in state
    assert 'pages_crawled' in state
    assert 'per_url_results' in state
    assert state['visited_urls'] == []
    assert state['pages_crawled'] == 0


def test_state_manager_sync_frontier():
    """sync_frontier 应将 CrawlFrontier 数据同步到状态"""
    from src.core.state_manager import StateManager

    sm = StateManager()
    sm.create_initial_state_sync('task_1', {})

    frontier = CrawlFrontier(base_url='https://example.com')
    frontier.push('https://example.com/a')
    frontier.push('https://example.com/b')
    item = frontier.pop()
    frontier.mark_visited(item.url)

    sm.sync_frontier(frontier)
    state = sm.get_state()
    assert state['pages_crawled'] == 1
    assert state['queue_size'] == 1
    assert len(state['visited_urls']) == 1


def test_state_manager_add_url_result():
    """add_url_result 应记录每个 URL 的结果"""
    from src.core.state_manager import StateManager

    sm = StateManager()
    sm.create_initial_state_sync('task_1', {})
    sm.add_url_result('https://example.com/page', {'records_count': 5, 'success': True})

    state = sm.get_state()
    assert 'https://example.com/page' in state['per_url_results']
    assert state['per_url_results']['https://example.com/page']['records_count'] == 5
