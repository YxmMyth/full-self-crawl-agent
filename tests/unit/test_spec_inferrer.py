"""
SpecInferrer 单元测试

覆盖：
- list 页特征 → crawl_mode=full_site
- detail 页特征 → crawl_mode=single_page
- form 页特征 → crawl_mode=single_page
- SPA 页特征 → crawl_mode=single_page
- has_pagination → crawl_mode=multi_page
- anti_bot_level=high 时 max_pages 应缩小
- 已有 crawl_mode 时不覆盖
- url_patterns 推断（full_site 模式下）
- infer_and_patch 原地更新 spec
"""

import pytest
from src.core.spec_inferrer import SpecInferrer


@pytest.fixture
def inferrer():
    return SpecInferrer()


# ---------------------------------------------------------------------------
# 辅助函数：生成各类页面的 features
# ---------------------------------------------------------------------------

def make_list_features(anti_bot='none', link_count=0):
    return {
        'page_type': 'list',
        'is_spa': False,
        'has_pagination': False,
        'container_info': {'found': True, 'tag': 'ul', 'count': 8, 'similarity': 0.9},
        'anti_bot_level': anti_bot,
    }


def make_detail_features():
    return {
        'page_type': 'detail',
        'is_spa': False,
        'has_pagination': False,
        'container_info': {'found': False, 'tag': None, 'count': 0, 'similarity': 0.0},
        'anti_bot_level': 'none',
    }


def make_form_features():
    return {
        'page_type': 'form',
        'is_spa': False,
        'has_pagination': False,
        'container_info': {'found': False, 'tag': None, 'count': 0, 'similarity': 0.0},
        'anti_bot_level': 'none',
    }


def make_spa_features():
    return {
        'page_type': 'spa',
        'is_spa': True,
        'has_pagination': False,
        'container_info': {'found': False, 'tag': None, 'count': 0, 'similarity': 0.0},
        'anti_bot_level': 'none',
    }


def make_paginated_features():
    return {
        'page_type': 'list',
        'is_spa': False,
        'has_pagination': True,
        'container_info': {'found': True, 'tag': 'ul', 'count': 10, 'similarity': 0.85},
        'anti_bot_level': 'none',
    }


# ---------------------------------------------------------------------------
# crawl_mode 推断
# ---------------------------------------------------------------------------

def test_list_with_sufficient_links_infers_full_site(inferrer):
    """列表页 + 足够链接 → full_site"""
    features = make_list_features()
    links = [f'https://example.com/item/{i}' for i in range(10)]
    patch = inferrer.infer(features, discovered_links=links)
    assert patch.get('crawl_mode') == 'full_site'


def test_list_with_few_links_infers_single_page(inferrer):
    """列表页但链接不足 5 条 → single_page"""
    features = make_list_features()
    links = ['https://example.com/item/1', 'https://example.com/item/2']
    patch = inferrer.infer(features, discovered_links=links)
    assert patch.get('crawl_mode') == 'single_page'


def test_detail_page_infers_single_page(inferrer):
    """详情页 → single_page"""
    patch = inferrer.infer(make_detail_features())
    assert patch.get('crawl_mode') == 'single_page'


def test_form_page_infers_single_page(inferrer):
    """表单页 → single_page"""
    patch = inferrer.infer(make_form_features())
    assert patch.get('crawl_mode') == 'single_page'


def test_spa_page_infers_single_page(inferrer):
    """SPA 页面 → single_page（不适合递归爬取）"""
    links = [f'https://example.com/item/{i}' for i in range(20)]
    patch = inferrer.infer(make_spa_features(), discovered_links=links)
    assert patch.get('crawl_mode') == 'single_page'


def test_paginated_page_infers_multi_page(inferrer):
    """有分页特征 → multi_page"""
    patch = inferrer.infer(make_paginated_features())
    assert patch.get('crawl_mode') == 'multi_page'


# ---------------------------------------------------------------------------
# 已有 crawl_mode 时不覆盖
# ---------------------------------------------------------------------------

def test_existing_crawl_mode_not_overwritten(inferrer):
    """existing_spec 中已有 crawl_mode 时不覆盖"""
    features = make_paginated_features()
    existing = {'crawl_mode': 'single_page'}
    patch = inferrer.infer(features, existing_spec=existing)
    assert 'crawl_mode' not in patch


def test_existing_max_pages_not_overwritten(inferrer):
    """existing_spec 中已有 max_pages 时不覆盖"""
    features = make_paginated_features()
    existing = {'max_pages': 50}
    patch = inferrer.infer(features, existing_spec=existing)
    assert 'max_pages' not in patch


# ---------------------------------------------------------------------------
# max_pages / max_depth 推断
# ---------------------------------------------------------------------------

def test_single_page_defaults(inferrer):
    """single_page 模式默认 max_pages=1, max_depth=0"""
    patch = inferrer.infer(make_detail_features())
    assert patch.get('max_pages') == 1
    assert patch.get('max_depth') == 0


def test_multi_page_defaults(inferrer):
    """multi_page 模式默认 max_pages=20, max_depth=2"""
    patch = inferrer.infer(make_paginated_features())
    assert patch.get('max_pages') == 20
    assert patch.get('max_depth') == 2


def test_full_site_defaults(inferrer):
    """full_site 模式默认 max_pages=100, max_depth=3"""
    features = make_list_features()
    links = [f'https://example.com/item/{i}' for i in range(10)]
    patch = inferrer.infer(features, discovered_links=links)
    assert patch.get('max_pages') == 100
    assert patch.get('max_depth') == 3


def test_high_anti_bot_reduces_max_pages(inferrer):
    """high 反爬时 max_pages 应按 0.2 系数缩减"""
    features = make_list_features(anti_bot='high')
    links = [f'https://example.com/item/{i}' for i in range(10)]
    patch = inferrer.infer(features, discovered_links=links)
    # full_site 100 * 0.2 = 20
    assert patch.get('max_pages') == 20


def test_medium_anti_bot_reduces_max_pages(inferrer):
    """medium 反爬时 max_pages 应按 0.5 系数缩减"""
    features = make_list_features(anti_bot='medium')
    links = [f'https://example.com/item/{i}' for i in range(10)]
    patch = inferrer.infer(features, discovered_links=links)
    # full_site 100 * 0.5 = 50
    assert patch.get('max_pages') == 50


# ---------------------------------------------------------------------------
# url_patterns 推断
# ---------------------------------------------------------------------------

def test_url_patterns_inferred_for_full_site(inferrer):
    """full_site 模式下从链接推断 url_patterns"""
    features = make_list_features()
    links = [f'https://example.com/products/{i}' for i in range(8)]
    # 强制 crawl_mode=full_site（不依赖推断）
    existing = {'crawl_mode': 'full_site'}
    patch = inferrer.infer(features, discovered_links=links, existing_spec=existing)
    assert 'url_patterns' in patch
    patterns = patch['url_patterns']
    assert len(patterns) >= 1
    assert any('/products' in p for p in patterns)


def test_url_patterns_not_inferred_for_single_page(inferrer):
    """single_page 模式下不推断 url_patterns"""
    features = make_detail_features()
    links = [f'https://example.com/item/{i}' for i in range(10)]
    patch = inferrer.infer(features, discovered_links=links)
    assert 'url_patterns' not in patch


def test_url_patterns_not_overwritten_if_existing(inferrer):
    """existing_spec 中已有 url_patterns 时不覆盖"""
    features = make_list_features()
    links = [f'https://example.com/item/{i}' for i in range(10)]
    existing = {'crawl_mode': 'full_site', 'url_patterns': ['/existing']}
    patch = inferrer.infer(features, discovered_links=links, existing_spec=existing)
    assert 'url_patterns' not in patch


# ---------------------------------------------------------------------------
# page_type / is_spa / has_pagination 补充
# ---------------------------------------------------------------------------

def test_page_type_propagated(inferrer):
    """features 中的 page_type 应被补充到 patch"""
    patch = inferrer.infer(make_detail_features())
    assert patch.get('page_type') == 'detail'


def test_page_type_not_overwritten_if_existing(inferrer):
    """existing_spec 中已有 page_type 时不覆盖"""
    patch = inferrer.infer(make_detail_features(), existing_spec={'page_type': 'list'})
    assert 'page_type' not in patch


def test_is_spa_propagated(inferrer):
    """features 中的 is_spa 应被补充到 patch"""
    patch = inferrer.infer(make_spa_features())
    assert patch.get('is_spa') is True


def test_has_pagination_propagated(inferrer):
    """features 中的 has_pagination 应被补充到 patch"""
    patch = inferrer.infer(make_paginated_features())
    assert patch.get('has_pagination') is True


# ---------------------------------------------------------------------------
# infer_and_patch 便捷接口
# ---------------------------------------------------------------------------

def test_infer_and_patch_updates_spec_in_place(inferrer):
    """infer_and_patch 应原地更新并返回 spec"""
    spec = {'task_id': 'test'}
    features = make_paginated_features()
    result = inferrer.infer_and_patch(spec, features)
    assert result is spec  # 同一对象
    assert spec.get('crawl_mode') == 'multi_page'
    assert spec.get('max_pages') == 20


def test_infer_and_patch_does_not_override_existing(inferrer):
    """infer_and_patch 不应覆盖已有字段"""
    spec = {'task_id': 'test', 'crawl_mode': 'single_page', 'max_pages': 5}
    features = make_paginated_features()
    inferrer.infer_and_patch(spec, features)
    assert spec['crawl_mode'] == 'single_page'
    assert spec['max_pages'] == 5
