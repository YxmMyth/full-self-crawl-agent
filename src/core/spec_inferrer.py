"""
SpecInferrer - Spec 自动推断
当 Spec 缺失或不完整时，从已检测的页面特征和发现的链接模式推断 Spec 字段。

推断规则：
- crawl_mode: 根据 is_spa / has_pagination / container_info / 链接数量推断
- page_type: 直接来自 FeatureDetector
- max_pages / max_depth: 根据 crawl_mode 和 anti_bot_level 设置保守默认值
- url_patterns: 从发现的链接中提取公共路径前缀模式
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


# 各模式的默认上限
_DEFAULTS: Dict[str, Dict[str, int]] = {
    'single_page': {'max_pages': 1, 'max_depth': 0},
    'multi_page':  {'max_pages': 20, 'max_depth': 2},
    'full_site':   {'max_pages': 200, 'max_depth': 5},
}

# 反爬等级对应的保守系数（高反爬时减少爬取量）
_ANTI_BOT_FACTOR: Dict[str, float] = {
    'none': 1.0,
    'medium': 0.5,
    'high': 0.2,
}


class SpecInferrer:
    """
    从页面特征和链接模式推断缺失的 Spec 字段。

    用法::

        inferrer = SpecInferrer()
        patch = inferrer.infer(features, discovered_links, existing_spec)
        spec.update(patch)
    """

    def __init__(self, browser: Optional[Any] = None):
        self.browser = browser

    async def infer_missing_fields(
        self,
        start_url: str,
        spec: Dict[str, Any],
        discovered_links: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        兼容旧接口：从当前页面提取特征并补全缺失字段，返回更新后的 spec。
        """
        features: Dict[str, Any] = {'url': start_url}

        if self.browser is not None and hasattr(self.browser, 'get_html'):
            try:
                from .smart_router import FeatureDetector
                html = await self.browser.get_html()
                features = FeatureDetector().analyze(html, start_url)
                features['url'] = start_url
                features['html'] = html[:10000]
            except Exception:
                # 特征提取失败时，使用默认空特征继续推断
                features = {'url': start_url}

        updated = dict(spec or {})
        patch = self.infer(features, discovered_links=discovered_links, existing_spec=updated)
        updated.update(patch)
        return updated

    def infer(
        self,
        features: Dict[str, Any],
        discovered_links: Optional[List[str]] = None,
        existing_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        推断并返回需要补充到 Spec 的字段字典。

        仅填充 existing_spec 中缺失或为空的字段（不覆盖用户已提供的值）。

        Args:
            features: FeatureDetector.analyze() 的返回值
            discovered_links: ExploreAgent 发现的链接列表（可选）
            existing_spec: 当前 Spec 字典（可选）；用于判断哪些字段需补充

        Returns:
            需要合并到 Spec 的字段字典（只含新增/补充字段）
        """
        existing = existing_spec or {}
        links = discovered_links or []
        patch: Dict[str, Any] = {}

        # 1. 推断 crawl_mode
        if not existing.get('crawl_mode'):
            patch['crawl_mode'] = self._infer_crawl_mode(features, links)

        crawl_mode = existing.get('crawl_mode') or patch.get('crawl_mode', 'full_site')

        # 2. 推断 max_pages / max_depth
        anti_bot = features.get('anti_bot_level', 'none')
        factor = _ANTI_BOT_FACTOR.get(anti_bot, 1.0)
        defaults = _DEFAULTS.get(crawl_mode, _DEFAULTS['full_site'])

        if not existing.get('max_pages'):
            patch['max_pages'] = max(1, int(defaults['max_pages'] * factor))

        if 'max_depth' not in existing:
            patch['max_depth'] = defaults['max_depth']

        # 3. 推断 url_patterns（从发现的链接中提取公共路径模式）
        if not existing.get('url_patterns') and crawl_mode == 'full_site' and links:
            patterns = self._infer_url_patterns(links)
            if patterns:
                patch['url_patterns'] = patterns

        # 4. 推断 page_type（便于上层使用）
        if not existing.get('page_type') and features.get('page_type'):
            patch['page_type'] = features['page_type']

        # 5. 推断 is_spa
        if 'is_spa' not in existing and 'is_spa' in features:
            patch['is_spa'] = features['is_spa']

        # 6. 推断 has_pagination
        if 'has_pagination' not in existing and 'has_pagination' in features:
            patch['has_pagination'] = features['has_pagination']

        # 7. 推断 targets（当 spec 未提供时）
        existing_targets = existing.get('targets')
        if not existing_targets:
            inferred_targets = self._infer_targets(features, features.get('html', ''))
            if inferred_targets:
                patch['targets'] = inferred_targets

        return patch

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _infer_crawl_mode(
        self,
        features: Dict[str, Any],
        links: List[str],
    ) -> str:
        """
        根据页面特征和链接数量推断 crawl_mode。

        规则优先级（从高到低）：
        1. 明确的 SPA 页面且无分页 → single_page（不适合递归爬取）
        2. 有分页 → multi_page
        3. 发现大量链接（≥5）且有重复容器 → full_site
        4. 检查 URL 特征（如包含分页参数）→ multi_page
        5. 默认 → full_site
        """
        is_spa = features.get('is_spa', False)
        has_pagination = features.get('has_pagination', False)

        # 如果既是SPA又有分页，这可能是一个具有动态分页的SPA，仍然作为single_page处理
        if is_spa:
            return 'single_page'

        if has_pagination:
            return 'multi_page'

        container_info = features.get('container_info', {})
        has_repeating_containers = container_info.get('found', False)
        significant_link_count = len(links) >= 5

        # 更细致的 full_site 判断：需要同时满足有重复容器和相当数量的链接
        if has_repeating_containers and significant_link_count:
            return 'full_site'

        # 作为补充判断，检查 URL 是否包含分页参数
        url = features.get('url', '')
        pagination_indicators = ['page=', 'p=', 'pg=', 'num=']
        if any(indicator in url.lower() for indicator in pagination_indicators):
            return 'multi_page'

        # 进一步检查是否有明显的类别页面链接
        category_indicators = ['/category/', '/tag/', '/archive/', '/search/', '/page/']
        if links and any(any(indicator in link.lower() for indicator in category_indicators) for link in links):
            return 'full_site'

        return 'full_site'

    def _infer_targets(self, features: Dict[str, Any], html: str = '') -> List[Dict[str, Any]]:
        """在缺失 targets 时自动推断基础提取目标。"""
        page_type = features.get('page_type', 'unknown')
        is_list_like = page_type in ('list', 'static', 'other') or features.get('has_pagination', False)

        fields: List[Dict[str, Any]] = [
            {'name': 'title', 'description': '标题文本', 'required': True},
            {'name': 'link', 'description': '详情链接URL', 'required': True, 'type': 'url'},
            {'name': 'summary', 'description': '摘要或描述文本', 'required': False},
            {'name': 'date', 'description': '发布时间', 'required': False},
        ]

        if 'price' in html.lower() or '￥' in html or '$' in html:
            fields.append({'name': 'price', 'description': '价格信息', 'required': False})

        target_name = 'items' if is_list_like else 'record'
        return [{
            'name': target_name,
            'fields': fields,
        }]

    def _infer_url_patterns(self, links: List[str]) -> List[str]:
        """
        从链接列表中提取公共路径前缀，生成 url_patterns 正则列表。

        算法：
        1. 提取所有路径
        2. 找到出现频率最高的路径前缀（至少 2 个链接共享）
        3. 将前缀转换为正则模式（转义特殊字符）
        4. 最多返回 3 个模式，避免过于宽泛
        """
        if not links:
            return []

        paths = []
        for link in links:
            try:
                parsed = urlparse(link)
                path = parsed.path
                if path and path != '/':
                    paths.append(path)
            except Exception:
                continue

        if not paths:
            return []

        # 统计第一级路径前缀出现次数（取路径第一段）
        prefix_counts: Dict[str, int] = {}
        for path in paths:
            parts = path.strip('/').split('/')
            if parts:
                prefix = '/' + parts[0]
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

        # 只保留出现 ≥2 次的前缀
        common = sorted(
            [(cnt, pfx) for pfx, cnt in prefix_counts.items() if cnt >= 2],
            reverse=True,
        )

        patterns = []
        for _, prefix in common[:3]:
            # 转义特殊正则字符，然后附加 .* 匹配子路径
            escaped = re.escape(prefix)
            patterns.append(f'^{escaped}')

        return patterns

    def infer_and_patch(
        self,
        spec: Dict[str, Any],
        features: Dict[str, Any],
        discovered_links: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        便捷方法：直接将推断结果合并到 spec 并返回更新后的 spec（原地修改）。

        Args:
            spec: 当前 Spec 字典（将被原地更新）
            features: FeatureDetector.analyze() 返回值
            discovered_links: 可选的已发现链接列表

        Returns:
            更新后的 spec 字典
        """
        patch = self.infer(features, discovered_links, existing_spec=spec)
        spec.update(patch)
        return spec
