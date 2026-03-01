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
    'full_site':   {'max_pages': 100, 'max_depth': 3},
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

        crawl_mode = existing.get('crawl_mode') or patch.get('crawl_mode', 'single_page')

        # 2. 推断 max_pages / max_depth
        anti_bot = features.get('anti_bot_level', 'none')
        factor = _ANTI_BOT_FACTOR.get(anti_bot, 1.0)
        defaults = _DEFAULTS.get(crawl_mode, _DEFAULTS['single_page'])

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
        1. SPA 页面 → single_page（不适合递归爬取）
        2. 有分页 → multi_page
        3. 发现大量链接（≥5）且有重复容器 → full_site
        4. 默认 → single_page
        """
        if features.get('is_spa'):
            return 'single_page'

        if features.get('has_pagination'):
            return 'multi_page'

        container_info = features.get('container_info', {})
        if container_info.get('found') and len(links) >= 5:
            return 'full_site'

        return 'single_page'

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
