"""
SpecInferrer - Spec 自动推断（LLM 主导 + 规则回退）

LLM 可用时：
- 理解自然语言数据需求（如"找 HTML 格式的 PPT 模板"）
- 生成站点探索计划（去哪些页面、找什么数据、如何判断找到）
- 推断合理的 crawl_mode / max_pages / targets 等参数

LLM 不可用时（回退）：
- 从页面特征和链接模式推断 crawl_mode / max_pages / max_depth 等
"""

from __future__ import annotations

import json
import re
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


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

_LLM_PROMPT = """\
你是一个自主数据探索代理。用户给了一个 URL 和自然语言数据需求描述，你需要理解需求并生成探索计划。

URL: {url}
用户需求: {description}
{page_context}

请返回 **严格 JSON**（无注释、无 markdown），格式如下：
{{
  "data_description": "用一句话描述用户想要什么数据",
  "exploration_plan": {{
    "strategy": "如何探索这个站点找到数据（搜索/分类浏览/分页遍历/直接提取）",
    "target_pages": ["可能包含目标数据的页面类型，如列表页/搜索结果页/下载页"],
    "navigation_hints": ["导航建议，如点击哪类链接、使用搜索框等"]
  }},
  "success_criteria": "什么情况下算是找到了数据（如找到包含下载链接的列表）",
  "crawl_mode": "single_page 或 multi_page 或 full_site",
  "max_pages": 数字,
  "targets": [
    {{
      "name": "目标数据集名称",
      "fields": [
        {{"name": "字段名", "description": "字段描述", "required": true/false}}
      ]
    }}
  ]
}}
"""


def _extract_first_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("LLM 返回为空")
    decoder = json.JSONDecoder()
    idx = text.find('{')
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = text.find('{', idx + 1)
    raise ValueError(f"LLM 返回无法解析为 JSON: {text[:200]}")


class SpecInferrer:
    """
    从用户需求和页面特征推断 Spec。LLM 可用时使用 LLM 理解需求，否则回退到规则推断。

    用法::

        inferrer = SpecInferrer(browser, llm_client)
        spec = await inferrer.infer_missing_fields(url, spec)
    """

    def __init__(self, browser: Optional[Any] = None, llm_client: Optional[Any] = None):
        self.browser = browser
        self.llm_client = llm_client

    async def infer_missing_fields(
        self,
        start_url: str,
        spec: Dict[str, Any],
        discovered_links: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        主入口：从 URL + spec 推断缺失字段，返回更新后的 spec。
        LLM 可用且 spec 含 description 时走 LLM 路径，否则走规则路径。
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
                features = {'url': start_url}

        updated = dict(spec or {})
        description = updated.get('description', '') or updated.get('goal', '')

        # LLM 路径：有 LLM + 有自然语言描述
        if self.llm_client and description:
            try:
                llm_patch = await self._llm_infer(start_url, description, features)
                # LLM 结果只填充缺失字段
                for k, v in llm_patch.items():
                    if k not in updated or not updated[k]:
                        updated[k] = v
                return updated
            except Exception as e:
                logger.warning(f"LLM 推断失败，回退到规则: {e}")

        # 规则回退路径
        patch = self.infer(features, discovered_links=discovered_links, existing_spec=updated)
        updated.update(patch)
        return updated

    async def _llm_infer(
        self, url: str, description: str, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用 LLM 理解自然语言需求并生成探索计划。"""
        page_context = ""
        if features.get('page_type'):
            page_context = f"页面特征: 类型={features.get('page_type')}, " \
                           f"SPA={features.get('is_spa', False)}, " \
                           f"分页={features.get('has_pagination', False)}, " \
                           f"反爬={features.get('anti_bot_level', 'none')}"

        prompt = _LLM_PROMPT.format(
            url=url,
            description=description,
            page_context=page_context,
        )

        response = await self.llm_client.chat([
            {'role': 'system', 'content': '你是数据探索计划生成器。只返回 JSON，不要任何其他文字。'},
            {'role': 'user', 'content': prompt},
        ])

        text = response.get('content', '') if isinstance(response, dict) else str(response)
        result = _extract_first_json(text)

        patch: Dict[str, Any] = {}
        if result.get('data_description'):
            patch['data_description'] = result['data_description']
        if result.get('exploration_plan'):
            patch['exploration_plan'] = result['exploration_plan']
        if result.get('success_criteria'):
            patch['success_criteria'] = result['success_criteria']
        if result.get('crawl_mode') in ('single_page', 'multi_page', 'full_site'):
            patch['crawl_mode'] = result['crawl_mode']
        if isinstance(result.get('max_pages'), (int, float)) and result['max_pages'] > 0:
            patch['max_pages'] = int(result['max_pages'])
        if isinstance(result.get('targets'), list) and result['targets']:
            patch['targets'] = result['targets']

        # 从 LLM 推断的 crawl_mode 推导 max_depth（如果 LLM 没给）
        if 'crawl_mode' in patch and 'max_depth' not in patch:
            patch['max_depth'] = _DEFAULTS.get(patch['crawl_mode'], _DEFAULTS['full_site'])['max_depth']

        return patch

    def infer(
        self,
        features: Dict[str, Any],
        discovered_links: Optional[List[str]] = None,
        existing_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        规则推断：从页面特征推断缺失的 Spec 字段（LLM 不可用时的回退路径）。

        仅填充 existing_spec 中缺失或为空的字段（不覆盖用户已提供的值）。
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

        # 3. 推断 url_patterns
        if not existing.get('url_patterns') and crawl_mode == 'full_site' and links:
            patterns = self._infer_url_patterns(links)
            if patterns:
                patch['url_patterns'] = patterns

        # 4. 推断 page_type
        if not existing.get('page_type') and features.get('page_type'):
            patch['page_type'] = features['page_type']

        # 5. 推断 is_spa
        if 'is_spa' not in existing and 'is_spa' in features:
            patch['is_spa'] = features['is_spa']

        # 6. 推断 has_pagination
        if 'has_pagination' not in existing and 'has_pagination' in features:
            patch['has_pagination'] = features['has_pagination']

        # 7. 推断 targets
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
        """根据页面特征和链接数量推断 crawl_mode。"""
        is_spa = features.get('is_spa', False)
        has_pagination = features.get('has_pagination', False)

        if is_spa:
            return 'single_page'

        if has_pagination:
            return 'multi_page'

        container_info = features.get('container_info', {})
        has_repeating_containers = container_info.get('found', False)
        significant_link_count = len(links) >= 5

        if has_repeating_containers and significant_link_count:
            return 'full_site'

        url = features.get('url', '')
        pagination_indicators = ['page=', 'p=', 'pg=', 'num=']
        if any(indicator in url.lower() for indicator in pagination_indicators):
            return 'multi_page'

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
        """从链接列表中提取公共路径前缀，生成 url_patterns 正则列表。"""
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

        prefix_counts: Dict[str, int] = {}
        for path in paths:
            parts = path.strip('/').split('/')
            if parts:
                prefix = '/' + parts[0]
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

        common = sorted(
            [(cnt, pfx) for pfx, cnt in prefix_counts.items() if cnt >= 2],
            reverse=True,
        )

        patterns = []
        for _, prefix in common[:3]:
            escaped = re.escape(prefix)
            patterns.append(f'^{escaped}')

        return patterns

    def infer_and_patch(
        self,
        spec: Dict[str, Any],
        features: Dict[str, Any],
        discovered_links: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """便捷方法：直接将推断结果合并到 spec 并返回更新后的 spec（原地修改）。"""
        patch = self.infer(features, discovered_links, existing_spec=spec)
        spec.update(patch)
        return spec
