"""
探索智能体 - ExploreAgent（LLM 主导 + 规则回退）

LLM 可用时：
- 根据数据需求描述对链接进行相关性评估和优先级排序
- 生成导航建议（去哪些子页面最可能找到目标数据）

LLM 不可用时：
- 关键词匹配分类链接（detail/list/other）
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import json
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from .base import AgentInterface


class ExploreAgent(AgentInterface):
    """探索智能体 — LLM 主导的有目的站点导航"""

    def __init__(self, llm_client=None):
        super().__init__("ExploreAgent", "explore")
        self.llm_client = llm_client

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        browser = context.get('browser')
        current_url = context.get('current_url', '')
        depth = context.get('depth', 0)
        max_depth = context.get('max_depth', 2)
        base_url = context.get('base_url', current_url)
        frontier = context.get('frontier')
        discover_sitemap = context.get('discover_sitemap', False)
        llm_client = context.get('llm_client') or self.llm_client
        spec = context.get('spec', {})

        if depth >= max_depth:
            return {'success': True, 'links': [], 'message': '已达到最大探索深度'}

        # 1. 提取所有链接（含规范化和静态资源过滤）
        all_links = await self._extract_links(browser, base_url)

        # 2. 过滤相关链接（同域 + 去重）
        relevant_links = self._filter_links(all_links, base_url, context)

        # 3. （可选）发现并解析 sitemap.xml
        sitemap_links: List[str] = []
        if discover_sitemap and browser is not None:
            sitemap_links = await self._discover_sitemap(browser, base_url)
            for lnk in sitemap_links:
                if lnk not in relevant_links:
                    relevant_links.append(lnk)

        # 4. LLM 主导的链接评估 + 分类（有 LLM 且有数据需求描述时）
        description = spec.get('description', '') or spec.get('goal', '')
        navigation_hints = []

        if llm_client and description and relevant_links:
            try:
                ranked = await self._llm_rank_links(
                    llm_client, relevant_links, description, current_url
                )
                relevant_links = ranked.get('ranked_links', relevant_links)
                categorized = ranked.get('categorized', self._categorize_links(relevant_links, context))
                navigation_hints = ranked.get('navigation_hints', [])
            except Exception as e:
                logger.debug(f"LLM 链接评估失败，回退到规则: {e}")
                categorized = self._categorize_links(relevant_links, context)
        else:
            categorized = self._categorize_links(relevant_links, context)

        # 5. 如果提供了 CrawlFrontier，将链接推入队列
        pushed_count = 0
        if frontier is not None:
            pushed_count = frontier.push_many(
                relevant_links,
                depth=depth + 1,
                base_url=base_url,
            )

        return {
            'success': True,
            'links': relevant_links,
            'categorized': categorized,
            'count': len(relevant_links),
            'next_depth': depth + 1,
            'sitemap_links': sitemap_links,
            'pushed_to_frontier': pushed_count,
            'navigation_hints': navigation_hints,
        }

    async def _llm_rank_links(
        self, llm_client, links: List[str], description: str, current_url: str
    ) -> Dict[str, Any]:
        """使用 LLM 根据数据需求对链接进行相关性排序。"""
        # 限制发送给 LLM 的链接数量
        sample = links[:50]

        prompt = f"""你是数据探索代理。当前在页面: {current_url}
用户需求: {description}

以下是页面上发现的链接（共 {len(sample)} 条）：
{json.dumps(sample, ensure_ascii=False)}

请返回严格 JSON（无注释、无 markdown）：
{{
  "ranked_links": ["最可能包含目标数据的链接排在前面，最多 20 条"],
  "categorized": {{
    "high_relevance": ["高相关性链接"],
    "medium_relevance": ["中等相关性"],
    "low_relevance": ["低相关性"]
  }},
  "navigation_hints": ["导航建议，如'点击分类页面可能找到更多数据'"]
}}"""

        response = await llm_client.chat([
            {'role': 'system', 'content': '你是站点导航专家。根据用户数据需求评估链接相关性。只返回 JSON。'},
            {'role': 'user', 'content': prompt},
        ])

        text = response.get('content', '') if isinstance(response, dict) else str(response)
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            raise ValueError("LLM 返回无法解析")

        result = json.loads(json_match.group())

        # 确保 ranked_links 中的链接都来自原始列表
        ranked = [l for l in result.get('ranked_links', []) if l in links]
        # 补充 LLM 未提及的链接
        for l in links:
            if l not in ranked:
                ranked.append(l)

        return {
            'ranked_links': ranked,
            'categorized': result.get('categorized', {}),
            'navigation_hints': result.get('navigation_hints', []),
        }

    async def _extract_links(self, browser, base_url: str = '') -> List[str]:
        """提取页面链接，过滤无效链接和静态资源，并规范化 URL"""
        from urllib.parse import urljoin
        from src.core.crawl_frontier import canonicalize_url, _is_static_resource

        html = await browser.get_html()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        links = []
        seen = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            # 过滤无效链接
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue

            # 转为绝对 URL
            if base_url:
                href = urljoin(base_url, href)

            # 规范化
            href = canonicalize_url(href)

            # 过滤静态资源
            if _is_static_resource(href):
                continue

            if href not in seen:
                seen.add(href)
                links.append(href)

        return links

    def _filter_links(self, links: List[str], base_url: str, context: Dict) -> List[str]:
        """过滤相关链接（同域名）"""
        from urllib.parse import urlparse

        filtered = []
        base_domain = urlparse(base_url).netloc.lower() if base_url else ''

        for link in links:
            try:
                parsed = urlparse(link)
                # 只保留同域名链接
                if base_domain and parsed.netloc.lower() != base_domain:
                    continue
                filtered.append(link)
            except Exception:
                continue

        return filtered

    async def _discover_sitemap(self, browser, base_url: str) -> List[str]:
        """
        尝试发现并解析 sitemap.xml，返回其中的 URL 列表。

        尝试顺序：
        1. /sitemap.xml
        2. /sitemap_index.xml
        """
        from urllib.parse import urlparse, urljoin
        from src.core.crawl_frontier import canonicalize_url, _is_static_resource

        parsed = urlparse(base_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        candidates = [
            urljoin(base, '/sitemap.xml'),
            urljoin(base, '/sitemap_index.xml'),
        ]

        discovered: List[str] = []

        for sitemap_url in candidates:
            try:
                # 使用 page.evaluate fetch（与页面同源，避免 CORS）
                page = getattr(browser, 'page', None)
                if page is not None:
                    xml_text = await page.evaluate(
                        """async (url) => {
                            try {
                                const r = await fetch(url);
                                if (!r.ok) return null;
                                return await r.text();
                            } catch (e) { return null; }
                        }""",
                        sitemap_url,
                    )
                else:
                    xml_text = None

                if not xml_text:
                    continue

                urls = self._parse_sitemap_xml(xml_text)
                for u in urls:
                    u = canonicalize_url(u)
                    if not _is_static_resource(u):
                        discovered.append(u)

                if discovered:
                    break  # 找到一个即可

            except Exception:
                continue

        return discovered

    def _parse_sitemap_xml(self, xml_text: str) -> List[str]:
        """解析 sitemap XML，提取 <loc> 中的 URL"""
        urls: List[str] = []
        # 使用正则提取 <loc> 标签内容（避免引入 xml 解析依赖）
        for match in re.finditer(r'<loc>\s*(https?://[^<\s]+)\s*</loc>', xml_text, re.IGNORECASE):
            urls.append(match.group(1).strip())
        return urls

    def _categorize_links(self, links: List[str], context: Dict) -> Dict[str, List]:
        """分类链接"""
        categories = {
            'detail': [],   # 详情页
            'list': [],     # 列表页
            'other': []     # 其他
        }

        # 简单分类逻辑
        for link in links:
            link_lower = link.lower()
            if any(k in link_lower for k in ['detail', 'item', 'article', 'product', 'news', 'post']):
                categories['detail'].append(link)
            elif any(k in link_lower for k in ['list', 'page', 'category', 'search']):
                categories['list'].append(link)
            else:
                categories['other'].append(link)

        return categories

    def get_description(self) -> str:
        return "探索页面链接，发现新的数据源，支持CrawlFrontier和sitemap.xml"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'browser' in context