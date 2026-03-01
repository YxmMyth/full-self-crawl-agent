"""
SPAHandler - SPA页面处理智能体
处理单页面应用（SPA），通过以下方式提取数据：
1. 拦截 JSON API 响应（via Playwright response 事件）
2. 发现候选 API
3. 从页面上下文中直接 fetch API
4. 从常见 JSON 信封键（data/items/results/list/records）提取列表数组
5. （可选）通过 LLM 将字段映射到 Spec 目标
6. 当 API 不可用时降级到渲染 DOM 提取
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


# JSON 信封中常见的列表键，按优先级排列
_LIST_ENVELOPE_KEYS = [
    'data', 'items', 'results', 'list', 'records',
    'rows', 'content', 'entries', 'products', 'articles',
]


def _extract_list_from_json(obj: Any, depth: int = 0) -> Optional[List[Dict]]:
    """
    从 JSON 对象中提取列表数组。

    按优先级顺序检查常见信封键；若对象本身即是非空列表则直接返回。
    仅返回元素为 dict 的列表（避免返回纯字符串/数字列表）。
    """
    if depth > 3:
        return None

    if isinstance(obj, list):
        if obj and all(isinstance(item, dict) for item in obj):
            return obj
        return None

    if isinstance(obj, dict):
        # 优先检查已知信封键
        for key in _LIST_ENVELOPE_KEYS:
            value = obj.get(key)
            if value is not None:
                result = _extract_list_from_json(value, depth + 1)
                if result:
                    return result

        # 回退：递归搜索所有值
        for value in obj.values():
            result = _extract_list_from_json(value, depth + 1)
            if result:
                return result

    return None


def _is_json_content_type(content_type: str) -> bool:
    """判断响应是否为 JSON 类型"""
    return 'json' in content_type.lower() or 'javascript' in content_type.lower()


def _is_api_url(url: str) -> bool:
    """
    判断 URL 是否像是 API 端点。

    启发式规则：路径包含 /api/、/v1/、/v2/ 等，或返回带 JSON 的查询接口。
    """
    patterns = [
        r'/api/',
        r'/v\d+/',
        r'/rest/',
        r'/graphql',
        r'\.json',
        r'/data/',
        r'/feed',
        r'/search',
        r'/query',
    ]
    url_lower = url.lower()
    return any(re.search(p, url_lower) for p in patterns)


class SPAHandler:
    """
    SPA 处理器

    使用步骤：
    1. start_intercept(page)  - 注册响应拦截器
    2. （等待页面加载完成）
    3. get_api_data()          - 获取已拦截的 API 数据
    4. fetch_api_direct(page, url) - 从页面上下文直接 fetch
    5. extract_from_dom(html)  - 降级 DOM 提取
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self._intercepted: List[Dict[str, Any]] = []  # [{url, data, list_data}]
        self._candidate_urls: List[str] = []

    # ------------------------------------------------------------------
    # 响应拦截
    # ------------------------------------------------------------------

    def start_intercept(self, page: Any) -> None:
        """
        注册 Playwright page.on('response') 监听器以拦截 JSON API 响应。

        Args:
            page: Playwright Page 对象
        """
        self._intercepted.clear()
        self._candidate_urls.clear()

        async def _on_response(response: Any) -> None:
            try:
                url = response.url
                content_type = response.headers.get('content-type', '')
                if not _is_json_content_type(content_type):
                    return
                if not _is_api_url(url):
                    return
                body = await response.text()
                obj = json.loads(body)
                list_data = _extract_list_from_json(obj)
                self._intercepted.append({
                    'url': url,
                    'raw': obj,
                    'list_data': list_data,
                })
                if url not in self._candidate_urls:
                    self._candidate_urls.append(url)
            except Exception:
                pass  # 忽略解析失败的响应

        page.on('response', _on_response)

    def get_intercepted(self) -> List[Dict[str, Any]]:
        """返回已拦截的 API 响应列表"""
        return list(self._intercepted)

    def get_best_list_data(self) -> Optional[List[Dict]]:
        """
        从所有已拦截响应中返回最优的列表数据（按条目数量排序，取最多的那个）。
        """
        best: Optional[List[Dict]] = None
        for entry in self._intercepted:
            lst = entry.get('list_data')
            if lst and (best is None or len(lst) > len(best)):
                best = lst
        return best

    # ------------------------------------------------------------------
    # 直接 fetch
    # ------------------------------------------------------------------

    async def fetch_api_direct(self, page: Any, api_url: str) -> Optional[List[Dict]]:
        """
        从页面的 JavaScript 上下文中直接 fetch API URL，绕过跨域限制。

        Args:
            page: Playwright Page 对象
            api_url: 要 fetch 的 API URL

        Returns:
            提取到的列表数据，或 None
        """
        try:
            result = await page.evaluate(
                """async (url) => {
                    try {
                        const resp = await fetch(url, {credentials: 'include'});
                        if (!resp.ok) return null;
                        return await resp.json();
                    } catch (e) {
                        return null;
                    }
                }""",
                api_url,
            )
            if result is None:
                return None
            return _extract_list_from_json(result)
        except Exception:
            return None

    async def try_candidate_apis(self, page: Any, base_url: str) -> Optional[List[Dict]]:
        """
        依次尝试已发现的候选 API 和从 base_url 推断的通用端点，返回第一个有效结果。
        """
        # 已拦截到的最优数据直接返回（无需额外 fetch）
        best = self.get_best_list_data()
        if best:
            return best

        # 尝试候选 API URL（直接 fetch）
        for url in self._candidate_urls:
            data = await self.fetch_api_direct(page, url)
            if data:
                return data

        # 推断常用 API 路径
        parsed = urlparse(base_url)
        guesses = [
            urljoin(base_url, '/api/data'),
            urljoin(base_url, '/api/list'),
            urljoin(base_url, f'{parsed.path.rstrip("/")}/data.json'),
        ]
        for url in guesses:
            data = await self.fetch_api_direct(page, url)
            if data:
                return data

        return None

    # ------------------------------------------------------------------
    # 降级 DOM 提取
    # ------------------------------------------------------------------

    def extract_from_dom(self, html: str, target_fields: Optional[List[Dict]] = None) -> List[Dict]:
        """
        从渲染后的 DOM 提取数据（API 不可用时的降级方案）。

        尝试从重复结构容器中提取文本，每个容器子元素对应一条记录。

        Args:
            html: 渲染后的完整 HTML
            target_fields: Spec 中的字段定义列表（可选，用于指导提取）

        Returns:
            提取到的记录列表
        """
        if not _BS4_AVAILABLE:
            return []

        soup = BeautifulSoup(html, 'html.parser')

        # 找到第一个有足够重复子元素的容器
        for tag_name in ['ul', 'ol', 'div', 'section', 'table']:
            containers = soup.find_all(tag_name)
            for container in containers:
                children = [c for c in container.children
                            if hasattr(c, 'name') and c.name]
                if len(children) < 3:
                    continue

                records = []
                for child in children:
                    text = child.get_text(separator=' ', strip=True)
                    if text:
                        record: Dict[str, Any] = {'text': text}
                        # 若提供了字段定义，尝试按选择器匹配
                        if target_fields:
                            for field_def in target_fields:
                                selector = field_def.get('selector', '')
                                field_name = field_def.get('name', 'field')
                                if selector:
                                    found = child.select_one(selector)
                                    if found:
                                        record[field_name] = found.get_text(strip=True)
                        records.append(record)

                if records:
                    return records

        return []

    # ------------------------------------------------------------------
    # LLM 字段映射（可选）
    # ------------------------------------------------------------------

    async def map_fields_with_llm(
        self,
        records: List[Dict],
        target_fields: List[Dict],
    ) -> List[Dict]:
        """
        使用 LLM 将 API 返回的字段映射到 Spec 目标字段。

        Args:
            records: 原始记录列表（来自 API）
            target_fields: Spec 中的目标字段定义

        Returns:
            映射后的记录列表；LLM 不可用时返回原始记录
        """
        if not self.llm_client or not records:
            return records

        sample = records[:3]
        field_names = [f.get('name') for f in target_fields]

        prompt = (
            f"以下是从 API 提取的数据样本（JSON）：\n"
            f"{json.dumps(sample, ensure_ascii=False, indent=2)}\n\n"
            f"目标字段：{field_names}\n\n"
            f"请将样本中的字段映射到目标字段，以 JSON 格式输出映射规则，"
            f"格式：{{\"source_key\": \"target_key\", ...}}"
        )

        try:
            response = await self.llm_client.chat(prompt)
            # 尝试解析 mapping
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0]
            else:
                json_str = response

            mapping: Dict[str, str] = json.loads(json_str)

            # 应用映射
            mapped = []
            for record in records:
                new_record = {}
                for src, tgt in mapping.items():
                    if src in record:
                        new_record[tgt] = record[src]
                # 保留未映射的字段
                for k, v in record.items():
                    if k not in mapping:
                        new_record[k] = v
                mapped.append(new_record)
            return mapped

        except Exception:
            return records

    # ------------------------------------------------------------------
    # 主执行流程（集成使用）
    # ------------------------------------------------------------------

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行 SPA 提取流程。

        context 键：
        - browser: 浏览器对象（需有 .page 属性）
        - current_url: 当前页面 URL
        - spec: SpecContract（可选，包含 targets）
        - wait_ms: 渲染等待时间（毫秒，默认 2000）

        Returns:
            {
                'success': bool,
                'method': 'api_intercept' | 'api_fetch' | 'dom_fallback',
                'records': List[Dict],
                'candidate_apis': List[str],
                'intercepted_count': int,
            }
        """
        browser = context.get('browser')
        current_url = context.get('current_url', '')
        spec = context.get('spec') or {}
        wait_ms = context.get('wait_ms', 2000)

        if browser is None:
            return {
                'success': False,
                'error': 'browser is required',
                'method': None,
                'records': [],
                'candidate_apis': [],
                'intercepted_count': 0,
            }

        page = getattr(browser, 'page', None)

        # 获取 Spec 中的目标字段
        target_fields: List[Dict] = []
        for target in spec.get('targets', []):
            target_fields.extend(target.get('fields', []))

        # 步骤 1：注册拦截（若 page 对象可用）
        if page is not None:
            self.start_intercept(page)

            # 等待 JS 渲染完成
            try:
                await page.wait_for_load_state('networkidle')
            except Exception:
                try:
                    await asyncio.sleep(wait_ms / 1000)
                except Exception:
                    pass

        # 步骤 2：尝试从已拦截数据或直接 fetch API 获取数据
        records: Optional[List[Dict]] = None
        method = 'dom_fallback'

        if page is not None:
            records = await self.try_candidate_apis(page, current_url)
            if records:
                method = 'api_intercept' if self.get_best_list_data() else 'api_fetch'

        # 步骤 3：降级 DOM 提取
        if not records:
            try:
                html = await browser.get_html()
                records = self.extract_from_dom(html, target_fields)
                method = 'dom_fallback'
            except Exception:
                records = []

        # 步骤 4：LLM 字段映射（可选）
        if records and target_fields and self.llm_client:
            records = await self.map_fields_with_llm(records, target_fields)

        return {
            'success': bool(records),
            'method': method,
            'records': records or [],
            'candidate_apis': list(self._candidate_urls),
            'intercepted_count': len(self._intercepted),
        }

    def get_description(self) -> str:
        return "处理SPA页面：拦截API响应，提取列表数据，降级DOM提取"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'browser' in context
