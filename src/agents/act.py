"""
执行智能体 - ActAgent
执行数据提取操作
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import json
import re
import logging
from datetime import datetime
from urllib.parse import urljoin
from pathlib import Path
import zipfile

logger = logging.getLogger(__name__)

from .base import _safe_parse_json, AgentInterface


class ActAgent(AgentInterface):
    """执行智能体 - 执行数据提取操作"""

    def __init__(self, sandbox=None):
        super().__init__("ActAgent", "act")
        self.sandbox = sandbox

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行提取任务
        """
        browser = context.get('browser')
        selectors = context.get('selectors', {})
        strategy = context.get('strategy', {})
        generated_code = context.get('generated_code')
        crawl_mode = context.get('crawl_mode', 'single_page')
        max_pages = context.get('max_pages', 1)

        try:
            extracted_data = []
            errors = []
            strategy_type = strategy.get('strategy_type', 'css')

            # 策略为 LLM 时优先使用 LLM 提取
            llm_client = context.get('llm_client')
            spec = context.get('spec', {})
            if strategy_type == 'llm' and llm_client:
                try:
                    extracted_data = await self._extract_with_llm(
                        browser, llm_client, spec.get('targets', []),
                        context.get('html_snapshot', '')
                    )
                except Exception as e:
                    logger.debug(f"LLM 主提取失败, 降级为 CSS: {e}")

            # CSS 选择器提取（LLM 策略时作为补充；CSS 策略时作为主方法）
            if not extracted_data:
                if crawl_mode == 'multi_page':
                    extracted_data = await self._extract_with_pagination(
                        browser, selectors, strategy, max_pages
                    )
                else:
                    extracted_data = await self._extract_simple(browser, selectors, strategy)

            # CSS 提取结果不佳时自动回退到 LLM（无论策略类型）
            if llm_client and len(extracted_data) < 3 and strategy_type != 'llm':
                try:
                    llm_data = await self._extract_with_llm(
                        browser, llm_client, spec.get('targets', []),
                        context.get('html_snapshot', '')
                    )
                    if len(llm_data) > len(extracted_data):
                        logger.info(f"LLM 提取更优: {len(llm_data)} 条 (CSS: {len(extracted_data)} 条)")
                        extracted_data = llm_data
                except Exception as e:
                    logger.debug(f"LLM 提取回退失败: {e}")

            zip_fallback = None
            if self._should_try_zip_fallback(extracted_data, selectors):
                zip_fallback = await self._extract_from_zip_bundle(browser)
                if zip_fallback.get('records'):
                    extracted_data.extend(zip_fallback['records'])

            result = {
                'success': True,
                'extracted_data': extracted_data,
                'count': len(extracted_data),
                'extraction_metrics': self._calculate_extraction_metrics(extracted_data, selectors)
            }
            if zip_fallback:
                result['zip_fallback'] = zip_fallback.get('metadata', {})

            if errors:
                result['errors'] = errors

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'extracted_data': [],
                'extraction_metrics': {}
            }

    def _should_try_zip_fallback(self, extracted_data: List[Dict[str, Any]],
                                 selectors: Dict[str, str]) -> bool:
        """当正常提取结果较少时，尝试 ZIP 代码包回退提取。"""
        if not extracted_data:
            return True
        if len(extracted_data) > 1:
            return False
        field_count = len(selectors or {})
        non_empty_fields = sum(1 for value in extracted_data[0].values() if value)
        return field_count > 1 and non_empty_fields <= 1

    async def _extract_from_zip_bundle(self, browser) -> Dict[str, Any]:
        """检测并下载 ZIP 代码包，从中提取 HTML/CSS/JS。"""
        metadata: Dict[str, Any] = {
            'attempted': True,
            'used': False,
            'zip_links_found': 0,
            'errors': []
        }
        records: List[Dict[str, Any]] = []
        try:
            html = await browser.get_html()
            referer = await browser.get_current_url()
            zip_links = self._find_zip_links(html, referer)
            metadata['zip_links_found'] = len(zip_links)
            if not zip_links:
                metadata['errors'].append('no_zip_link_found')
                return {'records': records, 'metadata': metadata}

            zip_url = zip_links[0]
            metadata['download_url'] = zip_url
            from src.tools.downloader import FileDownloader
            downloader = FileDownloader()
            download_result = await downloader.download(zip_url, file_type='zip', referer=referer)
            metadata['download'] = {
                'success': bool(download_result.get('success')),
                'error': download_result.get('error')
            }
            if not download_result.get('success') or not download_result.get('content'):
                metadata['errors'].append(
                    download_result.get('error') or 'zip_download_failed'
                )
                return {'records': records, 'metadata': metadata}

            zip_path = Path(download_result['content'])
            metadata['archive_path'] = str(zip_path)
            records = self._extract_code_records_from_zip(zip_path, zip_url)
            metadata['records_extracted'] = len(records)
            metadata['used'] = len(records) > 0
            if not records:
                metadata['errors'].append('no_html_css_js_files_in_zip')
            return {'records': records, 'metadata': metadata}
        except Exception as e:
            metadata['errors'].append(f'zip_fallback_exception: {e}')
            return {'records': records, 'metadata': metadata}

    def _find_zip_links(self, html: str, base_url: str) -> List[str]:
        """从 HTML 中发现 ZIP 下载链接。"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        links: List[str] = []
        seen = set()
        for anchor in soup.select('a[href]'):
            href = (anchor.get('href') or '').strip()
            if not href:
                continue
            href_lower = href.lower()
            text_lower = anchor.get_text(' ', strip=True).lower()
            if (
                '.zip' in href_lower
                or ('zip' in href_lower and any(k in href_lower for k in ('download', 'export')))
                or ('zip' in text_lower and any(k in text_lower for k in ('download', 'export', '源码', 'code')))
            ):
                absolute = urljoin(base_url, href)
                if absolute not in seen:
                    seen.add(absolute)
                    links.append(absolute)
        return links

    def _extract_code_records_from_zip(self, zip_path: Path, source_url: str) -> List[Dict[str, Any]]:
        """从 ZIP 中提取 HTML/CSS/JS 文件内容。"""
        ext_map = {
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.js': 'javascript',
            '.mjs': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'javascript',
            '.tsx': 'javascript'
        }
        records: List[Dict[str, Any]] = []
        with zipfile.ZipFile(zip_path, 'r') as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                suffix = Path(info.filename).suffix.lower()
                asset_type = ext_map.get(suffix)
                if not asset_type:
                    continue
                raw = archive.read(info.filename)
                try:
                    content = raw.decode('utf-8')
                except UnicodeDecodeError:
                    content = raw.decode('utf-8', errors='replace')
                records.append({
                    'record_type': 'code_asset',
                    'asset_type': asset_type,
                    'file_path': info.filename,
                    'content': content,
                    'source': source_url
                })
        return records

    # 字段名中含这些关键词时，提取 HTML 源码而非纯文本
    _HTML_FIELD_HINTS = {'html', 'code', 'source', 'markup', 'template', 'structure', 'snippet'}

    def _should_extract_html(self, field_name: str) -> bool:
        """判断字段是否应提取 HTML 源码"""
        name_lower = field_name.lower()
        return any(hint in name_lower for hint in self._HTML_FIELD_HINTS)

    @staticmethod
    def _sanitize_selector(selector: str) -> tuple:
        """
        预处理 LLM 生成的 CSS 选择器，修复常见错误。

        LLM 常生成 `a.class@href` 或 `img.photo@src` 格式（CSS selector + attribute），
        但 `@` 不是合法的 CSS selector 字符。

        Returns:
            (clean_selector, target_attr) — 清理后的选择器和要提取的属性名（或 None）
        """
        if not selector or not isinstance(selector, str):
            return (selector or '', None)

        selector = selector.strip()
        target_attr = None

        # 模式 1: sel@attr — 如 "a.card@href", "img@src", "span.price@data-value"
        if '@' in selector:
            parts = selector.rsplit('@', 1)
            selector = parts[0].strip()
            target_attr = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None

        # 模式 2: sel::attr(name) — 如 "a::attr(href)" (Scrapy 风格，BS4 不支持)
        attr_match = re.match(r'^(.+?)::attr\(([^)]+)\)$', selector)
        if attr_match:
            selector = attr_match.group(1).strip()
            target_attr = attr_match.group(2).strip()

        # 移除 selector 中残留的非法字符
        selector = re.sub(r'[{}]', '', selector)

        return (selector, target_attr)

    async def _extract_simple(self, browser, selectors: Dict[str, str],
                             strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """简单提取（单页）"""
        html = await browser.get_html()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # 获取项目容器 — 优先用 container_selector 精确定位
        raw_container = strategy.get('container_selector') or strategy.get('item_selector', 'div')
        container_selector, _ = self._sanitize_selector(raw_container)
        items = soup.select(container_selector) if container_selector else [soup]
        # 如果容器太宽泛（只匹配到 1 个且是 body/main/html），尝试用首个 target selector 的父级
        if len(items) == 1 and items[0].name in ('html', 'body', 'main', 'div') and selectors:
            raw_first = next(iter(selectors.values()), None)
            if raw_first:
                first_sel, _ = self._sanitize_selector(raw_first)
                sub_items = soup.select(first_sel) if first_sel else []
                if len(sub_items) > 1:
                    items = [el.parent for el in sub_items if el.parent]
                    # 去重保持顺序
                    seen_ids = set()
                    unique = []
                    for it in items:
                        oid = id(it)
                        if oid not in seen_ids:
                            seen_ids.add(oid)
                            unique.append(it)
                    items = unique

        extracted = []
        for item in items:
            data = {}
            for field_name, selector in selectors.items():
                try:
                    clean_sel, target_attr = self._sanitize_selector(selector)
                    if not clean_sel:
                        data[field_name] = ''
                        continue

                    element = item.select_one(clean_sel)
                    # 回退：容器内找不到时在全局搜索
                    if not element:
                        element = soup.select_one(clean_sel)
                    if element:
                        if target_attr:
                            # LLM 明确指定了要提取的属性
                            value = element.get(target_attr, '') or element.get_text().strip()
                        elif element.has_attr('href') or element.has_attr('src'):
                            value = element.get('href') or element.get('src') or element.get_text().strip()
                        elif self._should_extract_html(field_name):
                            # 结构/代码类字段提取 HTML 源码
                            value = str(element)
                        else:
                            value = element.get_text().strip()
                    else:
                        value = ''

                    data[field_name] = value
                except Exception as e:
                    logger.debug(f"提取字段 {field_name} 失败: {e}")
                    data[field_name] = ''

            # 只添加包含至少一个非空字段的项目
            if any(v for v in data.values() if v):
                extracted.append(data)

        return extracted

    async def _extract_with_pagination(self, browser, selectors: Dict[str, str],
                                     strategy: Dict[str, Any], max_pages: int) -> List[Dict[str, Any]]:
        """带分页的提取"""
        all_data = []

        # 当前页面数据
        current_data = await self._extract_simple(browser, selectors, strategy)
        all_data.extend(current_data)

        # 如果需要分页处理
        if strategy.get('needs_pagination'):
            pagination_strategy = strategy.get('pagination_strategy', 'standard')
            page_count = 1

            while page_count < max_pages:
                # 查找下一页按钮
                next_page_found = await self._click_next_page(browser, pagination_strategy)

                if not next_page_found:
                    break

                # 等待页面加载
                await browser.page.wait_for_timeout(2000)  # 2秒等待

                # 提取新页面数据
                page_data = await self._extract_simple(browser, selectors, strategy)
                all_data.extend(page_data)

                page_count += 1

        return all_data

    async def _click_next_page(self, browser, strategy: str) -> bool:
        """点击下一页"""
        # 不同的分页策略
        next_selectors = [
            'a[rel="next"]',
            'a:contains("下一页")', 'a:contains("Next")', 'a:contains("»")',
            '.next', '[class*="next"]', '[class*="pagination"] a:not([rel="prev"])',
            '.pager-next a', '.pagination .next a'
        ]

        for selector in next_selectors:
            try:
                element = await browser.page.query_selector(selector)
                if element:
                    # 滚动到元素可见
                    await element.scroll_into_view_if_needed()
                    await browser.page.wait_for_timeout(500)  # 等待滚动完成

                    # 点击元素
                    await element.click()
                    await browser.page.wait_for_load_state('networkidle')
                    return True
            except Exception as e:
                print(f"点击下一页按钮失败 {selector}: {e}")
                continue

        return False

    def _calculate_extraction_metrics(self, data: List[Dict], selectors: Dict[str, str]) -> Dict[str, Any]:
        """计算提取指标"""
        total_items = len(data)

        # 统计每个选择器的成功率
        selector_stats = {}
        failed_selectors = {}
        for field_name in selectors.keys():
            successful_extractions = sum(1 for item in data if item.get(field_name))
            failed = total_items - successful_extractions if total_items > 0 else 0
            selector_stats[field_name] = {
                'success_count': successful_extractions,
                'success_rate': successful_extractions / total_items if total_items > 0 else 0
            }
            if failed > 0:
                failed_selectors[field_name] = failed

        overall_success_rate = (
            sum(stat['success_rate'] for stat in selector_stats.values()) / len(selector_stats)
            if selector_stats else 0
        )

        return {
            'total_items': total_items,
            'selector_performance': selector_stats,
            'failed_selectors': failed_selectors,
            'success_rate': overall_success_rate,
            'average_fields_per_item': sum(len(item) for item in data) / total_items if total_items > 0 else 0
        }

    def _get_next_page_url(self, current_url: str, next_page: int) -> Optional[str]:
        """
        根据当前 URL 和目标页码推算下一页 URL。

        Args:
            current_url: 当前页面 URL
            next_page: 目标页码

        Returns:
            下一页 URL，若无法推算则返回 None
        """
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

        parsed = urlparse(current_url)
        params = parse_qs(parsed.query, keep_blank_values=True)

        # 页码类参数
        for param in ('page', 'p', 'pageNum', 'pn', 'pg'):
            if param in params:
                params[param] = [str(next_page)]
                new_query = urlencode(params, doseq=True)
                return urlunparse(parsed._replace(query=new_query))

        # offset/start 类参数
        # current_page = next_page - 1; page_size = current_offset / (current_page - 1) if current_page > 1 else current_offset
        for param in ('offset', 'start'):
            if param in params:
                current_offset = int(params[param][0])
                current_page = next_page - 1
                page_size = current_offset // max(current_page - 1, 1) if current_page > 1 else current_offset
                params[param] = [str(page_size * (next_page - 1))]
                new_query = urlencode(params, doseq=True)
                return urlunparse(parsed._replace(query=new_query))

        # /page/N 路径模式
        path = parsed.path
        page_path_match = re.search(r'(/page/)(\d+)', path)
        if page_path_match:
            new_path = path[:page_path_match.start(2)] + str(next_page) + path[page_path_match.end(2):]
            return urlunparse(parsed._replace(path=new_path))

        # 路径末尾数字模式（如 /articles/1）
        numeric_end_match = re.search(r'(/\d+)$', path)
        if numeric_end_match:
            new_path = path[:numeric_end_match.start(1)] + '/' + str(next_page)
            return urlunparse(parsed._replace(path=new_path))

        return None

    async def _extract_next_url_from_dom(self, browser) -> Optional[str]:
        """从 DOM 中提取 rel=next 链接作为下一页 URL。"""
        from urllib.parse import urljoin
        try:
            href = await browser.page.evaluate(
                'document.querySelector("a[rel~=\'next\']")?.getAttribute("href") ?? null'
            )
            if not href:
                return None
            if href.startswith('http'):
                return href
            base_url = await browser.get_current_url()
            return urljoin(base_url, href)
        except Exception:
            return None

    async def _discover_next_url_from_links(self, browser, current_url: str,
                                            next_page: int) -> Optional[str]:
        """从页面中的所有链接里发现下一页 URL。"""
        try:
            hrefs = await browser.page.evaluate(
                'Array.from(document.querySelectorAll("a[href]")).map(a => a.getAttribute("href"))'
            )
            for href in (hrefs or []):
                candidate = self._get_next_page_url(href, next_page)
                if candidate:
                    return candidate
            return None
        except Exception:
            return None

    async def _execute_code(self, code: str, html: str) -> List[Dict]:
        """执行 LLM 生成的代码 — 统一走 Sandbox"""
        import json

        if self.sandbox is None:
            from src.executors.executor import Sandbox
            self.sandbox = Sandbox(strict_mode=True)

        result = await self.sandbox.execute(code, stdin_data=html,
                                            timeout=self.sandbox.default_timeout)

        if result['success'] and result['stdout']:
            try:
                return json.loads(result['stdout'])
            except json.JSONDecodeError:
                logger.warning("代码输出不是合法 JSON")
                return []
        else:
            logger.warning(f"代码执行失败: {result['stderr'][:200]}")
            return []

    async def _extract_with_llm(self, browser, llm_client, targets: List[Dict],
                                html_snapshot: str = '') -> List[Dict[str, Any]]:
        """
        LLM 驱动的数据提取：将页面 HTML 发送给 LLM，由其直接提取结构化数据。
        当 CSS 选择器提取效果不佳时作为备选方案。
        """
        # 获取页面 HTML（优先用已有快照，减少浏览器调用）
        html = html_snapshot or await browser.get_html()
        # 截断到合理长度（约 12K tokens）
        max_chars = 15000
        if len(html) > max_chars:
            html = html[:max_chars] + '\n<!-- ... truncated ... -->'

        # 构建目标字段描述
        field_descriptions = []
        for target in targets:
            for field in target.get('fields', []):
                field_descriptions.append(
                    f"- {field['name']}: {field.get('description', field['name'])}"
                )
        if not field_descriptions:
            return []

        fields_text = '\n'.join(field_descriptions)
        field_names = [f['name'] for t in targets for f in t.get('fields', [])]

        prompt = f"""Analyze this HTML page and extract structured data matching the required fields.

Required fields:
{fields_text}

Rules:
1. Return a JSON array of objects, each object having the required field names as keys.
2. Extract ALL matching records you can find on the page.
3. For fields about HTML/code/structure, include the actual HTML markup.
4. For text fields, extract clean readable text.
5. If a field value is not found for a record, use empty string "".
6. Return ONLY the JSON array, no explanation.

HTML:
{html}"""

        try:
            response = await llm_client.generate(
                prompt=prompt,
                system_prompt="You are a precise data extraction assistant. Return only valid JSON arrays.",
                max_tokens=4096,
                temperature=0.1
            )

            # 解析 LLM 返回的 JSON
            parsed = _safe_parse_json(response)
            if isinstance(parsed, list):
                records = parsed
            elif isinstance(parsed, dict):
                # LLM 可能返回 {"data": [...]} 或 {"records": [...]}
                for key in ('data', 'records', 'items', 'results'):
                    if isinstance(parsed.get(key), list):
                        records = parsed[key]
                        break
                else:
                    records = [parsed]
            else:
                return []

            # 过滤：只保留含有至少一个目标字段且非空的记录
            valid_records = []
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                # 只保留目标字段
                filtered = {k: str(v) for k, v in rec.items() if k in field_names and v}
                if filtered:
                    valid_records.append(filtered)

            logger.info(f"LLM 提取: {len(valid_records)} 条有效记录 (原始 {len(records)})")
            return valid_records

        except Exception as e:
            logger.warning(f"LLM 提取失败: {e}")
            return []

    def get_description(self) -> str:
        return "执行数据提取操作"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'browser' in context and ('selectors' in context or 'strategy' in context)
