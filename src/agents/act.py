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

logger = logging.getLogger(__name__)

from .sense import _safe_parse_json, AgentInterface


class ActAgent(AgentInterface):
    """执行智能体 - 执行数据提取操作"""

    def __init__(self):
        super().__init__("ActAgent", "act")

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

            if crawl_mode == 'multi_page':
                # 处理分页
                extracted_data = await self._extract_with_pagination(
                    browser, selectors, strategy, max_pages
                )
            elif crawl_mode == 'full_site':
                # 在 full_site 模式下调用简单提取
                extracted_data = await self._extract_simple(browser, selectors, strategy)
            else:  # single_page
                extracted_data = await self._extract_simple(browser, selectors, strategy)

            result = {
                'success': True,
                'extracted_data': extracted_data,
                'extraction_metrics': self._calculate_extraction_metrics(extracted_data, selectors)
            }

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

    async def _extract_simple(self, browser, selectors: Dict[str, str],
                             strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """简单提取（单页）"""
        html = await browser.get_html()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # 获取项目容器
        item_selector = strategy.get('item_selector', 'div')
        items = soup.select(item_selector) if item_selector else [soup]

        extracted = []
        for item in items:
            data = {}
            for field_name, selector in selectors.items():
                try:
                    element = item.select_one(selector)
                    if element:
                        # 检查元素是否具有 href 或 src 属性
                        if element.has_attr('href') or element.has_attr('src'):
                            value = element.get('href') or element.get('src') or element.get_text().strip()
                        else:
                            value = element.get_text().strip()
                    else:
                        value = ''

                    data[field_name] = value
                except Exception as e:
                    print(f"提取字段 {field_name} 失败: {e}")
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
        for field_name in selectors.keys():
            successful_extractions = sum(1 for item in data if item.get(field_name))
            selector_stats[field_name] = {
                'success_count': successful_extractions,
                'success_rate': successful_extractions / total_items if total_items > 0 else 0
            }

        return {
            'total_items': total_items,
            'selector_performance': selector_stats,
            'average_fields_per_item': sum(len(item) for item in data) / total_items if total_items > 0 else 0
        }

    def get_description(self) -> str:
        return "执行数据提取操作"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'browser' in context and ('selectors' in context or 'strategy' in context)