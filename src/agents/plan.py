"""
规划智能体 - PlanAgent
根据页面结构和目标规划提取策略
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import json
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from .sense import _safe_parse_json, DegradationTracker, AgentInterface


class PlanAgent(AgentInterface):
    """规划智能体 - 规划数据提取策略"""

    def __init__(self, llm_client=None, degradation_tracker: Optional[DegradationTracker] = None):
        super().__init__("PlanAgent", "plan")
        self.llm_client = llm_client
        self.degradation_tracker = degradation_tracker or DegradationTracker()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行规划任务
        """
        page_structure = context.get('page_structure', {})
        spec = context.get('spec', {})
        llm_client = context.get('llm_client') or self.llm_client
        degradation_info = None

        try:
            # 1. 分析目标字段
            targets = spec.get('targets', [])
            if not targets:
                return {
                    'success': False,
                    'error': 'Spec 中未定义目标字段',
                    'selectors': {},
                    'strategy': {}
                }

            # 2. 生成选择器策略
            selectors = await self._generate_selectors(
                page_structure, targets, llm_client
            )

            # 3. 确定提取策略
            strategy = self._determine_extraction_strategy(
                page_structure, targets
            )

            result = {
                'success': True,
                'selectors': selectors,
                'strategy': strategy,
                'generated_code': await self._generate_extraction_code(
                    selectors, strategy, targets
                )
            }

            return result

        except Exception as e:
            error_msg = str(e)
            print(f"规划失败: {error_msg}")
            # 记录降级
            degradation_info = self.degradation_tracker.record_degradation(
                self.name, 'execute', error_msg
            )
            return {
                'success': False,
                'error': str(e),
                'selectors': {},
                'strategy': {},
                'degradation': degradation_info
            }

    async def _generate_selectors(self, structure: Dict, targets: List[Dict],
                                 llm_client) -> Dict[str, str]:
        """生成提取选择器"""
        if not llm_client:
            # 降级：基于结构特征生成基础选择器
            return self._generate_fallback_selectors(structure, targets)

        try:
            prompt = f"""根据页面结构为以下目标生成CSS选择器：

页面结构：
{json.dumps(structure, ensure_ascii=False)}

目标字段：
{json.dumps(targets, ensure_ascii=False)}

请输出 JSON 格式：
{{
    "field_name": "CSS选择器",
    ...
}}"""

            # 使用推理任务（DeepSeek）
            if hasattr(llm_client, 'reason'):
                response = await llm_client.reason(prompt)
            else:
                response = await llm_client.chat([{"role": "user", "content": prompt}])

            selectors = _safe_parse_json(response, "选择器生成")

            # 只返回字段名作为键的选择器
            filtered_selectors = {}
            for target in targets:
                for field in target.get('fields', []):
                    field_name = field['name']
                    if field_name in selectors:
                        filtered_selectors[field_name] = selectors[field_name]

            return filtered_selectors

        except Exception as e:
            print(f"LLM 选择器生成失败: {e}")
            return self._generate_fallback_selectors(structure, targets)

    def _generate_fallback_selectors(self, structure: Dict, targets: List[Dict]) -> Dict[str, str]:
        """降级：基于结构特征生成基础选择器"""
        selectors = {}

        # 使用页面结构中的主要选择器
        main_selector = structure.get('main_content_selector', '.content')
        content_selectors = structure.get('content_selectors', [])

        for target in targets:
            for field in target.get('fields', []):
                field_name = field['name']

                # 根据字段名称推测选择器
                if 'title' in field_name.lower():
                    selectors[field_name] = 'h1, h2, h3, .title, .headline, [class*="title"]'
                elif 'price' in field_name.lower():
                    selectors[field_name] = '.price, [class*="price"], [class*="cost"], span:last-child'
                elif 'desc' in field_name.lower() or 'content' in field_name.lower():
                    selectors[field_name] = f'{main_selector}, p, [class*="desc"], [class*="content"]'
                elif 'date' in field_name.lower():
                    selectors[field_name] = '[datetime], time, [class*="date"], .time'
                elif 'author' in field_name.lower() or 'by' in field_name.lower():
                    selectors[field_name] = '[class*="author"], [class*="user"], .byline'
                else:
                    # 使用内容选择器或通用选择器
                    if content_selectors:
                        selectors[field_name] = content_selectors[0]
                    else:
                        selectors[field_name] = 'div, p, span'

        return selectors

    def _determine_extraction_strategy(self, structure: Dict, targets: List[Dict]) -> Dict[str, Any]:
        """确定提取策略"""
        page_type = structure.get('page_type', 'unknown')
        has_pagination = structure.get('has_pagination', False)
        pagination_type = structure.get('pagination_type', 'none')

        strategy = {
            'strategy_type': 'css',  # 目前主要是 CSS 选择器
            'page_type': page_type,
            'needs_pagination': has_pagination,
            'pagination_strategy': pagination_type if has_pagination else 'none',
            'approach': 'direct' if page_type == 'detail' else 'list',
            'item_selector': structure.get('main_content_selector', 'div'),
            'batch_size': 10 if has_pagination else 1
        }

        # 根据爬取模式调整策略
        crawl_mode = targets[0].get('crawl_mode', 'single_page') if targets else 'single_page'
        if crawl_mode == 'multi_page':
            strategy['approach'] = 'pagination'
        elif crawl_mode == 'full_site':
            strategy['approach'] = 'breadth_first'

        return strategy

    async def _generate_extraction_code(self, selectors: Dict[str, str],
                                      strategy: Dict[str, Any], targets: List[Dict]) -> str:
        """生成提取代码"""
        # 由于代码生成器可能在 executor 模块中定义，这里简化实现
        code_parts = [
            "# 生成的提取代码",
            "from bs4 import BeautifulSoup",
            "import json",
            "",
            "# HTML 内容由外部提供",
            "html = context.get('html', '')",
            "soup = BeautifulSoup(html, 'html.parser')",
            ""
        ]

        # 根据策略生成相应代码
        if strategy.get('approach') == 'list':
            code_parts.extend([
                "# 提取列表项目",
                f"items = soup.select('{strategy.get('item_selector', 'div')}')",
                "results = []",
                "for item in items:",
                "    data = {}"
            ])

            # 添加字段提取
            for field_name, selector in selectors.items():
                code_parts.append(f"    data['{field_name}'] = item.select_one('{selector}').get_text().strip() if item.select_one('{selector}') else ''")

            code_parts.extend([
                "    results.append(data)",
                "",
                "print(json.dumps(results, ensure_ascii=False))"
            ])
        else:
            # 单个项目提取
            code_parts.append("# 提取单个项目")
            code_parts.append("data = {}")
            for field_name, selector in selectors.items():
                code_parts.append(f"data['{field_name}'] = soup.select_one('{selector}').get_text().strip() if soup.select_one('{selector}') else ''")
            code_parts.extend([
                "results = [data]",
                "",
                "print(json.dumps(results, ensure_ascii=False))"
            ])

        return '\n'.join(code_parts)

    def get_description(self) -> str:
        return "根据页面结构规划提取策略和生成选择器"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'page_structure' in context or 'spec' in context