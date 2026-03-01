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

    def _fallback_strategy(self, structure: Dict, spec: Dict) -> Dict[str, Any]:
        """
        降级策略：为缺少 selector 的字段生成默认选择器。

        Returns:
            {'selectors': {...}, 'selector_info': {...}}
        """
        selectors = {}
        selector_info = {}

        for target in spec.get('targets', []):
            for field in target.get('fields', []):
                name = field.get('name', '')
                if field.get('selector'):
                    selectors[name] = field['selector']
                    selector_info[name] = 'from_spec'
                else:
                    description = field.get('description', '')
                    selectors[name] = self._generate_default_selector(name, description)
                    selector_info[name] = 'generated'

        return {'selectors': selectors, 'selector_info': selector_info}

    def _generate_default_selector(self, field_name: str, description: str = '') -> str:
        """根据字段名称和描述生成默认 CSS 选择器。"""
        name_lower = field_name.lower()
        desc_lower = description.lower()

        if 'title' in name_lower or '标题' in desc_lower or 'title' in desc_lower:
            return 'h1, h2, h3, [class*="title"], [class*="headline"]'
        if 'price' in name_lower or '价格' in desc_lower or 'price' in desc_lower:
            return '[class*="price"], [class*="cost"], .price'
        if 'author' in name_lower or '作者' in desc_lower or 'author' in desc_lower:
            return '[class*="author"], [class*="user"], .byline, [rel="author"]'
        if 'date' in name_lower or '日期' in desc_lower or 'date' in desc_lower:
            return '[datetime], time, [class*="date"], [class*="time"]'
        if 'desc' in name_lower or 'content' in name_lower:
            return 'p, [class*="desc"], [class*="content"]'

        # 通用：将字段名转为 kebab-case 选择器
        kebab = re.sub(r'[_\s]+', '-', field_name.lower())
        return f'[class*="{kebab}"], #{kebab}, .{kebab}'

    def _extract_html_context(self, html: str, targets: List[Dict]) -> str:
        """从 HTML 中提取与目标字段相关的上下文文本。"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        context_parts = []
        field_names = [
            f.get('name', '')
            for t in targets for f in t.get('fields', [])
        ]

        for tag in soup.find_all(['h1', 'h2', 'h3', 'span', 'div', 'p', 'a'], limit=100):
            text = tag.get_text(strip=True)
            if not text:
                continue
            classes = ' '.join(tag.get('class', []))
            # 如果标签的文字或类名与字段相关则收录
            for name in field_names:
                if name.lower() in classes.lower() or name.lower() in text.lower():
                    context_parts.append(f'{tag.name}.{classes}: {text[:100]}')
                    break

        return '\n'.join(context_parts[:50])

    async def _generate_conservative(self, structure: Dict, spec: Dict,
                                     llm_client, html: str) -> Dict[str, Any]:
        """保守策略：使用宽泛选择器，优先从 spec 字段或语义推断。"""
        selectors = {}
        for target in spec.get('targets', []):
            for field in target.get('fields', []):
                name = field.get('name', '')
                if field.get('selector'):
                    selectors[name] = field['selector']
                else:
                    selectors[name] = self._generate_default_selector(
                        name, field.get('description', ''))

        return {
            'success': True,
            'strategy_type': 'conservative',
            'selectors': selectors,
        }

    async def _generate_aggressive(self, structure: Dict, spec: Dict,
                                   llm_client, html: str) -> Dict[str, Any]:
        """激进策略：直接分析 HTML 内容匹配字段名/描述。"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        selectors = {}
        for target in spec.get('targets', []):
            for field in target.get('fields', []):
                name = field.get('name', '')
                description = field.get('description', name)
                # 尝试在 HTML 中找到包含字段名或描述关键词的元素
                for tag in soup.find_all(['h1', 'h2', 'h3', 'span', 'div', 'p'], limit=200):
                    classes = ' '.join(tag.get('class', []))
                    tag_id = tag.get('id', '')
                    if (name.lower() in classes.lower() or
                            name.lower() in tag_id.lower() or
                            description.lower() in tag.get_text(strip=True).lower()):
                        class_sel = f'.{classes.split()[0]}' if tag.get('class') else tag.name
                        selectors[name] = class_sel
                        break
                else:
                    selectors[name] = self._generate_default_selector(name, description)

        return {
            'success': True,
            'strategy_type': 'aggressive',
            'selectors': selectors,
        }

    def get_description(self) -> str:
        return "根据页面结构规划提取策略和生成选择器"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'page_structure' in context or 'spec' in context