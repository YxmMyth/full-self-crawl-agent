"""
Agent 能力定义
执行层的 7 种智能体能力
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime
import json
import re
import logging

logger = logging.getLogger(__name__)


class DegradationTracker:
    """
    降级追踪器

    追踪 LLM 调用降级情况，提供警告和统计
    """

    def __init__(self, warning_threshold: int = 3):
        self.degradation_count = 0
        self.warning_threshold = warning_threshold
        self.degradation_history: List[Dict[str, Any]] = []

    def record_degradation(self, agent_name: str, operation: str, error: str) -> Dict[str, Any]:
        """
        记录降级事件

        Returns:
            包含 is_degraded 和 should_warn 的字典
        """
        self.degradation_count += 1
        event = {
            'agent': agent_name,
            'operation': operation,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.degradation_history.append(event)

        return {
            'is_degraded': True,
            'should_warn': self.degradation_count >= self.warning_threshold,
            'degradation_count': self.degradation_count,
            'message': f"LLM {operation} 降级 (总计: {self.degradation_count}次)"
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取降级统计"""
        return {
            'total_degradations': self.degradation_count,
            'warning_threshold': self.warning_threshold,
            'history': self.degradation_history[-10:]  # 最近10条
        }


def _safe_parse_json(response: str, context: str = "JSON解析") -> Dict:
    """
    安全解析 LLM 响应中的 JSON

    Args:
        response: LLM 返回的原始响应字符串
        context: 解析上下文描述，用于错误日志

    Returns:
        解析后的字典，失败时返回空字典
    """
    import logging
    logger = logging.getLogger(__name__)

    if not response or not response.strip():
        logger.warning(f"{context}失败: LLM 返回空响应")
        return {}

    try:
        # 尝试提取代码块中的 JSON
        if '```json' in response:
            json_str = response.split('```json')[1].split('```')[0].strip()
        elif '```' in response:
            json_str = response.split('```')[1].split('```')[0].strip()
        else:
            json_str = response.strip()

        # 检查是否为空字符串
        if not json_str:
            logger.warning(f"{context}失败: 提取的 JSON 字符串为空")
            return {}

        # 尝试解析
        result = json.loads(json_str)
        return result if result else {}

    except json.JSONDecodeError as e:
        logger.warning(f"{context}失败: JSON 解析错误 - {e}")
        logger.debug(f"原始响应前200字符: {response[:200]}")
        return {}
    except Exception as e:
        logger.warning(f"{context}失败: 解析异常 - {e}")
        return {}


class AgentCapability(str, Enum):
    """智能体能力"""
    SENSE = "sense"  # 感知页面结构和特征
    PLAN = "plan"  # 规划提取策略
    ACT = "act"  # 执行提取操作
    VERIFY = "verify"  # 验证数据质量
    JUDGE = "judge"  # 做出决策判断
    EXPLORE = "explore"  # 探索页面链接
    REFLECT = "reflect"  # 反思和优化策略


class AgentInterface:
    """智能体接口"""

    def __init__(self, name: str, capability: AgentCapability):
        self.name = name
        self.capability = capability

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行智能体

        Args:
            context: 执行上下文

        Returns:
            执行结果
        """
        raise NotImplementedError()

    def get_description(self) -> str:
        """获取智能体描述"""
        raise NotImplementedError()

    def can_handle(self, context: Dict[str, Any]) -> bool:
        """判断是否能处理"""
        raise NotImplementedError()


# ==================== 感知智能体 ====================

class SenseAgent(AgentInterface):
    """感知智能体 - 分析页面结构"""

    def __init__(self, degradation_tracker: Optional[DegradationTracker] = None):
        super().__init__("SenseAgent", AgentCapability.SENSE)
        self.degradation_tracker = degradation_tracker or DegradationTracker()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        browser = context.get('browser')
        spec = context.get('spec')
        llm_client = context.get('llm_client')
        degradation_info = None

        # 检查浏览器是否可用
        if not browser:
            return {
                'success': False,
                'error': 'Browser not available',
                'structure': {},
                'features': {},
                'anti_bot_detected': False
            }

        # 1. 获取初始 HTML
        html = await browser.get_html()

        # 2. 用 FeatureDetector 做初步分析
        from src.core.smart_router import FeatureDetector
        detector = FeatureDetector()
        quick_features = detector.analyze(html)

        # 3. 如果检测到 SPA，执行智能等待后重新获取 HTML 并重新分析
        if quick_features.get('is_spa'):
            html = await self._wait_for_spa_render(browser, html)
            features = detector.analyze(html)
        else:
            features = quick_features

        # 4. 截图
        screenshot = await browser.take_screenshot()

        # 5. 从 features 的 container_info 中提取容器信息（不再调用 _analyze_structure）
        container_info = features.get('container_info', {})
        features['main_content_selector'] = container_info.get('container_selector')
        features['item_selector'] = container_info.get('item_selector')
        features['estimated_items'] = container_info.get('estimated_items', 0)

        # 6. 集成 parser.py 的分页检测
        try:
            from src.tools.parser import HTMLParser
            current_url = ''
            if browser.page:
                current_url = browser.page.url
            parser = HTMLParser(html, current_url)
            pagination_info = parser.detect_pagination()
            features['pagination_type'] = self._determine_pagination_type(pagination_info, features)
            features['pagination_next_url'] = pagination_info.get('next_url')
            features['pagination_selector'] = None  # 由 ActAgent 动态发现
        except Exception:
            features['pagination_type'] = 'url' if features.get('has_pagination') else 'none'
            features['pagination_next_url'] = None
            features['pagination_selector'] = None

        # 7. LLM 深度分析（如有 LLM 客户端）
        if llm_client:
            try:
                deep_analysis = await self._llm_analyze(html, spec, llm_client)
                # 只补充，不覆盖已有的核心字段
                for key, value in deep_analysis.items():
                    if key not in features or not features[key]:
                        features[key] = value
            except Exception as e:
                error_msg = str(e)
                print(f"LLM 分析失败: {error_msg}")
                # 记录降级
                degradation_info = self.degradation_tracker.record_degradation(
                    self.name, 'llm_analyze', error_msg
                )
                if degradation_info.get('should_warn'):
                    print(f"警告: {degradation_info['message']}")

        # 8. 检测反爬
        anti_bot_info = self._detect_anti_bot(html)

        result = {
            'success': True,
            'structure': features,
            'features': features,
            'anti_bot_detected': anti_bot_info.get('detected', False),
            'anti_bot_info': anti_bot_info,
            'html_snapshot': html[:50000],
            'screenshot': screenshot
        }

        # 反爬警告
        if anti_bot_info.get('detected'):
            print(f"[SenseAgent] 检测到反爬机制: {anti_bot_info.get('type', 'unknown')}")
            if anti_bot_info.get('handling_suggestions'):
                print(f"  处理建议: {anti_bot_info['handling_suggestions'][:2]}")

        # 添加降级信息
        if degradation_info:
            result['degradation'] = degradation_info

        return result

    async def _wait_for_spa_render(self, browser, initial_html: str, max_wait: float = 10.0) -> str:
        """
        SPA 智能等待 - 等待 JS 渲染完成

        策略：
        1. 等待 networkidle
        2. 轮询 DOM 变化，直到稳定（连续 2 次变化 < 100 字符）
        3. 返回渲染后的完整 HTML
        """
        import asyncio

        # 第一步：等待 networkidle
        try:
            await browser.page.wait_for_load_state('networkidle', timeout=8000)
        except Exception:
            pass

        # 第二步：轮询 DOM 变化，直到稳定
        # DOM_STABILITY_THRESHOLD: 认为 DOM 稳定的最大字符变化量
        DOM_STABILITY_THRESHOLD = 100
        # STABLE_CHECK_COUNT: 需要连续几次稳定才认为渲染完成
        STABLE_CHECK_COUNT = 2
        stable_count = 0
        prev_html = initial_html
        elapsed = 0.0
        interval = 0.5

        while elapsed < max_wait:
            await asyncio.sleep(interval)
            elapsed += interval
            try:
                current_html = await browser.get_html()
            except Exception:
                break
            if abs(len(current_html) - len(prev_html)) < DOM_STABILITY_THRESHOLD:
                stable_count += 1
                if stable_count >= STABLE_CHECK_COUNT:
                    return current_html
            else:
                stable_count = 0
            prev_html = current_html

        # 超时后返回最后获取的 HTML
        try:
            return await browser.get_html()
        except Exception:
            return prev_html

    def _determine_pagination_type(self, pagination_info: Dict, features: Dict) -> str:
        """根据 parser 的分页检测结果 + FeatureDetector 的结果确定分页类型"""
        if pagination_info.get('next_url'):
            return 'url'
        if pagination_info.get('has_next'):
            return 'click'
        if features.get('has_pagination'):
            return 'url'  # FeatureDetector 检测到分页但 parser 没有具体 URL
        return 'none'

    async def _llm_analyze(self, html: str, spec: Any, llm_client) -> Dict:
        """使用 LLM 增强分析 - 推理任务，使用 DeepSeek"""
        goal = spec.get('goal', '未知') if spec else '未知'

        # 提取 <body> 内容，避免 <head> 占用过多字符
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            body = soup.find('body')
            html_snippet = str(body)[:5000] if body else html[:5000]
        except Exception:
            html_snippet = html[:5000]

        prompt = f"""分析以下 HTML 页面，提取关键信息：

目标：{goal}

HTML 片段：
```
{html_snippet}
```

请输出 JSON 格式：
{{
    "page_type": "list|detail|form|other",
    "main_content_selector": "CSS选择器",
    "pagination_type": "none|click|scroll|url",
    "pagination_selector": "CSS选择器或空",
    "data_fields": ["字段1", "字段2"],
    "special_handling": ["login", "captcha", "spa"]
}}"""
        try:
            # 推理任务 - 使用 reason() 方法（DeepSeek）
            if hasattr(llm_client, 'reason'):
                response = await llm_client.reason(prompt)
            else:
                response = await llm_client.chat([{"role": "user", "content": prompt}])
            # 安全解析 JSON
            return _safe_parse_json(response, "LLM 感知分析")
        except Exception as e:
            print(f"LLM 分析失败: {e}")
            return {}

    def _detect_anti_bot(self, html: str) -> Dict[str, Any]:
        """
        检测反爬机制并返回详细信息

        改进：不仅检测，还返回处理建议
        """
        anti_bot_info = {
            'detected': False,
            'type': None,
            'confidence': 0,
            'handling_suggestions': [],
            'requires_stealth': False
        }

        html_lower = html.lower()

        # Cloudflare Turnstile 检测
        if 'turnstile' in html_lower or 'cf-turnstile' in html_lower:
            anti_bot_info.update({
                'detected': True,
                'type': 'cloudflare_turnstile',
                'confidence': 0.9,
                'handling_suggestions': [
                    '使用隐形浏览器模式',
                    '等待 JavaScript 挑战完成',
                    '尝试人工验证后继续'
                ],
                'requires_stealth': True
            })

        # Cloudflare 一般检测
        elif 'cloudflare' in html_lower or 'cf-' in html_lower:
            anti_bot_info.update({
                'detected': True,
                'type': 'cloudflare',
                'confidence': 0.8,
                'handling_suggestions': [
                    '检查是否为浏览器验证页面',
                    '增加等待时间',
                    '使用隐形浏览器模式'
                ],
                'requires_stealth': True
            })

        # reCAPTCHA 检测
        elif 'recaptcha' in html_lower or 'g-recaptcha' in html_lower:
            anti_bot_info.update({
                'detected': True,
                'type': 'recaptcha',
                'confidence': 0.9,
                'handling_suggestions': [
                    '需要人工处理验证码',
                    '使用验证码识别服务',
                    '切换到其他数据源'
                ],
                'requires_stealth': False
            })

        # 通用挑战页面
        elif 'challenge' in html_lower and ('form' in html_lower or 'verify' in html_lower):
            anti_bot_info.update({
                'detected': True,
                'type': 'challenge_page',
                'confidence': 0.7,
                'handling_suggestions': [
                    '检查是否需要人工验证',
                    '尝试等待页面自动通过',
                    '检查 cookies 是否正确'
                ],
                'requires_stealth': True
            })

        return anti_bot_info

    def get_description(self) -> str:
        return "感知页面结构和特征，识别页面类型和反爬机制"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'browser' in context and context.get('browser') is not None


# ==================== 规划智能体 ====================

class PlanAgent(AgentInterface):
    """规划智能体 - 生成提取策略"""

    def __init__(self, llm_client=None, degradation_tracker: Optional[DegradationTracker] = None):
        super().__init__("PlanAgent", AgentCapability.PLAN)
        self.llm_client = llm_client
        self.degradation_tracker = degradation_tracker or DegradationTracker()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        单元内自动重试3次，采用不同策略
        """
        max_attempts = 3
        best_result = None

        for attempt in range(max_attempts):
            try:
                result = await self._attempt_generate(context, attempt)

                # 验证结果质量
                if result.get('success'):
                    selectors = result.get('selectors', {})
                    html = context.get('html_snapshot', '')

                    if self._quick_validate_selectors(selectors, html):
                        logger.info(f"[Plan] 尝试{attempt + 1}成功")
                        return result
                    else:
                        logger.warning(f"[Plan] 尝试{attempt + 1}选择器无效，继续重试")
                        if not best_result or len(result.get('selectors', {})) > len(best_result.get('selectors', {})):
                            best_result = result

            except Exception as e:
                logger.warning(f"[Plan] 尝试{attempt + 1}异常: {e}")
                continue

        # 所有尝试失败，返回最佳结果或失败
        if best_result:
            logger.info("[Plan] 返回历史最佳结果")
            return best_result
        return {'success': False, 'error': '所有尝试都失败'}

    async def _attempt_generate(self, context, attempt):
        """第N次尝试生成"""
        page_structure = context.get('page_structure', {})
        spec = context.get('spec')
        llm_client = context.get('llm_client') or self.llm_client
        html_sample = context.get('html_snapshot', '') or context.get('html', '')[:8000]

        if attempt == 0:
            logger.info("[Plan] 第1次尝试 - 正常生成")
            return await self._generate_normal(page_structure, spec, llm_client, html_sample)
        elif attempt == 1:
            logger.info("[Plan] 第2次尝试 - 保守策略")
            return await self._generate_conservative(page_structure, spec, llm_client, html_sample)
        elif attempt == 2:
            logger.info("[Plan] 第3次尝试 - 激进策略")
            return await self._generate_aggressive(page_structure, spec, llm_client, html_sample)

    def _quick_validate_selectors(self, selectors, html):
        """快速验证选择器是否能匹配到元素"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        valid_count = 0
        for field, selector in selectors.items():
            try:
                if soup.select(selector):
                    valid_count += 1
            except:
                pass

        return valid_count > 0  # 至少有一个有效

    async def _generate_normal(self, structure, spec, llm_client, html):
        """正常生成 - 保持原有逻辑"""
        if llm_client:
            try:
                strategy = await self._generate_with_llm(structure, spec, llm_client, html)
            except Exception as e:
                logger.warning(f"LLM策略生成失败: {e}")
                strategy = self._fallback_strategy(structure, spec)
        else:
            strategy = self._fallback_strategy(structure, spec)

        code = strategy.get('extraction_code')
        if not code:
            if llm_client and hasattr(llm_client, 'code'):
                try:
                    code = await self._generate_code_with_llm(strategy, spec, llm_client)
                except:
                    code = self._generate_code(strategy, spec)
            else:
                code = self._generate_code(strategy, spec)

        return {
            'success': True,
            'strategy': strategy,
            'selectors': strategy.get('selectors', {}),
            'generated_code': code
        }

    async def _generate_conservative(self, structure, spec, llm_client, html):
        """
        保守策略：更稳定、更宽泛的选择器

        核心：
        1. 使用属性包含匹配 [class*="xxx"]
        2. 避免复杂伪类
        3. 增加等待时间建议
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        selectors = {}
        container_selector = structure.get('main_content_selector', 'body')

        if spec and 'targets' in spec:
            for target in spec['targets']:
                for field in target.get('fields', []):
                    field_name = field.get('name', '')

                    # 保守选择器：使用属性包含匹配
                    conservative_selectors = [
                        f'[class*="{field_name}"]',
                        f'[id*="{field_name}"]',
                        f'[data-*="{field_name}"]',
                    ]

                    for sel in conservative_selectors:
                        try:
                            if soup.select(sel):
                                selectors[field_name] = sel
                                break
                        except:
                            continue

                    # 通用降级
                    if field_name not in selectors:
                        common_tags = {
                            'title': 'h1, h2, h3',
                            'name': 'h1, h2, h3, .name',
                            'price': '[class*="price"]',
                            'date': 'time, [class*="date"]',
                            'author': '[class*="author"]',
                            'url': 'a[href]',
                            'link': 'a[href]',
                            'image': 'img',
                            'content': 'p, article',
                        }
                        selectors[field_name] = common_tags.get(field_name.lower(), f'[class*="{field_name}"]')

        return {
            'success': True,
            'strategy_type': 'conservative',
            'selectors': selectors,
            'container_selector': container_selector,
            'wait_multiplier': 1.5,
            'pagination_strategy': structure.get('pagination_type', 'none'),
        }

    async def _generate_aggressive(self, structure, spec, llm_client, html):
        """
        激进策略：尝试完全不同的方法

        核心：
        1. 基于文本内容匹配
        2. 识别数据模式
        3. 最宽泛选择器
        """
        from bs4 import BeautifulSoup
        import re
        soup = BeautifulSoup(html, 'html.parser')

        selectors = {}

        # 识别可能的数据容器
        all_elements = soup.find_all(True)
        element_counts = {}
        for elem in all_elements:
            class_name = elem.get('class', [''])
            if class_name and class_name[0]:
                key = f".{class_name[0]}"
                element_counts[key] = element_counts.get(key, 0) + 1

        container_selector = 'body'
        if element_counts:
            sorted_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)
            for sel, count in sorted_elements[:5]:
                if count >= 3:
                    container_selector = sel
                    break

        # 激进选择器
        if spec and 'targets' in spec:
            for target in spec['targets']:
                for field in target.get('fields', []):
                    field_name = field.get('name', '')
                    description = field.get('description', '').lower()

                    if '标题' in description or 'title' in description:
                        selectors[field_name] = 'h1, h2, h3, [class*="title"], [class*="heading"]'
                    elif '价格' in description or 'price' in description:
                        selectors[field_name] = '[class*="price"], [class*="cost"]'
                    elif '作者' in description or 'author' in description:
                        selectors[field_name] = '[class*="author"], [class*="by"]'
                    elif '链接' in description or 'url' in description:
                        selectors[field_name] = 'a[href]'
                    elif '日期' in description or 'date' in description:
                        selectors[field_name] = 'time, [class*="date"], [class*="time"]'
                    else:
                        selectors[field_name] = f'*[class*="{field_name}"], *[id*="{field_name}"]'

        return {
            'success': True,
            'strategy_type': 'aggressive',
            'selectors': selectors,
            'container_selector': container_selector,
            'use_text_matching': True,
            'pagination_strategy': 'none',
        }

    async def _generate_with_llm(self, structure: Dict, spec: Any, llm_client, html_sample: str = '') -> Dict:
        """使用 LLM 生成策略 - 推理任务，使用 DeepSeek

        改进：
        1. 接收 HTML 样本，让 LLM 能看到真实 DOM 结构
        2. 支持目标驱动模式：字段可以只有 description，由 LLM 推断选择器
        """
        targets = spec.get('targets', []) if spec else []

        # 构建字段描述信息，包含 description 和 examples
        fields_with_descriptions = []
        for target in targets:
            for field in target.get('fields', []):
                field_info = {
                    'name': field.get('name'),
                    'selector': field.get('selector'),  # 可能为空
                    'description': field.get('description', ''),
                    'examples': field.get('examples', []),
                    'required': field.get('required', False),
                    'type': field.get('type', 'text')
                }
                fields_with_descriptions.append(field_info)

        # 提取 HTML 样本中的关键元素用于选择器推断
        html_context = self._extract_html_context(html_sample, targets)

        prompt = f"""基于页面结构分析和真实 HTML 内容，生成数据提取策略。

页面分析结果：
{json.dumps(structure, ensure_ascii=False, indent=2)}

需要提取的字段（目标驱动模式）：
{json.dumps(fields_with_descriptions, ensure_ascii=False, indent=2)}

HTML 样本（关键元素）：
```
{html_context}
```

重要规则：
1. 如果字段已有 selector，优先使用但需验证其在 HTML 中的有效性
2. 如果字段只有 description 没有 selector，根据语义描述推断最可能的 CSS 选择器
3. 利用 examples 示例值辅助理解字段的语义和格式
4. 基于 HTML 结构和字段语义进行智能匹配

【选择器语法要求 - 非常重要】：
- 只使用标准 CSS 选择器（如 div.title, a[href*="/pdf/"], .list-item）
- 不要使用 Scrapy/parsel 伪元素语法（::attr(), ::text, ::extract()）
- 不要使用伪类选择器（:contains, :has, :first-child 之外的）
- 对于属性提取（如 href, src），在 JSON 中用 "attribute": "href" 表示

请分析 HTML 结构，为每个目标字段生成最可能的 CSS 选择器。

输出 JSON 格式的提取策略：
{{
    "strategy_type": "css",
    "selectors": {{"field_name": "标准CSS选择器"}},
    "attributes": {{"field_name": "要提取的属性名（如href,text）"}},
    "container_selector": "数据容器选择器",
    "pagination_strategy": "none|click|scroll|url",
    "pagination_selector": "下一页按钮选择器",
    "estimated_items": 100,
    "selector_confidence": {{"field_name": "high|medium|low"}},
    "selector_reasoning": {{"field_name": "推断依据说明"}}
}}"""
        # 推理任务 - 使用 reason() 方法（DeepSeek）
        if hasattr(llm_client, 'reason'):
            response = await llm_client.reason(prompt)
        else:
            response = await llm_client.chat([{"role": "user", "content": prompt}])
        # 解析并返回
        try:
            return _safe_parse_json(response, "LLM 策略生成")
        except:
            return self._fallback_strategy(structure, spec)

    def _extract_html_context(self, html: str, targets: List) -> str:
        """
        从 HTML 中提取与目标字段相关的上下文

        改进：
        1. 支持基于 description 的语义搜索
        2. 当字段没有 selector 时，根据描述智能查找相关元素
        """
        if not html:
            return "(无 HTML 内容)"

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        context_parts = []

        # 收集所有字段信息（包含 description）
        all_fields = []
        for target in targets:
            for field in target.get('fields', []):
                all_fields.append({
                    'name': field.get('name', ''),
                    'selector': field.get('selector', ''),
                    'description': field.get('description', ''),
                    'examples': field.get('examples', []),
                    'has_selector': bool(field.get('selector', ''))
                })

        # 1. 提取常见数据容器
        container_tags = ['article', 'div[class*="item"]', 'div[class*="card"]',
                         'div[class*="post"]', 'li', 'tr']
        for tag in container_tags:
            elements = soup.select(tag)[:3]  # 最多3个
            for elem in elements:
                text = elem.get_text(strip=True)[:200]
                if text and len(text) > 20:
                    context_parts.append(f"容器示例 ({tag}):\n{elem.prettify()[:500]}")
                    break

        # 2. 为每个字段查找相关元素
        for field in all_fields[:10]:  # 最多处理10个字段
            field_name = field['name']
            field_selector = field['selector']
            field_description = field['description']
            has_selector = field['has_selector']

            # 2a. 如果有 selector，提取对应元素
            if has_selector and field_selector:
                try:
                    matches = soup.select(field_selector)[:2]
                    if matches:
                        for match in matches:
                            context_parts.append(
                                f"字段 '{field_name}' (selector: {field_selector}):\n"
                                f"{match.prettify()[:400]}"
                            )
                        continue  # 找到了就跳过语义搜索
                except Exception:
                    pass  # 选择器无效，继续语义搜索

            # 2b. 无 selector 或选择器无效，进行语义搜索
            context_found = self._find_elements_by_semantic(
                soup, field_name, field_description, field.get('examples', [])
            )
            if context_found:
                context_parts.append(context_found)

        # 3. 提取标题链接模式（常见于新闻/产品）
        links = soup.select('h1 a, h2 a, h3 a, h4 a')[:3]
        for link in links:
            context_parts.append(f"标题链接模式:\n{link.prettify()[:300]}")

        # 4. 提取作者/时间模式
        meta_patterns = ['[class*="author"]', '[class*="date"]', '[class*="time"]', 'time']
        for pattern in meta_patterns:
            try:
                matches = soup.select(pattern)[:2]
                if matches:
                    for match in matches:
                        context_parts.append(f"元数据元素 ({pattern}):\n{match.prettify()[:200]}")
                        break
            except:
                pass

        # 合并并限制总长度
        result = '\n\n'.join(context_parts[:20])  # 最多20个片段
        return result[:8000] if result else html[:2000]  # 兜底返回部分原始 HTML

    def _find_elements_by_semantic(self, soup, field_name: str, description: str, examples: List) -> str:
        """
        基于字段描述语义搜索相关 HTML 元素

        当 Spec 字段没有 selector 时，根据描述推断可能的选择器
        """
        from bs4 import Tag

        # 构建搜索关键词
        keywords = []

        # 1. 从字段名提取关键词
        field_lower = field_name.lower().replace('_', ' ').replace('-', ' ')
        keywords.extend(field_lower.split())

        # 2. 从描述提取关键词
        if description:
            # 简单的关键词提取：移除常见停用词
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                         'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                         'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                         'through', 'during', 'before', 'after', 'above', 'below'}
            desc_words = description.lower().split()
            keywords.extend([w for w in desc_words if w not in stop_words and len(w) > 2])

        candidates = []
        seen_elements = set()

        # 3. 基于关键词搜索元素
        for keyword in keywords[:5]:  # 最多使用5个关键词
            # 尝试 class/id 包含关键词
            patterns = [
                f'[class*="{keyword}"]',
                f'[id*="{keyword}"]',
                f'[data-*="{keyword}"]',
            ]
            for pattern in patterns:
                try:
                    matches = soup.select(pattern)[:3]
                    for match in matches:
                        # 避免重复
                        elem_id = id(match)
                        if elem_id not in seen_elements:
                            seen_elements.add(elem_id)
                            text = match.get_text(strip=True)[:100]
                            candidates.append({
                                'element': match,
                                'keyword': keyword,
                                'pattern': pattern,
                                'text_preview': text
                            })
                except Exception:
                    pass

        # 4. 如果有示例值，尝试文本匹配
        for example in examples[:3]:
            if example and len(str(example)) > 2:
                # 搜索包含示例文本的元素
                for tag in soup.find_all(string=re.compile(re.escape(str(example)), re.I)):
                    if isinstance(tag, str):
                        parent = tag.parent
                    else:
                        parent = tag
                    if parent and id(parent) not in seen_elements:
                        seen_elements.add(id(parent))
                        candidates.append({
                            'element': parent,
                            'keyword': f'example:{example}',
                            'pattern': 'text_match',
                            'text_preview': str(example)
                        })

        # 5. 格式化输出
        if not candidates:
            return ""

        result_parts = [f"字段 '{field_name}' 候选元素 (语义搜索):"]
        if description:
            result_parts.append(f"  描述: {description}")
        if examples:
            result_parts.append(f"  示例: {examples[:3]}")

        for i, cand in enumerate(candidates[:4]):  # 最多展示4个候选
            elem = cand['element']
            prettify = elem.prettify()[:300] if isinstance(elem, Tag) else str(elem)[:300]
            result_parts.append(
                f"  候选{i+1} [关键词: {cand['keyword']}]:\n"
                f"    {prettify}"
            )

        return '\n'.join(result_parts)

    def _fallback_strategy(self, structure: Dict[str, Any], spec: Any) -> Dict[str, Any]:
        """
        降级策略

        改进：
        1. 支持 Spec 中有 selector 的字段
        2. 对无 selector 的字段，根据字段名生成默认选择器
        3. 包含字段描述信息帮助后续处理
        """
        selectors = {}
        selector_info = {}  # 记录选择器来源
        container_selector = structure.get('main_content_selector', 'body')

        if spec and 'targets' in spec:
            for target in spec['targets']:
                for field in target.get('fields', []):
                    field_name = field.get('name', '')
                    field_selector = field.get('selector', '')
                    field_description = field.get('description', '')

                    if field_name:
                        if field_selector:
                            # 使用 Spec 中定义的选择器
                            selectors[field_name] = field_selector
                            selector_info[field_name] = 'from_spec'
                        else:
                            # 无 selector，生成基于字段名的默认选择器
                            default_selector = self._generate_default_selector(field_name, field_description)
                            selectors[field_name] = default_selector
                            selector_info[field_name] = 'generated'

        return {
            'strategy_type': 'css',
            'selectors': selectors,
            'selector_info': selector_info,  # 新增：选择器来源信息
            'container_selector': container_selector,
            'pagination_strategy': structure.get('pagination_type', 'none'),
            'pagination_selector': structure.get('pagination_selector'),
            'estimated_items': structure.get('estimated_items', 10)
        }

    def _generate_default_selector(self, field_name: str, description: str = '') -> str:
        """
        根据字段名和描述生成默认 CSS 选择器

        用于目标驱动模式下，当字段没有指定 selector 时的降级策略
        """
        # 转换字段名为可能的 class/id 格式
        field_lower = field_name.lower()
        field_kebab = field_lower.replace('_', '-')  # snake_case -> kebab-case
        field_camel = ''.join(word.capitalize() for word in field_lower.split('_'))  # PascalCase

        # 常见字段名到选择器的映射
        common_mappings = {
            'title': 'h1, h2, h3, .title, [class*="title"]',
            'name': '.name, [class*="name"], h1, h2',
            'price': '.price, [class*="price"]',
            'date': 'time, .date, [class*="date"]',
            'author': '.author, [class*="author"]',
            'description': '.description, .desc, [class*="desc"], p',
            'content': '.content, .body, article, [class*="content"]',
            'image': 'img, [class*="image"], [class*="img"]',
            'url': 'a, [href]',
            'link': 'a, [href]',
            'id': '[id], [data-id], [class*="id"]',
            'summary': '.summary, .abstract, [class*="summary"]',
            'abstract': '.abstract, .summary, [class*="abstract"]',
            'tag': '.tag, [class*="tag"]',
            'category': '.category, [class*="category"]',
        }

        # 1. 检查常见映射
        if field_lower in common_mappings:
            return common_mappings[field_lower]

        # 2. 基于 description 关键词推断
        if description:
            desc_lower = description.lower()
            for keyword, selector in [
                ('标题', 'h1, h2, h3, [class*="title"]'),
                ('title', 'h1, h2, h3, [class*="title"]'),
                ('价格', '[class*="price"]'),
                ('price', '[class*="price"]'),
                ('作者', '[class*="author"]'),
                ('author', '[class*="author"]'),
                ('时间', 'time, [class*="date"], [class*="time"]'),
                ('date', 'time, [class*="date"]'),
                ('摘要', '.abstract, .summary, [class*="abstract"]'),
                ('abstract', '.abstract, .summary'),
                ('链接', 'a, [href]'),
                ('link', 'a, [href]'),
                ('图片', 'img, [class*="image"]'),
                ('image', 'img, [class*="image"]'),
            ]:
                if keyword in desc_lower:
                    return selector

        # 3. 基于字段名生成通用选择器
        return f'[class*="{field_lower}"], [id*="{field_lower}"], [class*="{field_kebab}"], [class*="{field_camel}"]'

    async def _generate_code_with_llm(self, strategy: Dict, spec: Any, llm_client) -> str:
        """使用 LLM 生成代码 - 编码任务，使用 GLM"""
        targets = spec.get('targets', []) if spec else []
        container_selector = strategy.get('container_selector', '.item')
        selectors = strategy.get('selectors', {})

        prompt = f"""生成 Python 数据提取代码。

目标字段：
{json.dumps(targets, ensure_ascii=False, indent=2)}

选择器配置：
- 容器选择器: {container_selector}
- 字段选择器: {json.dumps(selectors, ensure_ascii=False, indent=2)}

请生成完整的 Python 代码，使用 BeautifulSoup 进行数据提取。

代码要求：
1. 函数名 extract_data(html_content)
2. 返回列表，每个元素是字典
3. 使用 json.dumps 输出 JSON
4. 处理空值情况

只输出代码，不要解释。"""

        try:
            # 编码任务 - 使用 code() 方法（GLM）
            if hasattr(llm_client, 'code'):
                response = await llm_client.code(prompt)
            else:
                response = await llm_client.chat([{"role": "user", "content": prompt}])

            # 提取代码块
            if '```python' in response:
                code = response.split('```python')[1].split('```')[0]
            elif '```' in response:
                code = response.split('```')[1].split('```')[0]
            else:
                code = response

            return code.strip()
        except Exception as e:
            print(f"LLM 代码生成失败: {e}")
            return self._generate_code(strategy, spec)

    def _generate_code(self, strategy: Dict, spec: Any) -> str:
        """生成可执行的 Python 代码"""
        selectors = strategy.get('selectors', {})
        container_selector = strategy.get('container_selector', '.item') or '.item'

        code = '''from bs4 import BeautifulSoup
import json

def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []

    # 主容器选择器
    containers = soup.select('CONTAINER_SELECTOR')

    for container in containers:
        item = {}
'''
        # 根据选择器生成提取逻辑
        for field_name, field_selector in selectors.items():
            code += f'''
        # 提取 {field_name}
        {field_name}_elem = container.select_one('{field_selector}')
        item['{field_name}'] = {field_name}_elem.get_text(strip=True) if {field_name}_elem else ''
'''
        code += '''
        if any(item.values()):  # 只保留有内容的项
            results.append(item)

    return results

if __name__ == "__main__":
    import sys
    html = sys.stdin.read()
    result = extract_data(html)
    print(json.dumps(result, ensure_ascii=False))
'''
        return code.replace('CONTAINER_SELECTOR', container_selector)

    def get_description(self) -> str:
        return "基于页面结构和契约生成智能提取策略"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'spec' in context or 'page_structure' in context


# ==================== 执行智能体 ====================

class ExtractionMetrics:
    """提取质量指标"""

    def __init__(self):
        self.total_items = 0
        self.successful_items = 0
        self.failed_selectors: Dict[str, int] = {}  # 选择器 -> 失败次数
        self.missing_fields: Dict[str, int] = {}  # 字段名 -> 缺失次数
        self.empty_fields: Dict[str, int] = {}  # 字段名 -> 空值次数
        self.required_missing: Dict[str, int] = {}  # 必填字段缺失次数

    def record_selector_result(self, field_name: str, selector: str,
                                found: bool, has_value: bool, required: bool = False):
        """记录选择器匹配结果"""
        if not found:
            self.failed_selectors[selector] = self.failed_selectors.get(selector, 0) + 1
            self.missing_fields[field_name] = self.missing_fields.get(field_name, 0) + 1
            if required:
                self.required_missing[field_name] = self.required_missing.get(field_name, 0) + 1
        elif not has_value:
            self.empty_fields[field_name] = self.empty_fields.get(field_name, 0) + 1

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        return {
            'total_items': self.total_items,
            'successful_items': self.successful_items,
            'success_rate': self.successful_items / self.total_items if self.total_items > 0 else 0,
            'failed_selectors': self.failed_selectors,
            'missing_fields': self.missing_fields,
            'empty_fields': self.empty_fields,
            'required_missing': self.required_missing,
            'quality_score': self._calculate_quality()
        }

    def _calculate_quality(self) -> float:
        """计算质量分数"""
        if self.total_items == 0:
            return 0.0

        # 基础分数：成功提取的项目比例
        base_score = self.successful_items / self.total_items

        # 惩罚必填字段缺失
        required_penalty = 0
        if self.required_missing:
            total_required_missing = sum(self.required_missing.values())
            required_penalty = min(total_required_missing / (self.total_items + 1), 0.5)

        return max(0, base_score - required_penalty)


class SelectorValidator:
    """
    选择器预验证器

    在执行提取前验证选择器是否能匹配到元素，
    提前发现问题，避免无效迭代。
    """

    def __init__(self, html: str):
        from bs4 import BeautifulSoup
        self.soup = BeautifulSoup(html, 'html.parser')

    def validate_selectors(self, selectors: Dict[str, str],
                           container_selector: str = None) -> Dict[str, Any]:
        """
        验证选择器有效性

        Returns:
            {
                'valid': bool,  # 是否所有选择器都有效
                'results': {selector: {'found': int, 'sample': str}},
                'warnings': [str],
                'suggestions': {field: [str]}
            }
        """
        results = {}
        warnings = []
        suggestions = {}

        # 验证容器选择器
        if container_selector:
            containers = self.soup.select(container_selector)
            if not containers:
                warnings.append(f"容器选择器 '{container_selector}' 未匹配到任何元素")
                # 尝试推荐替代容器
                alt_containers = self._find_alternative_containers()
                suggestions['_container'] = alt_containers
            results['_container'] = {
                'selector': container_selector,
                'found': len(containers),
                'sample': str(containers[0])[:200] if containers else None
            }

        # 验证字段选择器
        for field_name, selector in selectors.items():
            if field_name.startswith('_'):
                continue

            try:
                elements = self.soup.select(selector)
                found = len(elements)

                results[field_name] = {
                    'selector': selector,
                    'found': found,
                    'sample': elements[0].get_text(strip=True)[:100] if elements else None
                }

                if found == 0:
                    warnings.append(f"选择器 '{selector}' (字段: {field_name}) 未匹配到任何元素")
                    # 尝试推荐替代选择器
                    alt_selectors = self._find_alternative_selectors(field_name)
                    if alt_selectors:
                        suggestions[field_name] = alt_selectors

            except Exception as e:
                results[field_name] = {
                    'selector': selector,
                    'found': 0,
                    'error': str(e)
                }
                warnings.append(f"选择器 '{selector}' 语法错误: {e}")

        # 计算整体有效性
        valid_count = sum(1 for r in results.values() if r.get('found', 0) > 0)
        total = len(results)
        all_valid = valid_count == total and total > 0

        return {
            'valid': all_valid,
            'results': results,
            'warnings': warnings,
            'suggestions': suggestions,
            'valid_count': valid_count,
            'total_count': total
        }

    def _find_alternative_containers(self) -> List[str]:
        """寻找替代容器选择器"""
        alternatives = []

        # 常见容器模式
        container_patterns = [
            'article', '.article', '.post', '.item', '.card',
            '.news-item', '.product', '[class*="item"]',
            '[class*="card"]', 'li', 'tr'
        ]

        for pattern in container_patterns:
            try:
                elements = self.soup.select(pattern)
                if len(elements) >= 3:  # 至少3个才可能是数据容器
                    alternatives.append(f"{pattern} (找到 {len(elements)} 个)")
                    if len(alternatives) >= 3:
                        break
            except:
                pass

        return alternatives

    def _find_alternative_selectors(self, field_name: str) -> List[str]:
        """为字段寻找替代选择器"""
        alternatives = []

        # 基于字段名猜测选择器
        field_lower = field_name.lower().replace('_', '-')
        patterns = [
            f'[class*="{field_lower}"]',
            f'[id*="{field_lower}"]',
            f'.{field_lower}',
            f'#{field_lower}',
            f'[data-{field_lower}]',
        ]

        for pattern in patterns:
            try:
                elements = self.soup.select(pattern)
                if elements:
                    alternatives.append(f"{pattern} (找到 {len(elements)} 个)")
                    if len(alternatives) >= 3:
                        break
            except:
                pass

        return alternatives

    def get_validation_report(self, validation_result: Dict) -> str:
        """生成人类可读的验证报告"""
        lines = ["选择器验证报告:", "-" * 40]

        for field, result in validation_result.get('results', {}).items():
            found = result.get('found', 0)
            status = "✓" if found > 0 else "✗"
            lines.append(f"  {status} {field}: '{result.get('selector')}' -> {found} 个元素")

        if validation_result.get('warnings'):
            lines.append("\n警告:")
            for warning in validation_result['warnings']:
                lines.append(f"  - {warning}")

        if validation_result.get('suggestions'):
            lines.append("\n建议:")
            for field, sugs in validation_result['suggestions'].items():
                lines.append(f"  {field}: {sugs[:3]}")

        return "\n".join(lines)


class ActAgent(AgentInterface):
    """执行智能体 - 执行提取操作"""

    def __init__(self):
        super().__init__("ActAgent", AgentCapability.ACT)

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        browser = context.get('browser')
        selectors = context.get('selectors', {})
        strategy = context.get('strategy', {})
        generated_code = context.get('generated_code')
        spec = context.get('spec')

        # 提取必填字段
        required_fields = set()
        if spec and 'targets' in spec:
            for target in spec['targets']:
                for field in target.get('fields', []):
                    if field.get('required', False):
                        required_fields.add(field.get('name'))

        # 1. 获取 HTML
        html = await browser.get_html()

        # 新增：预验证选择器
        validation_result = None
        if selectors:
            validator = SelectorValidator(html)
            validation_result = validator.validate_selectors(
                selectors,
                strategy.get('container_selector')
            )

            if not validation_result['valid']:
                print(f"[ActAgent] 选择器预验证警告:")
                for warning in validation_result['warnings'][:5]:
                    print(f"  - {warning}")

                # 如果有建议的替代选择器，更新 selectors
                if validation_result['suggestions']:
                    for field, alts in validation_result['suggestions'].items():
                        if field != '_container' and alts:
                            # 提取第一个建议的选择器（去掉注释部分）
                            alt_selector = alts[0].split(' (')[0]
                            print(f"  建议: {field} -> {alt_selector}")

        # 初始化指标追踪
        metrics = ExtractionMetrics()

        # 2. 执行提取
        if generated_code and strategy.get('strategy_type') == 'css':
            # 使用生成的代码提取
            extracted_data = await self._execute_code(generated_code, html)
        else:
            # 使用选择器直接提取
            extracted_data = await self._extract_with_selectors(
                browser, selectors, strategy, metrics, required_fields
            )

        # 3. 处理分页
        pagination_type = strategy.get('pagination_strategy', 'none')
        if pagination_type != 'none' and len(extracted_data) > 0:
            all_data = await self._handle_pagination(
                browser, extracted_data, strategy, selectors, metrics, required_fields
            )
            extracted_data = all_data

        # 获取指标
        metrics.total_items = len(extracted_data)
        metrics.successful_items = sum(1 for item in extracted_data if any(item.values()))

        result = {
            'success': True,
            'extracted_data': extracted_data,
            'count': len(extracted_data),
            'extraction_metrics': metrics.get_metrics()
        }

        # 添加选择器验证结果
        if validation_result:
            result['selector_validation'] = validation_result

        return result

    async def _extract_with_selectors(self, browser, selectors: Dict, strategy: Dict,
                                       metrics: Optional[ExtractionMetrics] = None,
                                       required_fields: Optional[set] = None) -> List[Dict]:
        """使用选择器提取数据"""
        from bs4 import BeautifulSoup
        html = await browser.get_html()
        soup = BeautifulSoup(html, 'html.parser')

        # 找到所有数据项容器
        container_selector = strategy.get('container_selector') or '.item'
        containers = soup.select(container_selector)

        results = []
        required_fields = required_fields or set()

        for container in containers:
            item = {}
            item_has_data = False

            for field_name, field_selector in selectors.items():
                if field_name.startswith('_'):
                    continue
                try:
                    elem = container.select_one(field_selector)

                    if metrics:
                        # 记录选择器匹配结果
                        found = elem is not None
                        has_value = found and bool(elem.get_text(strip=True))
                        is_required = field_name in required_fields
                        metrics.record_selector_result(field_name, field_selector,
                                                       found, has_value, is_required)

                    if elem:
                        text = elem.get_text(strip=True)
                        item[field_name] = text
                        if text:
                            item_has_data = True
                    else:
                        # 区分"元素不存在"和"元素为空"
                        item[field_name] = None  # None 表示元素不存在
                except Exception as e:
                    item[field_name] = None
                    if metrics:
                        metrics.failed_selectors[field_selector] = \
                            metrics.failed_selectors.get(field_selector, 0) + 1

            if item_has_data:  # 只保留有内容的项
                # 将 None 转换为空字符串以便兼容
                for key in item:
                    if item[key] is None:
                        item[key] = ''
                results.append(item)

        return results

    async def _execute_code(self, code: str, html: str) -> List[Dict]:
        """在沙箱中执行生成的代码（异步版本）"""
        import asyncio
        import json
        import tempfile
        import os

        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            script_path = f.name

        try:
            # 使用异步子进程执行
            proc = await asyncio.create_subprocess_exec(
                'python', script_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(html.encode('utf-8', errors='replace')),
                timeout=60
            )

            if proc.returncode == 0 and stdout:
                try:
                    return json.loads(stdout.decode('utf-8', errors='replace'))
                except json.JSONDecodeError as e:
                    print(f"JSON 解析失败: {e}")
                    return []
            else:
                print(f"代码执行失败: {stderr.decode('utf-8', errors='replace')}")
                return []
        except asyncio.TimeoutError:
            print(f"代码执行超时 (60秒)")
            return []
        except Exception as e:
            print(f"代码执行异常: {e}")
            return []
        finally:
            os.unlink(script_path)

    async def _handle_pagination(self, browser, initial_data, strategy, selectors,
                                  metrics: Optional[ExtractionMetrics] = None,
                                  required_fields: Optional[set] = None):
        """处理分页（改进版：支持去重和滚动检测）"""
        all_data = initial_data.copy()
        max_pages = strategy.get('max_pages', 5)
        pagination_type = strategy.get('pagination_strategy', 'click')
        next_selector = strategy.get('pagination_selector', 'a.next')

        # 用于去重 - 使用关键字段组合
        def get_item_key(item: Dict) -> str:
            """生成唯一键用于去重"""
            # 使用所有非空值组合作为键
            key_parts = [f"{k}:{v}" for k, v in sorted(item.items()) if v]
            return "|".join(key_parts) if key_parts else str(id(item))

        seen_keys = {get_item_key(item) for item in all_data}
        required_fields = required_fields or set()

        for page_num in range(max_pages - 1):
            try:
                prev_data_count = len(all_data)

                if pagination_type == 'click':
                    next_btn = await browser.page.query_selector(next_selector)
                    if next_btn:
                        await next_btn.click()
                        await browser.page.wait_for_load_state('networkidle')
                        await browser.page.wait_for_timeout(1000)  # 等待内容加载
                    else:
                        break
                elif pagination_type == 'scroll':
                    # 滚动前记录位置
                    prev_scroll_height = await browser.page.evaluate('document.body.scrollHeight')

                    await browser.scroll_to_bottom()
                    await browser.page.wait_for_timeout(1000)

                    # 检测是否到底（滚动后高度没变化）
                    new_scroll_height = await browser.page.evaluate('document.body.scrollHeight')
                    if new_scroll_height == prev_scroll_height:
                        print("已到达页面底部")
                        break

                elif pagination_type == 'url':
                    # URL 分页逻辑
                    current_url = await browser.get_current_url()
                    next_url = self._get_next_page_url(current_url, page_num + 2)
                    if next_url:
                        await browser.navigate(next_url)
                    else:
                        break

                # 提取新数据
                html = await browser.get_html()
                new_data = await self._extract_with_selectors(
                    browser, selectors, strategy, metrics, required_fields
                )

                # 去重添加
                for item in new_data:
                    item_key = get_item_key(item)
                    if item_key not in seen_keys:
                        all_data.append(item)
                        seen_keys.add(item_key)

                # 如果没有新数据，可能已到底
                if len(all_data) == prev_data_count:
                    print(f"分页第 {page_num + 2} 页无新数据，停止翻页")
                    break

            except Exception as e:
                print(f"分页处理失败 (第 {page_num + 2} 页): {e}")
                break

        return all_data

    def _get_next_page_url(self, current_url: str, next_page: int) -> Optional[str]:
        """
        生成下一页 URL

        支持:
        - ?page=N 格式
        - /page/N 格式
        - /N 格式
        """
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

        parsed = urlparse(current_url)

        # 尝试 ?page=N 格式
        query_params = parse_qs(parsed.query)
        if 'page' in query_params:
            query_params['page'] = [str(next_page)]
            new_query = urlencode(query_params, doseq=True)
            return urlunparse(parsed._replace(query=new_query))

        # 尝试 /page/N 或 /N 格式
        path_parts = parsed.path.rstrip('/').split('/')
        if path_parts and path_parts[-1].isdigit():
            # 替换最后的数字
            path_parts[-1] = str(next_page)
            new_path = '/'.join(path_parts)
            return urlunparse(parsed._replace(path=new_path))

        # 尝试添加 /page/N
        if 'page' in parsed.path.lower():
            # 已有 page，替换数字
            import re
            new_path = re.sub(r'/page/\d+', f'/page/{next_page}', parsed.path)
            return urlunparse(parsed._replace(path=new_path))

        return None

    def get_description(self) -> str:
        return "执行实际的数据提取操作"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        # 需要browser和至少一个提取参数
        browser_ok = 'browser' in context and context.get('browser') is not None
        has_params = 'selectors' in context or 'strategy' in context or 'generated_code' in context
        return browser_ok and has_params


# ==================== 验证智能体 ====================

class VerifyAgent(AgentInterface):
    """验证智能体 - 验证数据质量"""

    def __init__(self, verifier=None):
        super().__init__("VerifyAgent", AgentCapability.VERIFY)
        self.verifier = verifier

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        extracted_data = context.get('extracted_data', [])
        spec = context.get('spec')
        extraction_metrics = context.get('extraction_metrics', {})

        # 使用外部验证器或内置验证逻辑
        if self.verifier:
            try:
                verification_result = self.verifier.verify(extracted_data, context)
                quality_score = self._calculate_quality_from_result(verification_result)
            except Exception as e:
                print(f"验证器执行失败: {e}")
                quality_score = self._calculate_quality(extracted_data, spec)
                verification_result = {'status': 'partial', 'error': str(e)}
        else:
            # 内置验证逻辑
            quality_score = self._calculate_quality(extracted_data, spec)
            verification_result = self._build_verification_result(
                extracted_data, spec, quality_score, extraction_metrics
            )

        return {
            'success': True,
            'verification_result': verification_result,
            'valid_items': verification_result.get('valid_items', len(extracted_data)),
            'total_items': len(extracted_data),
            'quality_score': quality_score
        }

    def _calculate_quality(self, data: List, spec: Any) -> float:
        """计算质量分数"""
        if not data:
            return 0.0

        # 获取必填字段
        required_fields = set()
        if spec and 'targets' in spec:
            for target in spec['targets']:
                for field in target.get('fields', []):
                    if field.get('required', False):
                        required_fields.add(field.get('name'))

        if not required_fields:
            # 没有必填字段，检查数据非空率
            non_empty = sum(1 for item in data if any(item.values()))
            return non_empty / len(data) if data else 0.0

        # 检查必填字段完整性
        complete_count = 0
        for item in data:
            if all(item.get(f) for f in required_fields):
                complete_count += 1

        return complete_count / len(data) if data else 0.0

    def _calculate_quality_from_result(self, result: Dict) -> float:
        """从验证结果计算质量分数"""
        scores = result.get('scores', {})
        if scores:
            return sum(scores.values()) / len(scores)
        valid = result.get('valid_items', 0)
        total = result.get('total_items', 1)
        return valid / total if total > 0 else 0.0

    def _build_verification_result(self, data: List, spec: Any, quality_score: float,
                                    extraction_metrics: Optional[Dict] = None) -> Dict:
        """构建验证结果"""
        issues = []

        # 检查必填字段
        required_fields = []
        if spec and 'targets' in spec:
            for target in spec['targets']:
                for field in target.get('fields', []):
                    if field.get('required', False):
                        required_fields.append(field.get('name'))

        missing_count = 0
        for item in data:
            for field in required_fields:
                if not item.get(field):
                    missing_count += 1

        if missing_count > len(data) * 0.5:
            issues.append(f"超过50%的数据缺少必填字段")

        # 添加提取指标相关的问题
        if extraction_metrics:
            failed_selectors = extraction_metrics.get('failed_selectors', {})
            if failed_selectors:
                issues.append(f"部分选择器匹配失败: {list(failed_selectors.keys())[:3]}")

            required_missing = extraction_metrics.get('required_missing', {})
            if required_missing:
                issues.append(f"必填字段缺失: {dict(list(required_missing.items())[:3])}")

        result = {
            'status': 'passed' if quality_score >= 0.8 else 'partial' if quality_score >= 0.5 else 'failed',
            'total_items': len(data),
            'valid_items': int(len(data) * quality_score),
            'quality_score': quality_score,
            'issues': issues,
            'scores': {
                'completeness': quality_score,
                'consistency': self._check_consistency(data)
            }
        }

        # 包含提取指标
        if extraction_metrics:
            result['extraction_metrics'] = extraction_metrics

        return result

    def _check_consistency(self, data: List) -> float:
        """检查数据一致性"""
        if len(data) < 2:
            return 1.0

        first_keys = set(data[0].keys())
        inconsistent = 0
        for item in data[1:10]:  # 检查前10个
            if set(item.keys()) != first_keys:
                inconsistent += 1

        return 1.0 - (inconsistent / min(len(data) - 1, 9))

    def get_description(self) -> str:
        return "验证提取数据的质量和完整性"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'extracted_data' in context


# ==================== 决策智能体 ====================

class JudgeAgent(AgentInterface):
    """决策智能体 - 做出关键决策"""

    def __init__(self, llm_client=None, degradation_tracker: Optional[DegradationTracker] = None):
        super().__init__("JudgeAgent", AgentCapability.JUDGE)
        self.llm_client = llm_client
        self.degradation_tracker = degradation_tracker or DegradationTracker()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        quality_score = context.get('quality_score', 0)
        iteration = context.get('iteration', 0)
        max_iterations = context.get('max_iterations', 10)
        errors = context.get('errors', [])
        spec = context.get('spec')
        extracted_data_count = len(context.get('extracted_data', []))
        degradation_info = None

        # 1. 程序快速判断
        decision, reasoning = self._quick_decision(
            quality_score, iteration, max_iterations, errors, extracted_data_count
        )

        # 2. LLM 增强判断（如有 LLM 且决策不是 complete）
        llm_client = context.get('llm_client') or self.llm_client
        if llm_client and decision != 'complete':
            try:
                enhanced_decision = await self._llm_judge(context, llm_client)
                if enhanced_decision:
                    decision = enhanced_decision.get('decision', decision)
                    reasoning = enhanced_decision.get('reasoning', reasoning)
            except Exception as e:
                error_msg = str(e)
                print(f"LLM 决策失败: {error_msg}")
                # 记录降级
                degradation_info = self.degradation_tracker.record_degradation(
                    self.name, 'llm_judge', error_msg
                )
                if degradation_info.get('should_warn'):
                    print(f"警告: {degradation_info['message']}")

        result = {
            'success': True,
            'decision': decision,
            'reasoning': reasoning,
            'confidence': quality_score,
            'suggestions': self._get_suggestions(decision, context)
        }

        # 添加降级信息
        if degradation_info:
            result['degradation'] = degradation_info

        return result

    def _quick_decision(self, quality_score: float, iteration: int, max_iterations: int,
                        errors: List, data_count: int) -> Tuple[str, str]:
        """快速决策"""
        # 成功条件
        if quality_score >= 0.8 and data_count >= 5:
            return 'complete', f"质量分数 {quality_score:.2f} >= 0.8，数据量 {data_count}，任务完成"

        # 失败条件
        if iteration >= max_iterations:
            return 'terminate', f"已达到最大迭代次数 {max_iterations}"

        if len(errors) >= 5:
            return 'terminate', f"错误过多 ({len(errors)} 个错误)"

        # 可提升
        if quality_score >= 0.3:
            return 'reflect_and_retry', f"质量分数 {quality_score:.2f} 可提升，继续迭代 {iteration + 1}/{max_iterations}"

        return 'terminate', f"质量分数过低 {quality_score:.2f}"

    async def _llm_judge(self, context: Dict, llm_client) -> Optional[Dict]:
        """使用 LLM 增强决策 - 推理任务，使用 DeepSeek"""
        prompt = f"""分析爬取任务的执行情况，决定下一步行动：

质量分数：{context.get('quality_score', 0)}
迭代次数：{context.get('iteration', 0)}/{context.get('max_iterations', 10)}
错误列表：{context.get('errors', [])[:5]}
目标：{context.get('spec', {}).get('goal', '未知') if context.get('spec') else '未知'}
数据量：{len(context.get('extracted_data', []))}

可选决策：
- complete: 任务完成，质量达标
- reflect_and_retry: 反思并重试
- terminate: 终止任务

请输出 JSON：
{{"decision": "complete|reflect_and_retry|terminate", "reasoning": "原因"}}"""

        try:
            # 推理任务 - 使用 reason() 方法（DeepSeek）
            if hasattr(llm_client, 'reason'):
                response = await llm_client.reason(prompt)
            else:
                response = await llm_client.chat([{"role": "user", "content": prompt}])
            return _safe_parse_json(response, "LLM 决策分析")
        except Exception as e:
            print(f"LLM 决策失败: {e}")
            return None

    def _get_suggestions(self, decision: str, context: Dict) -> List[str]:
        """获取改进建议"""
        if decision == 'complete':
            return ["任务已完成，可以结束"]

        suggestions = []
        errors = context.get('errors', [])
        quality_score = context.get('quality_score', 0)

        if any('selector' in str(e).lower() for e in errors):
            suggestions.append("考虑重新分析页面结构，更新选择器")
        if any('timeout' in str(e).lower() for e in errors):
            suggestions.append("增加等待时间或使用更稳定的选择器")
        if quality_score < 0.5:
            suggestions.append("质量分数过低，检查目标字段是否正确")
        if len(context.get('extracted_data', [])) < 5:
            suggestions.append("数据量不足，检查分页或数据容器选择器")

        return suggestions if suggestions else ["继续优化"]

    def get_description(self) -> str:
        return "在多个选项间做出最优决策"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'quality_score' in context or 'iteration' in context


# ==================== 探索智能体 ====================

class ExploreAgent(AgentInterface):
    """探索智能体 - 探索页面链接和结构"""

    def __init__(self):
        super().__init__("ExploreAgent", AgentCapability.EXPLORE)

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        browser = context.get('browser')
        current_url = context.get('current_url', '')
        depth = context.get('depth', 0)
        max_depth = context.get('max_depth', 2)
        base_url = context.get('base_url', current_url)

        if depth >= max_depth:
            return {'success': True, 'links': [], 'message': '已达到最大探索深度'}

        # 1. 提取所有链接
        all_links = await self._extract_links(browser)

        # 2. 过滤相关链接
        relevant_links = self._filter_links(all_links, base_url, context)

        # 3. 分类链接
        categorized = self._categorize_links(relevant_links, context)

        return {
            'success': True,
            'links': relevant_links,
            'categorized': categorized,
            'count': len(relevant_links),
            'next_depth': depth + 1
        }

    async def _extract_links(self, browser) -> List[str]:
        """提取页面链接"""
        html = await browser.get_html()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            # 过滤无效链接
            if href and not href.startswith(('#', 'javascript:', 'mailto:')):
                links.append(href)

        return list(set(links))

    def _filter_links(self, links: List[str], base_url: str, context: Dict) -> List[str]:
        """过滤相关链接"""
        from urllib.parse import urljoin, urlparse

        filtered = []
        base_domain = urlparse(base_url).netloc

        for link in links:
            try:
                absolute = urljoin(base_url, link)
                parsed = urlparse(absolute)

                # 只保留同域名链接
                if parsed.netloc == base_domain:
                    filtered.append(absolute)
            except Exception:
                continue

        return filtered

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
        return "探索页面链接，发现新的数据源"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'browser' in context


# ==================== 反思智能体 ====================

class ReflectAgent(AgentInterface):
    """反思智能体 - 反思和优化策略"""

    def __init__(self, llm_client=None, degradation_tracker: Optional[DegradationTracker] = None):
        super().__init__("ReflectAgent", AgentCapability.REFLECT)
        self.llm_client = llm_client
        self.degradation_tracker = degradation_tracker or DegradationTracker()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        execution_history = context.get('execution_history', [])
        errors = context.get('errors', [])
        quality_score = context.get('quality_score', 0)
        spec = context.get('spec')
        llm_client = context.get('llm_client') or self.llm_client
        degradation_info = None

        # 获取提取指标（包含失败的选择器）
        extraction_metrics = context.get('extraction_metrics', {})
        failed_selectors = extraction_metrics.get('failed_selectors', {})
        required_missing = extraction_metrics.get('required_missing', {})

        # 获取 HTML 样本
        html_sample = context.get('html_snapshot', '') or context.get('html', '')[:6000]

        # 1. 分析错误模式（增强版：包含选择器失败详情）
        error_analysis = self._analyze_errors_enhanced(
            errors, failed_selectors, required_missing
        )

        # 2. 分析执行历史
        history_analysis = self._analyze_history(execution_history)

        # 3. 生成改进建议
        if llm_client:
            try:
                improvements = await self._llm_reflect(
                    error_analysis, history_analysis, spec, llm_client, html_sample, failed_selectors
                )
            except Exception as e:
                error_msg = str(e)
                print(f"LLM 反思失败: {error_msg}")
                improvements = self._fallback_improvements(error_analysis)
                # 记录降级
                degradation_info = self.degradation_tracker.record_degradation(
                    self.name, 'llm_reflect', error_msg
                )
                if degradation_info.get('should_warn'):
                    print(f"警告: {degradation_info['message']}")
        else:
            improvements = self._fallback_improvements(error_analysis)

        result = {
            'success': True,
            'analysis': {
                'error_patterns': error_analysis,
                'history_summary': history_analysis
            },
            'improvements': improvements,
            'suggested_action': improvements.get('action', 'retry'),
            'new_selectors': improvements.get('selectors', {}),
            'new_strategy': improvements.get('strategy', None)
        }

        # 添加降级信息
        if degradation_info:
            result['degradation'] = degradation_info

        return result

    def _analyze_errors(self, errors: List[str]) -> Dict[str, Any]:
        """分析错误模式"""
        patterns = {}
        for error in errors:
            # 分类错误
            error_str = str(error).lower()
            if 'selector' in error_str or 'element' in error_str:
                key = 'selector_error'
            elif 'timeout' in error_str:
                key = 'timeout_error'
            elif 'network' in error_str or 'connection' in error_str:
                key = 'network_error'
            elif 'anti' in error_str or 'captcha' in error_str or 'block' in error_str:
                key = 'anti_bot_error'
            else:
                key = 'unknown_error'

            patterns[key] = patterns.get(key, 0) + 1

        return {
            'patterns': patterns,
            'most_common': max(patterns.items(), key=lambda x: x[1])[0] if patterns else 'none',
            'total_errors': len(errors)
        }

    def _analyze_history(self, history: List) -> Dict[str, Any]:
        """分析执行历史"""
        if not history:
            return {'summary': '无历史记录'}

        # 处理 StateSnapshot 对象或普通字典
        def get_item(item, key):
            if hasattr(item, 'state'):
                # StateSnapshot 对象
                return item.state.get(key)
            else:
                # 普通字典
                return item.get(key)

        return {
            'total_attempts': len(history),
            'stages': [get_item(h, 'stage') for h in history if get_item(h, 'stage')],
            'last_stage': get_item(history[-1], 'stage') if history else None,
            'quality_trend': [get_item(h, 'quality_score') for h in history if get_item(h, 'quality_score')]
        }

    def _analyze_errors_enhanced(self, errors: List[str], failed_selectors: Dict,
                                  required_missing: Dict) -> Dict[str, Any]:
        """
        增强的错误分析：包含选择器失败详情

        这是改进反思深度的关键：让 LLM 知道哪些选择器失败了
        """
        # 基础错误模式分析
        patterns = {}
        for error in errors:
            error_str = str(error).lower()
            if 'selector' in error_str or 'element' in error_str:
                key = 'selector_error'
            elif 'timeout' in error_str:
                key = 'timeout_error'
            elif 'network' in error_str or 'connection' in error_str:
                key = 'network_error'
            elif 'anti' in error_str or 'captcha' in error_str or 'block' in error_str:
                key = 'anti_bot_error'
            else:
                key = 'unknown_error'
            patterns[key] = patterns.get(key, 0) + 1

        # 如果有选择器失败，优先标记为选择器错误
        if failed_selectors:
            patterns['selector_error'] = patterns.get('selector_error', 0) + sum(failed_selectors.values())

        # 构建详细的选择器失败报告
        selector_failure_report = []
        for selector, count in failed_selectors.items():
            selector_failure_report.append({
                'selector': selector,
                'failure_count': count,
                'reason': '元素未找到或选择器无效'
            })

        # 构建必填字段缺失报告
        required_missing_report = []
        for field, count in required_missing.items():
            required_missing_report.append({
                'field': field,
                'missing_count': count
            })

        return {
            'patterns': patterns,
            'most_common': max(patterns.items(), key=lambda x: x[1])[0] if patterns else 'none',
            'total_errors': len(errors),
            'failed_selectors': selector_failure_report,  # 新增：失败选择器详情
            'required_fields_missing': required_missing_report  # 新增：必填字段缺失
        }

    async def _llm_reflect(self, error_analysis, history_analysis, spec, llm_client,
                           html_sample: str = '', failed_selectors: Dict = None) -> Dict:
        """
        使用 LLM 进行反思 - 推理任务，使用 DeepSeek

        改进：现在包含失败的选择器详情和 HTML 样本
        """
        goal = spec.get('goal', '未知') if spec else '未知'
        failed_selectors = failed_selectors or {}

        # 格式化失败选择器信息
        failed_selectors_text = ""
        if failed_selectors:
            failed_selectors_text = "\n失败的选择器详情：\n"
            for selector, count in failed_selectors.items():
                failed_selectors_text += f"  - '{selector}': 匹配失败 {count} 次\n"

        # 提取 HTML 上下文
        html_context = self._extract_html_context_for_reflection(html_sample)

        prompt = f"""分析爬取任务失败的原因并生成具体改进建议：

目标：{goal}

错误模式统计：{json.dumps(error_analysis.get('patterns', {}), ensure_ascii=False)}
{failed_selectors_text}

质量趋势：{json.dumps(history_analysis.get('quality_trend', []), ensure_ascii=False)}

HTML 样本（关键元素）：
```
{html_context}
```

请基于以上信息，特别是失败的选择器和 HTML 结构，生成具体的改进建议。

输出 JSON 格式：
{{
    "action": "retry|change_strategy|abort",
    "reasoning": "基于 HTML 分析的具体原因",
    "selectors": {{"field_name": "新的 CSS 选择器（基于 HTML 分析生成）"}},
    "container_selector": "新的容器选择器",
    "strategy": "具体策略描述",
    "precautions": ["注意事项"],
    "alternative_approaches": ["备选方案"]
}}"""

        try:
            # 推理任务 - 使用 reason() 方法（DeepSeek）
            if hasattr(llm_client, 'reason'):
                response = await llm_client.reason(prompt)
            else:
                response = await llm_client.chat([{"role": "user", "content": prompt}])
            return _safe_parse_json(response, "LLM 反思分析")
        except Exception as e:
            print(f"LLM 反思失败: {e}")
            return self._fallback_improvements(error_analysis)

    def _extract_html_context_for_reflection(self, html: str) -> str:
        """
        从 HTML 中提取对反思有用的上下文

        专注于可能包含目标数据的元素结构
        """
        if not html:
            return "(无 HTML 内容)"

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        context_parts = []

        # 1. 提取常见数据容器结构
        container_patterns = [
            'article', '.article', '.post', '.item', '.card',
            '.news-item', '.product', 'tr', 'li'
        ]
        for pattern in container_patterns:
            try:
                elements = soup.select(pattern)[:2]
                for elem in elements:
                    prettified = elem.prettify()[:400]
                    context_parts.append(f"数据容器 ({pattern}):\n{prettified}")
            except:
                pass

        # 2. 提取标题元素
        for tag in ['h1', 'h2', 'h3']:
            elements = soup.find_all(tag)[:2]
            for elem in elements:
                context_parts.append(f"标题元素 ({tag}):\n{elem.prettify()[:200]}")

        # 3. 提取链接模式
        links = soup.select('a[href]')[:5]
        for link in links:
            href = link.get('href', '')
            text = link.get_text(strip=True)[:50]
            if text:
                context_parts.append(f"链接: href='{href[:50]}' text='{text}'")

        result = '\n\n'.join(context_parts[:12])
        return result[:4000] if result else html[:1500]

    def _fallback_improvements(self, error_analysis: Dict) -> Dict:
        """降级改进建议"""
        most_common = error_analysis.get('most_common', 'unknown')

        suggestions = {
            'selector_error': {
                'action': 'retry',
                'reasoning': '选择器可能已更新，建议重新分析页面',
                'selectors': {},
                'strategy': 'reanalyze'
            },
            'timeout_error': {
                'action': 'retry',
                'reasoning': '增加等待时间',
                'selectors': {},
                'strategy': 'increase_timeout'
            },
            'network_error': {
                'action': 'retry',
                'reasoning': '网络问题，稍后重试',
                'selectors': {},
                'strategy': 'delay_retry'
            },
            'anti_bot_error': {
                'action': 'change_strategy',
                'reasoning': '检测到反爬机制，需要更换策略',
                'selectors': {},
                'strategy': 'use_stealth'
            },
            'unknown_error': {
                'action': 'abort',
                'reasoning': '未知错误，建议人工介入',
                'selectors': {},
                'strategy': None
            }
        }

        return suggestions.get(most_common, suggestions['unknown_error'])

    def get_description(self) -> str:
        return "分析失败原因，生成优化建议"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        # 可以在没有错误时也工作（用于分析改进）
        return True


# ==================== 智能体池 ====================

class AgentPool:
    """
    智能体池

    管理 7 种能力的智能体：
    - Sense: 感知页面结构
    - Plan: 规划提取策略
    - Act: 执行提取操作
    - Verify: 验证数据质量
    - Judge: 做出决策判断
    - Explore: 探索页面链接
    - Reflect: 反思和优化策略
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        # 共享的降级追踪器
        self.degradation_tracker = DegradationTracker()

        self.agents = {
            AgentCapability.SENSE: SenseAgent(self.degradation_tracker),
            AgentCapability.PLAN: PlanAgent(llm_client, self.degradation_tracker),
            AgentCapability.ACT: ActAgent(),
            AgentCapability.VERIFY: VerifyAgent(),
            AgentCapability.JUDGE: JudgeAgent(llm_client, self.degradation_tracker),
            AgentCapability.EXPLORE: ExploreAgent(),
            AgentCapability.REFLECT: ReflectAgent(llm_client, self.degradation_tracker)
        }

    def get_agent(self, capability: AgentCapability) -> Optional[AgentInterface]:
        """获取指定能力的智能体"""
        return self.agents.get(capability)

    async def execute_capability(self, capability: AgentCapability,
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """执行指定能力（异步版本）"""
        agent = self.get_agent(capability)
        if not agent:
            return {'success': False, 'error': f'Unknown capability: {capability}'}

        if not agent.can_handle(context):
            return {'success': False, 'error': f'Agent cannot handle this context'}

        return await agent.execute(context)

    def get_all_capabilities(self) -> List[AgentCapability]:
        """获取所有能力"""
        return list(self.agents.keys())

    def get_capability_description(self, capability: AgentCapability) -> str:
        """获取能力描述"""
        agent = self.get_agent(capability)
        return agent.get_description() if agent else ''

    def set_verifier(self, verifier):
        """设置验证器"""
        self.agents[AgentCapability.VERIFY] = VerifyAgent(verifier)

    def get_degradation_stats(self) -> Dict[str, Any]:
        """获取降级统计"""
        return self.degradation_tracker.get_stats()