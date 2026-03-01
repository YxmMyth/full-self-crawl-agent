"""
感知智能体 - SenseAgent
感知页面结构和内容特征
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import json
import re
import logging
from datetime import datetime

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
    安全解析 JSON 响应
    """
    if not response.strip():
        return {}

    # 尝试清理 Markdown 代码块
    if '```' in response:
        import re
        matches = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if matches:
            response = matches[0].strip()

    try:
        import json
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"{context} JSON解析失败: {e}")
        print(f"原始响应: {response[:500]}...")  # 仅打印前500字符
        return {}


class AgentInterface:
    """智能体接口"""

    def __init__(self, name: str, capability):
        self.name = name
        self.capability = capability

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行智能体任务"""
        raise NotImplementedError

    def can_handle(self, context: Dict[str, Any]) -> bool:
        """判断是否能处理当前上下文"""
        raise NotImplementedError

    def get_description(self) -> str:
        """获取智能体描述"""
        raise NotImplementedError


class SenseAgent(AgentInterface):
    """感知智能体 - 分析页面结构和提取特征"""

    def __init__(self, degradation_tracker: Optional[DegradationTracker] = None):
        super().__init__("SenseAgent", "sense")
        self.degradation_tracker = degradation_tracker or DegradationTracker()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行感知任务
        """
        browser = context.get('browser')
        spec = context.get('spec', {})
        llm_client = context.get('llm_client')
        degradation_info = None

        _empty_result = {
            'success': False,
            'html_snapshot': None,
            'screenshot': None,
            'structure': {
                'page_type': 'unknown',
                'pagination_type': 'none',
                'pagination_next_url': None,
                'main_content_selector': None,
                'estimated_items': 0,
                'complexity': 'simple',
                'has_pagination': False,
                'content_selectors': [],
            },
            'features': {},
            'anti_bot_detected': False,
            'anti_bot_info': {'detected': False},
        }

        try:
            # 1. 获取页面 HTML 和截图
            html = await browser.get_html()
            screenshot = await browser.take_screenshot()

            # 2. SPA 检测并等待渲染
            features = self._extract_features(html)
            if features.get('is_spa'):
                html = await self._wait_for_spa_render(browser)
                features = self._extract_features(html)

            # 3. 使用 LLM 深度分析（推理任务，使用 DeepSeek）
            llm_analysis = {}
            if llm_client:
                try:
                    llm_analysis = await self._llm_analyze(html, features, spec, llm_client)
                except Exception as e:
                    error_msg = str(e)
                    print(f"LLM 分析失败: {error_msg}")
                    # 记录降级
                    degradation_info = self.degradation_tracker.record_degradation(
                        self.name, 'llm_analyze', error_msg
                    )
                    if degradation_info.get('should_warn'):
                        print(f"警告: {degradation_info['message']}")

            # 4. 确定分页类型
            pagination_type = self._determine_pagination_type(
                features.get('pagination_info', {}),
                features
            )

            anti_bot_detected = features.get('anti_bot_detected', False)
            anti_bot_info = {'detected': anti_bot_detected}

            result = {
                'success': True,
                'html_snapshot': html,
                'screenshot': screenshot,
                'structure': {
                    'page_type': llm_analysis.get('page_type', features['page_type']),
                    'pagination_type': pagination_type,
                    'pagination_next_url': features.get('pagination_info', {}).get('next_url'),
                    'main_content_selector': features['main_content_selector'] or None,
                    'estimated_items': features.get('estimated_items', 0),
                    'complexity': features['complexity'],
                    'has_pagination': features['has_pagination'],
                    'content_selectors': features['content_selectors'],
                },
                'features': features,
                'llm_analysis': llm_analysis,
                'anti_bot_detected': anti_bot_detected,
                'anti_bot_info': anti_bot_info,
            }

            # 添加降级信息
            if degradation_info:
                result['degradation'] = degradation_info

            return result

        except Exception as e:
            error_msg = str(e)
            print(f"感知失败: {error_msg}")
            # 记录降级
            degradation_info = self.degradation_tracker.record_degradation(
                self.name, 'execute', error_msg
            )
            result = dict(_empty_result)
            result['error'] = str(e)
            result['degradation'] = degradation_info
            return result

    def _extract_features(self, html: str) -> Dict[str, Any]:
        """提取基础特征"""
        import re
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, 'html.parser')

        # 页面类型检测
        page_type = 'unknown'
        if soup.find('nav') or soup.find(class_=lambda x: x and 'nav' in x.lower()):
            page_type = 'list'
        elif soup.find('article') or soup.find(class_=lambda x: x and 'content' in x.lower()):
            page_type = 'detail'
        else:
            page_type = 'static'

        # SPA 检测
        scripts = soup.find_all('script', src=True)
        spa_patterns = [r'\.chunk\.js', r'vendor\.\w+\.js', r'app\.\w+\.js',
                        r'react', r'vue', r'angular', r'webpack']
        is_spa = any(
            re.search(p, s.get('src', ''), re.IGNORECASE)
            for s in scripts for p in spa_patterns
        )

        # 分页检测 - 使用更精确的逻辑（基于结构元素，不扫描文本内容）
        pagination_info = {'next_url': None, 'has_next': False}
        has_pagination = False

        # 优先检测 rel=next
        next_link = soup.find('a', rel=lambda r: r and 'next' in r)
        if next_link:
            pagination_info['next_url'] = next_link.get('href')
            pagination_info['has_next'] = True
            has_pagination = True
        else:
            # 检测分页容器元素（基于 class/id 属性，不检测文本）
            pagination_selectors = [
                '[class*="pager"]', '[class*="pagination"]',
                '[id*="pager"]', '[id*="pagination"]',
            ]
            for sel in pagination_selectors:
                if soup.select(sel):
                    has_pagination = True
                    pagination_info['has_next'] = True
                    break

        # 主内容选择器
        main_selectors = [
            'main', '[role="main"]', '.main-content', '.content', '#content',
            '.post', '.article', '.entry-content'
        ]

        main_content_selector = None
        for selector in main_selectors:
            if soup.select(selector):
                main_content_selector = selector
                break

        # 内容区域选择器
        content_selectors = []
        for tag in ['article', 'section', 'div']:
            elements = soup.find_all(tag)
            for elem in elements[:5]:  # 前5个
                classes = elem.get('class', [])
                if classes:
                    selector = f"{tag}.{'.'.join(classes)}"
                    if len(elem.get_text(strip=True)) > 50:  # 有实质内容
                        content_selectors.append(selector)

        # 容器/列表项检测
        container_info = {'found': False, 'selector': None, 'count': 0}
        estimated_items = 0
        for container_tag in ['ul', 'ol', 'table']:
            container = soup.find(container_tag)
            if container:
                items = container.find_all('li' if container_tag in ('ul', 'ol') else 'tr')
                if len(items) >= 3:
                    container_info = {
                        'found': True,
                        'selector': container_tag,
                        'count': len(items),
                    }
                    estimated_items = len(items)
                    if main_content_selector is None:
                        main_content_selector = container_tag
                    break

        # 复杂度评估
        complexity_score = len(soup.find_all(['script', 'style'])) + len(set(soup.get_text())) // 100
        complexity = 'simple' if complexity_score < 20 else ('medium' if complexity_score < 50 else 'complex')

        # 反爬检测
        anti_bot_indicators = [
            'captcha', 'turnstile', 'cloudflare', 'rate limit', 'access denied',
            'blocked', 'challenge', 'verify you are human'
        ]
        anti_bot_detected = any(indicator.lower() in html.lower() for indicator in anti_bot_indicators)

        return {
            'page_type': page_type,
            'is_spa': is_spa,
            'pagination_info': pagination_info,
            'has_pagination': has_pagination,
            'main_content_selector': main_content_selector,
            'estimated_items': estimated_items,
            'container_info': container_info,
            'content_selectors': content_selectors,
            'complexity': complexity,
            'anti_bot_detected': anti_bot_detected,
            'anti_bot_level': 'none' if not anti_bot_detected else 'medium',
            'dom_size': len(html),
            'link_count': len(soup.find_all('a')),
            'image_count': len(soup.find_all('img')),
        }

    def _determine_pagination_type(self, pagination_info: Dict[str, Any],
                                    features: Dict[str, Any]) -> str:
        """根据页面信号确定分页类型"""
        if pagination_info.get('next_url') or pagination_info.get('has_next'):
            return 'click'
        if features.get('has_pagination'):
            return 'url'
        return 'none'

    async def _wait_for_spa_render(self, browser) -> str:
        """等待 SPA 页面渲染完成并返回 HTML"""
        try:
            await browser.page.wait_for_load_state('networkidle')
        except Exception:
            try:
                await browser.page.wait_for_timeout(2000)
            except Exception:
                pass
        return await browser.get_html()

    async def _llm_analyze(self, html: str, features: Dict, spec: Dict, llm_client) -> Dict:
        """使用 LLM 深度分析页面结构"""
        goal = spec.get('goal', '未知') if spec else '未知'

        prompt = f"""分析网页结构，为爬取任务提供精确指导：

页面特征：{json.dumps(features, ensure_ascii=False)}
目标：{goal}

HTML样本（前3000字符）：
```
{html[:3000]}
```

请输出 JSON：
{{
    "page_type": "list|detail|static|unknown",
    "pagination_type": "standard|infinite_scroll|click_load|none",
    "main_content_selector": "CSS选择器定位主要内容区域",
    "item_selector": "CSS选择器定位单个项目（如果有列表）",
    "pagination_selector": "分页按钮选择器（如果适用）",
    "complexity_assessment": {{
        "js_heaviness": "low|medium|high",
        "dynamic_content": true|false,
        "spa_framework": "react|vue|angular|none|unknown",
        "interactivity_level": "low|medium|high"
    }},
    "recommendations": ["具体建议1", "具体建议2"]
}}"""

        # 使用推理方法（DeepSeek）
        if hasattr(llm_client, 'reason'):
            response = await llm_client.reason(prompt)
        else:
            response = await llm_client.chat([{"role": "user", "content": prompt}])

        return _safe_parse_json(response, "LLM 页面分析")

    def get_description(self) -> str:
        return "分析页面结构和特征，为后续提取提供指导"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'browser' in context