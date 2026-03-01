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

        try:
            # 1. 获取页面 HTML 和截图
            html = await browser.get_html()
            screenshot = await browser.take_screenshot()

            # 2. 基础特征提取
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

            result = {
                'success': True,
                'html_snapshot': html,
                'screenshot': screenshot,
                'structure': {
                    'page_type': llm_analysis.get('page_type', features['page_type']),
                    'pagination_type': llm_analysis.get('pagination_type', features['pagination_type']),
                    'main_content_selector': llm_analysis.get('main_content_selector', features['main_content_selector']),
                    'complexity': features['complexity'],
                    'has_pagination': features['has_pagination'],
                    'content_selectors': features['content_selectors'],
                },
                'features': features,
                'llm_analysis': llm_analysis,
                'anti_bot_detected': features['anti_bot_detected']
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
            return {
                'success': False,
                'error': str(e),
                'degradation': degradation_info
            }

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

        # 分页检测
        pagination_patterns = [
            r'page', r'pager', r'pagination', r'翻页', r'下一页', r'next',
            r'class=".*page', r'id=".*page', r'pagination'
        ]

        has_pagination = False
        pagination_type = 'none'
        for pattern in pagination_patterns:
            if re.search(pattern, html, re.IGNORECASE):
                has_pagination = True
                pagination_type = 'standard'  # 可以进一步细化
                break

        # 主内容选择器
        main_selectors = [
            'main', '[role="main"]', '.main-content', '.content', '#content',
            '.post', '.article', '.entry-content'
        ]

        main_content_selector = ''
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
            'pagination_type': pagination_type,
            'main_content_selector': main_content_selector,
            'content_selectors': content_selectors,
            'has_pagination': has_pagination,
            'complexity': complexity,
            'anti_bot_detected': anti_bot_detected,
            'dom_size': len(html),
            'link_count': len(soup.find_all('a')),
            'image_count': len(soup.find_all('img')),
        }

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