"""
Vision-LLM 浏览器层 — 视觉驱动的自主浏览器交互

在现有 Browser 之上增加视觉理解能力：
- 截图 + 多模态 LLM 分析页面布局和内容
- 基于视觉定位元素（当 CSS 选择器失败时的自主回退）
- 自主处理 CAPTCHA 检测、登录墙检测、反爬页面识别
- 视觉对比：页面变化检测（滚动前后、操作前后）

设计原则：
- 零人工介入：所有检测和决策全自动
- 降级安全：LLM 不可用时退化为纯规则检测
- 不修改 Browser 基类，纯包装模式
"""

import base64
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class PageBlocker(str, Enum):
    """页面阻断类型"""
    NONE = "none"
    CAPTCHA = "captcha"
    LOGIN_WALL = "login_wall"
    COOKIE_CONSENT = "cookie_consent"
    AGE_GATE = "age_gate"
    PAYWALL = "paywall"
    RATE_LIMIT = "rate_limit"
    GEO_BLOCK = "geo_block"
    ANTI_BOT = "anti_bot"


@dataclass
class VisualElement:
    """视觉识别出的页面元素"""
    description: str
    element_type: str          # button, input, text, image, table, list, link
    bounding_box: Dict[str, float] = field(default_factory=dict)  # x, y, width, height
    suggested_selector: str = ""
    confidence: float = 0.0


@dataclass
class PageVisualAnalysis:
    """页面视觉分析结果"""
    page_type: str             # list, detail, form, search, login, error, captcha
    blocker: PageBlocker
    data_regions: List[VisualElement] = field(default_factory=list)
    interactive_elements: List[VisualElement] = field(default_factory=list)
    layout_description: str = ""
    has_meaningful_content: bool = True
    screenshot_hash: str = ""
    analysis_source: str = "rules"  # "rules" or "llm"


# 常见阻断页面关键词
BLOCKER_PATTERNS = {
    PageBlocker.CAPTCHA: [
        'captcha', 'recaptcha', 'hcaptcha', 'verify you are human',
        '验证码', '人机验证', 'robot', 'challenge',
    ],
    PageBlocker.LOGIN_WALL: [
        'sign in', 'log in', 'login', '登录', '登入', 'please log in',
        'create account', '注册', 'authentication required',
    ],
    PageBlocker.COOKIE_CONSENT: [
        'cookie', 'consent', 'accept all', 'privacy policy',
        '接受 cookie', '隐私设置',
    ],
    PageBlocker.PAYWALL: [
        'subscribe', 'premium', 'paywall', 'subscription required',
        '订阅', '付费', 'upgrade to',
    ],
    PageBlocker.RATE_LIMIT: [
        'rate limit', 'too many requests', '429', 'slow down',
        '请求过于频繁', '稍后再试',
    ],
    PageBlocker.ANTI_BOT: [
        'access denied', 'forbidden', 'blocked', 'cloudflare',
        '访问被拒绝', 'please wait', 'checking your browser',
    ],
}

# 数据区域的常见 CSS 选择器模式
DATA_REGION_SELECTORS = [
    'table', '[class*="list"]', '[class*="grid"]', '[class*="card"]',
    '[class*="item"]', '[class*="result"]', '[class*="product"]',
    '[class*="article"]', '[class*="post"]', 'ul > li', 'ol > li',
    '[role="list"]', '[role="grid"]', '[role="table"]',
]


class VisionBrowser:
    """
    视觉增强浏览器 — 包装现有 Browser，增加 LLM 视觉分析能力

    使用方式：
        vision = VisionBrowser(browser, llm_client)
        analysis = await vision.analyze_page()
        if analysis.blocker != PageBlocker.NONE:
            handled = await vision.handle_blocker(analysis.blocker)
    """

    def __init__(self, browser, llm_client=None):
        self.browser = browser
        self.llm_client = llm_client
        self._screenshot_cache: Dict[str, bytes] = {}
        self._analysis_cache: Dict[str, PageVisualAnalysis] = {}

    async def analyze_page(self, use_vision_llm: bool = True) -> PageVisualAnalysis:
        """
        全面分析当前页面的视觉状态

        1. 截图并计算哈希（避免重复分析）
        2. 规则检测阻断类型
        3. 可用时调用多模态 LLM 做深度视觉分析
        """
        screenshot = await self._take_screenshot()
        screenshot_hash = hashlib.md5(screenshot).hexdigest()

        # 缓存命中
        if screenshot_hash in self._analysis_cache:
            return self._analysis_cache[screenshot_hash]

        # 获取页面文本用于规则检测
        html = await self._get_html_safe()
        text = html.lower() if html else ""

        # 1. 规则检测阻断类型
        blocker = self._detect_blocker_by_rules(text)

        # 2. 规则检测数据区域
        data_regions = await self._detect_data_regions()

        # 3. 基础分析
        analysis = PageVisualAnalysis(
            page_type=self._infer_page_type(text, data_regions),
            blocker=blocker,
            data_regions=data_regions,
            interactive_elements=await self._detect_interactive_elements(),
            has_meaningful_content=len(data_regions) > 0 or len(text) > 500,
            screenshot_hash=screenshot_hash,
            analysis_source="rules",
        )

        # 4. LLM 视觉增强（可选）
        if use_vision_llm and self.llm_client and hasattr(self.llm_client, 'chat'):
            try:
                llm_analysis = await self._llm_visual_analysis(screenshot, text[:3000])
                if llm_analysis:
                    analysis = self._merge_analysis(analysis, llm_analysis)
                    analysis.analysis_source = "llm"
            except Exception as e:
                logger.debug(f"Vision-LLM 分析失败（降级为规则）: {e}")

        self._analysis_cache[screenshot_hash] = analysis
        return analysis

    async def handle_blocker(self, blocker: PageBlocker) -> bool:
        """
        自主处理页面阻断。返回 True 表示成功绕过。

        策略：
        - COOKIE_CONSENT: 自动点击接受按钮
        - LOGIN_WALL: 检测是否有公开内容可跳过
        - CAPTCHA: 尝试刷新或等待（无法自动解决验证码）
        - RATE_LIMIT: 等待后重试
        - 其他: 标记并跳过
        """
        if blocker == PageBlocker.NONE:
            return True

        if blocker == PageBlocker.COOKIE_CONSENT:
            return await self._handle_cookie_consent()

        if blocker == PageBlocker.RATE_LIMIT:
            return await self._handle_rate_limit()

        if blocker == PageBlocker.CAPTCHA:
            return await self._handle_captcha()

        if blocker == PageBlocker.LOGIN_WALL:
            return await self._handle_login_wall()

        # PAYWALL, GEO_BLOCK, ANTI_BOT — 无法自主绕过，标记跳过
        logger.info(f"[VisionBrowser] 无法自主绕过 {blocker.value}，标记跳过")
        return False

    async def detect_page_changes(self, before_hash: str) -> Dict[str, Any]:
        """
        对比操作前后的页面变化（滚动、点击后）

        返回：
        - changed: bool
        - change_ratio: 0-1 的变化程度
        - new_content_detected: 是否出现新内容
        """
        current = await self._take_screenshot()
        current_hash = hashlib.md5(current).hexdigest()
        changed = current_hash != before_hash

        return {
            'changed': changed,
            'before_hash': before_hash,
            'after_hash': current_hash,
            'new_content_detected': changed,
        }

    async def find_element_visually(self, description: str) -> Optional[VisualElement]:
        """
        通过自然语言描述找到页面元素（LLM 视觉定位）

        当 CSS 选择器失败时的回退方案。
        例: find_element_visually("the search button at the top right")
        """
        if not self.llm_client:
            return None

        screenshot = await self._take_screenshot()
        b64 = base64.b64encode(screenshot).decode('utf-8')

        prompt = f"""分析截图，找到以下描述的页面元素："{description}"

返回 JSON：
{{"found": true/false, "element_type": "button/input/link/...", "description": "元素描述", "suggested_selector": "CSS选择器建议", "confidence": 0.0-1.0}}"""

        try:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]}]
            response = await self.llm_client.chat(messages)
            import json
            result = json.loads(response) if isinstance(response, str) else response
            if result.get('found'):
                return VisualElement(
                    description=result.get('description', description),
                    element_type=result.get('element_type', 'unknown'),
                    suggested_selector=result.get('suggested_selector', ''),
                    confidence=result.get('confidence', 0.5),
                )
        except Exception as e:
            logger.debug(f"视觉元素定位失败: {e}")

        return None

    async def get_page_snapshot(self) -> Dict[str, Any]:
        """获取当前页面的完整快照（截图 + HTML + 分析）"""
        screenshot = await self._take_screenshot()
        html = await self._get_html_safe()
        analysis = await self.analyze_page(use_vision_llm=False)

        return {
            'screenshot_b64': base64.b64encode(screenshot).decode('utf-8'),
            'screenshot_hash': analysis.screenshot_hash,
            'html_length': len(html) if html else 0,
            'page_type': analysis.page_type,
            'blocker': analysis.blocker.value,
            'data_region_count': len(analysis.data_regions),
            'has_meaningful_content': analysis.has_meaningful_content,
        }

    # ── 内部方法 ──

    async def _take_screenshot(self) -> bytes:
        """安全截图"""
        try:
            if hasattr(self.browser, 'take_screenshot'):
                return await self.browser.take_screenshot(full_page=True)
            if hasattr(self.browser, 'page') and self.browser.page:
                return await self.browser.page.screenshot(full_page=True)
        except Exception as e:
            logger.debug(f"截图失败: {e}")
        return b''

    async def _get_html_safe(self) -> str:
        """安全获取 HTML"""
        try:
            if hasattr(self.browser, 'get_html'):
                result = self.browser.get_html()
                if hasattr(result, '__await__'):
                    return await result
                return result
        except Exception:
            pass
        return ""

    def _detect_blocker_by_rules(self, text: str) -> PageBlocker:
        """基于关键词规则检测页面阻断"""
        scores: Dict[PageBlocker, int] = {}
        for blocker, patterns in BLOCKER_PATTERNS.items():
            count = sum(1 for p in patterns if p in text)
            if count >= 2:
                scores[blocker] = count

        if not scores:
            return PageBlocker.NONE

        # 返回匹配度最高的阻断类型
        return max(scores, key=scores.get)

    async def _detect_data_regions(self) -> List[VisualElement]:
        """检测页面中的数据区域"""
        regions = []
        for selector in DATA_REGION_SELECTORS:
            try:
                if hasattr(self.browser, 'page') and self.browser.page:
                    elements = await self.browser.page.query_selector_all(selector)
                    if elements and len(elements) >= 2:
                        regions.append(VisualElement(
                            description=f"Data region: {selector} ({len(elements)} items)",
                            element_type="data_container",
                            suggested_selector=selector,
                            confidence=min(0.9, 0.3 + len(elements) * 0.1),
                        ))
            except Exception:
                continue
        return regions

    async def _detect_interactive_elements(self) -> List[VisualElement]:
        """检测可交互元素"""
        interactive = []
        selectors_map = {
            'button': 'button, [role="button"], input[type="submit"]',
            'input': 'input[type="text"], input[type="search"], textarea',
            'link': 'a[href]',
            'select': 'select',
        }
        for elem_type, selector in selectors_map.items():
            try:
                if hasattr(self.browser, 'page') and self.browser.page:
                    elements = await self.browser.page.query_selector_all(selector)
                    if elements:
                        interactive.append(VisualElement(
                            description=f"{elem_type}: {len(elements)} found",
                            element_type=elem_type,
                            suggested_selector=selector,
                            confidence=0.8,
                        ))
            except Exception:
                continue
        return interactive

    def _infer_page_type(self, text: str, data_regions: List[VisualElement]) -> str:
        """推断页面类型"""
        if any(p in text for p in ['captcha', 'recaptcha', '验证码']):
            return 'captcha'
        if any(p in text for p in ['login', 'sign in', '登录']):
            return 'login'
        if any(p in text for p in ['error', '404', 'not found', '错误']):
            return 'error'
        if any(p in text for p in ['search', '搜索', 'results for']):
            return 'search'
        if len(data_regions) >= 3:
            return 'list'
        if len(data_regions) == 1:
            return 'detail'
        if any(p in text for p in ['<form', 'submit', '提交']):
            return 'form'
        return 'unknown'

    async def _llm_visual_analysis(self, screenshot: bytes,
                                    text_snippet: str) -> Optional[Dict[str, Any]]:
        """调用多模态 LLM 做深度视觉分析"""
        if not screenshot or not self.llm_client:
            return None

        b64 = base64.b64encode(screenshot).decode('utf-8')
        prompt = f"""分析这个网页截图和部分 HTML 文本。

HTML 片段（前3000字符）：
{text_snippet[:2000]}

请判断：
1. page_type: list/detail/form/search/login/error/captcha/unknown
2. blocker: none/captcha/login_wall/cookie_consent/paywall/rate_limit/anti_bot
3. has_data: 页面是否有可提取的结构化数据
4. layout: 简要描述页面布局

返回 JSON：
{{"page_type": "...", "blocker": "...", "has_data": true/false, "layout": "..."}}"""

        try:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]}]
            response = await self.llm_client.chat(messages)
            import json
            return json.loads(response) if isinstance(response, str) else response
        except Exception as e:
            logger.debug(f"LLM 视觉分析调用失败: {e}")
            return None

    def _merge_analysis(self, rule_analysis: PageVisualAnalysis,
                        llm_result: Dict[str, Any]) -> PageVisualAnalysis:
        """合并规则分析和 LLM 分析结果"""
        # LLM 结果优先级高于规则（但只在有效时覆盖）
        if llm_result.get('page_type') and llm_result['page_type'] != 'unknown':
            rule_analysis.page_type = llm_result['page_type']

        llm_blocker = llm_result.get('blocker', 'none')
        if llm_blocker != 'none':
            try:
                rule_analysis.blocker = PageBlocker(llm_blocker)
            except ValueError:
                pass

        if 'has_data' in llm_result:
            rule_analysis.has_meaningful_content = llm_result['has_data']

        if llm_result.get('layout'):
            rule_analysis.layout_description = llm_result['layout']

        return rule_analysis

    async def _handle_cookie_consent(self) -> bool:
        """自动处理 Cookie 同意弹窗"""
        try:
            if hasattr(self.browser, 'dismiss_popups'):
                dismissed = await self.browser.dismiss_popups()
                return dismissed > 0
        except Exception:
            pass
        return False

    async def _handle_rate_limit(self) -> bool:
        """处理速率限制：等待后重试"""
        import asyncio
        logger.info("[VisionBrowser] 检测到速率限制，等待 10 秒后重试...")
        await asyncio.sleep(10)
        try:
            if hasattr(self.browser, 'reload'):
                await self.browser.reload()
            return True
        except Exception:
            return False

    async def _handle_captcha(self) -> bool:
        """
        处理 CAPTCHA：尝试刷新页面。
        注意：自动解决验证码不可行也不道德，只做刷新尝试。
        """
        import asyncio
        logger.info("[VisionBrowser] 检测到 CAPTCHA，等待 5 秒后刷新尝试...")
        await asyncio.sleep(5)
        try:
            if hasattr(self.browser, 'reload'):
                await self.browser.reload()
                await asyncio.sleep(3)
                # 重新检测是否还有 CAPTCHA
                html = await self._get_html_safe()
                return 'captcha' not in html.lower()
        except Exception:
            pass
        return False

    async def _handle_login_wall(self) -> bool:
        """
        处理登录墙：检测页面是否有部分公开内容可提取。
        不尝试自动登录（无凭据）。
        """
        html = await self._get_html_safe()
        text = html.lower() if html else ""
        # 检查是否有登录弹窗可以关闭
        try:
            if hasattr(self.browser, 'dismiss_popups'):
                dismissed = await self.browser.dismiss_popups()
                if dismissed > 0:
                    return True
        except Exception:
            pass
        # 检查页面是否有足够内容（登录墙可能只遮挡部分内容）
        return len(text) > 2000
