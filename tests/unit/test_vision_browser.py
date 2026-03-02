"""VisionBrowser 单元测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.tools.vision_browser import (
    VisionBrowser, PageBlocker, PageVisualAnalysis, VisualElement,
    BLOCKER_PATTERNS, DATA_REGION_SELECTORS,
)


# ── Fixtures ──

class FakeBrowser:
    """最小模拟浏览器"""

    def __init__(self, html="", screenshot=b"fake-png-bytes"):
        self._html = html
        self._screenshot = screenshot
        self.page = None  # 无真实 page 对象

    async def take_screenshot(self, full_page=True):
        return self._screenshot

    def get_html(self):
        return self._html

    async def dismiss_popups(self):
        return 1

    async def reload(self):
        pass


@pytest.fixture
def browser():
    return FakeBrowser(html="<html><body>hello world</body></html>")


@pytest.fixture
def vision(browser):
    return VisionBrowser(browser, llm_client=None)


# ── 阻断检测 ──

class TestBlockerDetection:
    def test_no_blocker_on_normal_text(self, vision):
        assert vision._detect_blocker_by_rules("hello world product list") == PageBlocker.NONE

    def test_captcha_detected(self, vision):
        text = "please complete the captcha verify you are human. hcaptcha challenge: are you a robot?"
        assert vision._detect_blocker_by_rules(text) == PageBlocker.CAPTCHA

    def test_login_wall_detected(self, vision):
        text = "please log in. sign in to continue or log in to continue. authentication required"
        assert vision._detect_blocker_by_rules(text) == PageBlocker.LOGIN_WALL

    def test_cookie_consent_detected(self, vision):
        text = "cookie consent notice. accept all cookies. read our cookie policy"
        assert vision._detect_blocker_by_rules(text) == PageBlocker.COOKIE_CONSENT

    def test_paywall_detected(self, vision):
        text = "this is a paywall. premium content for subscribers only. subscription required"
        assert vision._detect_blocker_by_rules(text) == PageBlocker.PAYWALL

    def test_rate_limit_detected(self, vision):
        text = "rate limit exceeded. too many requests, please slow down"
        assert vision._detect_blocker_by_rules(text) == PageBlocker.RATE_LIMIT

    def test_anti_bot_detected(self, vision):
        text = "access denied. forbidden. checking your browser before proceeding. cloudflare"
        assert vision._detect_blocker_by_rules(text) == PageBlocker.ANTI_BOT

    def test_single_keyword_not_enough(self, vision):
        """少于 3 个关键词不应触发（需 >= 3 个匹配）"""
        assert vision._detect_blocker_by_rules("captcha") == PageBlocker.NONE
        assert vision._detect_blocker_by_rules("captcha verify you are human") == PageBlocker.NONE

    def test_highest_score_wins(self, vision):
        """多种阻断同时出现时返回匹配度最高的"""
        text = "captcha verify you are human are you a robot recaptcha hcaptcha not a robot sign in to continue"
        result = vision._detect_blocker_by_rules(text)
        assert result == PageBlocker.CAPTCHA


# ── 页面类型推断 ──

class TestPageTypeInference:
    def test_captcha_page(self, vision):
        assert vision._infer_page_type("captcha required", []) == "captcha"

    def test_login_page(self, vision):
        assert vision._infer_page_type("please login to continue", []) == "login"

    def test_error_page(self, vision):
        assert vision._infer_page_type("404 page not found error", []) == "error"

    def test_search_page(self, vision):
        assert vision._infer_page_type("search results for python", []) == "search"

    def test_list_page(self, vision):
        regions = [VisualElement("r", "data_container")] * 3
        assert vision._infer_page_type("products", regions) == "list"

    def test_detail_page(self, vision):
        regions = [VisualElement("r", "data_container")]
        assert vision._infer_page_type("product detail", regions) == "detail"

    def test_form_page(self, vision):
        assert vision._infer_page_type("<form>submit</form>", []) == "form"

    def test_unknown_page(self, vision):
        assert vision._infer_page_type("random content", []) == "unknown"


# ── 分析合并 ──

class TestMergeAnalysis:
    def _make_base(self):
        return PageVisualAnalysis(
            page_type="unknown", blocker=PageBlocker.NONE,
            has_meaningful_content=False, screenshot_hash="abc",
        )

    def test_llm_page_type_overrides(self, vision):
        base = self._make_base()
        result = vision._merge_analysis(base, {"page_type": "list", "blocker": "none"})
        assert result.page_type == "list"

    def test_llm_unknown_page_type_no_override(self, vision):
        base = self._make_base()
        base.page_type = "search"
        result = vision._merge_analysis(base, {"page_type": "unknown"})
        assert result.page_type == "search"

    def test_llm_blocker_overrides(self, vision):
        base = self._make_base()
        result = vision._merge_analysis(base, {"blocker": "paywall"})
        assert result.blocker == PageBlocker.PAYWALL

    def test_llm_invalid_blocker_ignored(self, vision):
        base = self._make_base()
        result = vision._merge_analysis(base, {"blocker": "invalid_value"})
        assert result.blocker == PageBlocker.NONE

    def test_has_data_flag(self, vision):
        base = self._make_base()
        result = vision._merge_analysis(base, {"has_data": True})
        assert result.has_meaningful_content is True

    def test_layout_description(self, vision):
        base = self._make_base()
        result = vision._merge_analysis(base, {"layout": "two-column grid"})
        assert result.layout_description == "two-column grid"


# ── analyze_page 集成 ──

class TestAnalyzePage:
    @pytest.mark.asyncio
    async def test_normal_page_analysis(self):
        browser = FakeBrowser(html="<html><body>lots of content here</body></html>" * 20)
        v = VisionBrowser(browser, llm_client=None)
        analysis = await v.analyze_page(use_vision_llm=False)
        assert analysis.blocker == PageBlocker.NONE
        assert analysis.analysis_source == "rules"

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        browser = FakeBrowser(html="<html>test</html>", screenshot=b"same-bytes")
        v = VisionBrowser(browser, llm_client=None)
        a1 = await v.analyze_page(use_vision_llm=False)
        a2 = await v.analyze_page(use_vision_llm=False)
        assert a1 is a2  # 同一对象 = 缓存命中

    @pytest.mark.asyncio
    async def test_blocker_page(self):
        html = "captcha recaptcha verify you are human hcaptcha are you a robot not a robot"
        browser = FakeBrowser(html=html)
        v = VisionBrowser(browser, llm_client=None)
        analysis = await v.analyze_page(use_vision_llm=False)
        assert analysis.blocker == PageBlocker.CAPTCHA


# ── handle_blocker ──

class TestHandleBlocker:
    @pytest.mark.asyncio
    async def test_handle_none(self):
        v = VisionBrowser(FakeBrowser())
        assert await v.handle_blocker(PageBlocker.NONE) is True

    @pytest.mark.asyncio
    async def test_handle_cookie_consent(self):
        v = VisionBrowser(FakeBrowser())
        assert await v.handle_blocker(PageBlocker.COOKIE_CONSENT) is True

    @pytest.mark.asyncio
    async def test_handle_paywall_returns_false(self):
        v = VisionBrowser(FakeBrowser())
        assert await v.handle_blocker(PageBlocker.PAYWALL) is False

    @pytest.mark.asyncio
    async def test_handle_anti_bot_returns_false(self):
        v = VisionBrowser(FakeBrowser())
        assert await v.handle_blocker(PageBlocker.ANTI_BOT) is False

    @pytest.mark.asyncio
    async def test_handle_geo_block_returns_false(self):
        v = VisionBrowser(FakeBrowser())
        assert await v.handle_blocker(PageBlocker.GEO_BLOCK) is False


# ── detect_page_changes ──

class TestDetectPageChanges:
    @pytest.mark.asyncio
    async def test_no_change(self):
        browser = FakeBrowser(screenshot=b"same")
        v = VisionBrowser(browser)
        import hashlib
        h = hashlib.md5(b"same").hexdigest()
        result = await v.detect_page_changes(h)
        assert result['changed'] is False

    @pytest.mark.asyncio
    async def test_change_detected(self):
        browser = FakeBrowser(screenshot=b"new-content")
        v = VisionBrowser(browser)
        result = await v.detect_page_changes("old-hash")
        assert result['changed'] is True
        assert result['new_content_detected'] is True
