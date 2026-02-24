"""
浏览器工具 - 封装 Playwright
"""

from typing import Dict, Any, List, Optional, Tuple
from playwright.async_api import async_playwright, Browser, Page
import asyncio


class BrowserTool:
    """
    浏览器工具

    基于 Playwright 的浏览器自动化工具
    支持：
    - 页面导航
    - 截图
    - 获取 HTML
    - 等待元素
    - 网络请求拦截
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.context = None

    async def start(self) -> None:
        """启动浏览器"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox'
            ]
        )

        # 创建上下文（可配置用户代理、视口等）
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )

        self.page = await self.context.new_page()

    async def stop(self) -> None:
        """关闭浏览器"""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def navigate(self, url: str, wait_until: str = 'networkidle',
                      timeout: int = 30000) -> None:
        """
        导航到指定页面

        Args:
            url: 目标 URL
            wait_until: 等待条件（load/networkidle/domcontentloaded）
            timeout: 超时时间（毫秒）
        """
        await self.page.goto(url, wait_until=wait_until, timeout=timeout)

    async def get_html(self) -> str:
        """获取当前页面的 HTML"""
        return await self.page.content()

    async def take_screenshot(self, path: Optional[str] = None,
                             full_page: bool = False) -> bytes:
        """
        截取页面截图

        Args:
            path: 保存路径（可选）
            full_page: 是否截取完整页面

        Returns:
            截图二进制数据
        """
        if path:
            await self.page.screenshot(path=path, full_page=full_page)
            with open(path, 'rb') as f:
                return f.read()
        else:
            return await self.page.screenshot(full_page=full_page)

    async def wait_for_selector(self, selector: str, timeout: int = 5000) -> bool:
        """等待元素出现"""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except:
            return False

    async def wait_for_navigation(self, timeout: int = 30000) -> None:
        """等待导航完成"""
        await self.page.wait_for_load_state('networkidle', timeout=timeout)

    async def scroll_to_bottom(self, delay: float = 0.5) -> None:
        """滚动到页面底部"""
        await self.page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        await asyncio.sleep(delay)

    async def scroll_to_element(self, selector: str) -> None:
        """滚动到指定元素"""
        await self.page.locator(selector).scroll_into_view_if_needed()

    async def click(self, selector: str) -> None:
        """点击元素"""
        await self.page.click(selector)

    async def fill(self, selector: str, value: str) -> None:
        """填充表单字段"""
        await self.page.fill(selector, value)

    async def get_text(self, selector: str) -> str:
        """获取元素文本"""
        element = await self.page.query_selector(selector)
        if element:
            return await element.inner_text()
        return ''

    async def get_attribute(self, selector: str, attr: str) -> str:
        """获取元素属性"""
        element = await self.page.query_selector(selector)
        if element:
            return await element.get_attribute(attr)
        return ''

    async def get_all_elements(self, selector: str) -> List[Any]:
        """获取所有匹配元素"""
        return await self.page.query_selector_all(selector)

    async def evaluate(self, script: str, *args) -> Any:
        """在页面上下文中执行 JavaScript"""
        return await self.page.evaluate(script, args)

    async def set_extra_http_headers(self, headers: Dict[str, str]) -> None:
        """设置额外的 HTTP 头"""
        await self.context.set_extra_http_headers(headers)

    async def get_current_url(self) -> str:
        """获取当前 URL"""
        return self.page.url

    async def go_back(self) -> None:
        """返回上一页"""
        await self.page.go_back()

    async def go_forward(self) -> None:
        """前进到下一页"""
        await self.page.go_forward()

    async def reload(self) -> None:
        """重新加载页面"""
        await self.page.reload()

    async def intercept_requests(self, callback: callable) -> None:
        """拦截网络请求"""
        await self.page.route('**/*', callback)

    async def set_timeout(self, timeout: int) -> None:
        """设置默认超时时间"""
        self.page.set_default_timeout(timeout)

    def get_page(self) -> Page:
        """获取页面对象"""
        return self.page

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
