"""
浏览器工具 - 封装 Playwright
"""

from typing import Dict, Any, List, Optional, Tuple
from playwright.async_api import async_playwright, Browser, Page, Error as PlaywrightError
import asyncio
import functools
import random
import logging
import subprocess
import shutil

logger = logging.getLogger('browser')


def check_playwright_browsers() -> Tuple[bool, str]:
    """
    检查 Playwright 浏览器是否已安装

    Returns:
        (是否已安装, 安装说明)
    """
    try:
        # 检查 playwright 命令是否存在
        playwright_cmd = shutil.which('playwright')
        if playwright_cmd:
            # 尝试检查浏览器状态
            result = subprocess.run(
                [playwright_cmd, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.debug(f"Playwright version: {version}")

        # 检查浏览器目录
        import os
        from pathlib import Path

        # 常见的浏览器缓存位置
        browser_paths = [
            Path.home() / '.cache' / 'ms-playwright',
            Path.home() / 'AppData' / 'Local' / 'ms-playwright' if os.name == 'nt' else None,
        ]

        for path in browser_paths:
            if path and path.exists():
                # 检查是否有 chromium 目录
                chromium_dirs = list(path.glob('chromium-*'))
                if chromium_dirs:
                    logger.debug(f"Found Chromium at: {chromium_dirs[0]}")
                    return True, ""

        # 浏览器未找到
        install_cmd = "playwright install chromium"
        if os.name == 'nt':
            install_cmd = "python -m playwright install chromium"

        return False, f"""
Playwright 浏览器未安装。请运行以下命令安装：

    {install_cmd}

或者安装所有浏览器：

    playwright install

详细信息请参考: https://playwright.dev/python/docs/browsers
"""
    except Exception as e:
        logger.warning(f"检查 Playwright 浏览器时出错: {e}")
        return True, ""  # 假设已安装，让后续启动失败时再提示


def with_retry(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0):
    """
    重试装饰器，支持指数退避

    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(self, *args, **kwargs)
                except (PlaywrightError, asyncio.TimeoutError, ConnectionError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        # 指数退避 + 随机抖动
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                        logger.warning(f"操作失败 (尝试 {attempt + 1}/{max_retries}): {e}, {delay:.1f}秒后重试...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"操作失败，已达最大重试次数 ({max_retries}): {e}")
            raise last_error
        return wrapper
    return decorator


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

    def __init__(self, headless: bool = True, check_browsers: bool = True):
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.context = None

        # 启动时检查浏览器
        if check_browsers:
            installed, message = check_playwright_browsers()
            if not installed:
                logger.warning(message)

    async def start(self) -> None:
        """启动浏览器"""
        try:
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
            logger.info("浏览器启动成功")

        except Exception as e:
            logger.error(f"浏览器启动失败: {e}")

            # 检查是否是浏览器未安装
            if 'Executable doesn\'t exist' in str(e) or 'chromium' in str(e).lower():
                installed, message = check_playwright_browsers()
                if not installed:
                    raise RuntimeError(f"浏览器启动失败: {message}") from e

            raise

    async def stop(self) -> None:
        """关闭浏览器"""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    @with_retry(max_retries=3, base_delay=1.0, max_delay=15.0)
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

    @with_retry(max_retries=2, base_delay=0.5, max_delay=3.0)
    async def wait_for_selector(self, selector: str, timeout: int = 5000) -> bool:
        """等待元素出现"""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except PlaywrightError:
            return False

    @with_retry(max_retries=2, base_delay=1.0, max_delay=10.0)
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

    @with_retry(max_retries=3, base_delay=0.5, max_delay=5.0)
    async def click(self, selector: str) -> None:
        """点击元素"""
        await self.page.click(selector)

    @with_retry(max_retries=3, base_delay=0.5, max_delay=5.0)
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

    async def is_page_loaded(self) -> bool:
        """
        检查页面是否已加载完成

        Returns:
            页面是否加载完成
        """
        try:
            # 检查 document.readyState
            ready_state = await self.page.evaluate('document.readyState')
            return ready_state == 'complete'
        except Exception:
            return False

    async def wait_for_page_ready(self, timeout: int = 30000) -> bool:
        """
        等待页面就绪

        Args:
            timeout: 超时时间（毫秒）

        Returns:
            是否成功就绪
        """
        try:
            await self.page.wait_for_load_state('domcontentloaded', timeout=timeout)
            await self.page.wait_for_load_state('networkidle', timeout=timeout)
            return True
        except PlaywrightError:
            return False

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
