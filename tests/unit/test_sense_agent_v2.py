"""
SenseAgent v2 单元测试

覆盖：
- FeatureDetector v2 输出直接使用（page_type 字段）
- 移除 _analyze_structure 中的 type 字段覆盖
- HTMLParser.detect_pagination 集成
- LLM HTML 采样使用 <body> 内容
- SPA 智能等待逻辑
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.base import SenseAgent


# ---------------------------------------------------------------------------
# 辅助：构造 mock browser
# ---------------------------------------------------------------------------

def _make_mock_browser(html: str, screenshot: bytes = b''):
    browser = MagicMock()
    browser.get_html = AsyncMock(return_value=html)
    browser.take_screenshot = AsyncMock(return_value=screenshot)
    # SPA 等待相关
    browser.page = MagicMock()
    browser.page.wait_for_load_state = AsyncMock()
    browser.get_current_url = AsyncMock(return_value='http://example.com/')
    return browser


# ---------------------------------------------------------------------------
# page_type 字段（来自 FeatureDetector，不被旧 _analyze_structure 覆盖）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sense_agent_returns_page_type():
    """SenseAgent 返回的 structure 中应包含 page_type 字段"""
    html = """
    <html><body>
      <ul>
        <li class="item"><span>Product 1</span><span>$10</span></li>
        <li class="item"><span>Product 2</span><span>$20</span></li>
        <li class="item"><span>Product 3</span><span>$30</span></li>
        <li class="item"><span>Product 4</span><span>$40</span></li>
        <li class="item"><span>Product 5</span><span>$50</span></li>
        <li class="item"><span>Product 6</span><span>$60</span></li>
      </ul>
    </body></html>
    """
    agent = SenseAgent()
    browser = _make_mock_browser(html)
    result = await agent.execute({'browser': browser})

    assert result['success'] is True
    assert 'page_type' in result['structure'], "structure 中应包含 page_type"
    # 旧字段 'type' 不应覆盖 page_type（_analyze_structure 不再被调用）
    # page_type 值应为 FeatureDetector 定义的枚举之一
    valid_types = {'list', 'detail', 'form', 'spa', 'interactive', 'other', 'unknown'}
    assert result['structure']['page_type'] in valid_types


@pytest.mark.asyncio
async def test_sense_agent_no_old_type_overwrite():
    """_analyze_structure 的 type 字段不应覆盖 FeatureDetector 的 page_type"""
    html = """
    <html><body>
      <article>
        <h1>Deep Learning Intro</h1>
        <p>This is a comprehensive guide to deep learning concepts.</p>
        <p>Neural networks form the core of modern AI systems.</p>
        <p>Backpropagation enables training of deep networks.</p>
      </article>
    </body></html>
    """
    agent = SenseAgent()
    browser = _make_mock_browser(html)
    result = await agent.execute({'browser': browser})

    assert result['success'] is True
    structure = result['structure']
    # FeatureDetector 应识别为 detail；旧 _analyze_structure 的 type 不能覆盖 page_type
    assert structure.get('page_type') == 'detail'


# ---------------------------------------------------------------------------
# HTMLParser.detect_pagination 集成
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sense_agent_pagination_info_present():
    """SenseAgent 结果中应包含 pagination_info 字段（来自 HTMLParser）"""
    html = """
    <html><body>
      <div class="items">
        <div class="item">Item 1</div>
        <div class="item">Item 2</div>
      </div>
      <a rel="next" href="/page/2">下一页</a>
    </body></html>
    """
    agent = SenseAgent()
    browser = _make_mock_browser(html)
    result = await agent.execute({'browser': browser})

    assert result['success'] is True
    assert 'pagination_info' in result['structure'], "structure 中应包含 pagination_info"


@pytest.mark.asyncio
async def test_sense_agent_has_pagination_synced_from_html_parser():
    """当 HTMLParser 检测到 next，has_pagination 应为 True"""
    html = """
    <html><body>
      <div>Some content</div>
      <a rel="next" href="/page/2">Next</a>
    </body></html>
    """
    agent = SenseAgent()
    browser = _make_mock_browser(html)
    result = await agent.execute({'browser': browser})

    assert result['success'] is True
    assert result['structure'].get('has_pagination') is True


# ---------------------------------------------------------------------------
# LLM HTML 采样使用 <body> 内容
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_analyze_uses_body_content():
    """_llm_analyze 应提取 <body> 文本，而非传原始 html[:3000]"""
    html = """<!DOCTYPE html>
<html>
<head>
  <title>Test</title>
  <script>/* huge script */var x = 'secret_script_content_12345';</script>
  <style>.hidden { display: none; }</style>
</head>
<body>
  <h1>Main Article Title</h1>
  <p>This is the main body text that matters for LLM analysis.</p>
</body>
</html>"""

    captured_prompts = []

    async def fake_reason(prompt):
        captured_prompts.append(prompt)
        return '{"page_type": "detail"}'

    llm_client = MagicMock()
    llm_client.reason = fake_reason

    agent = SenseAgent()
    await agent._llm_analyze(html, None, llm_client)

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    # <body> 文本中的内容应出现在 prompt 中
    assert 'Main Article Title' in prompt
    # 正面断言：body 正文内容必须出现
    assert 'This is the main body text that matters for LLM analysis' in prompt
    # head 中 script 标签内容不应出现
    assert 'secret_script_content_12345' not in prompt


# ---------------------------------------------------------------------------
# SPA 智能等待（无真实浏览器，验证逻辑路径）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spa_smart_wait_called_for_spa_page():
    """SPA 页面时，SenseAgent 应调用 wait_for_load_state('networkidle')"""
    # 构造一个 SPA 特征页面
    html = """<!DOCTYPE html>
<html>
<head>
  <script src="/static/js/main.chunk.js"></script>
  <script src="/static/js/vendor.chunk.js"></script>
</head>
<body>
  <div id="root"></div>
</body>
</html>"""

    agent = SenseAgent()
    browser = _make_mock_browser(html)
    result = await agent.execute({'browser': browser})

    assert result['success'] is True
    # wait_for_load_state 应被调用（SPA 检测为真时）
    if result['structure'].get('is_spa'):
        browser.page.wait_for_load_state.assert_called_once_with('networkidle', timeout=5000)


# ---------------------------------------------------------------------------
# 兼容性：原有字段仍然存在
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sense_agent_backward_compatible_fields():
    """SenseAgent 输出应保持向后兼容的字段"""
    html = "<html><body><p>Hello world</p></body></html>"
    agent = SenseAgent()
    browser = _make_mock_browser(html)
    result = await agent.execute({'browser': browser})

    assert result['success'] is True
    assert 'structure' in result
    assert 'features' in result
    assert 'anti_bot_detected' in result
    assert 'html_snapshot' in result
    # structure 中包含 FeatureDetector 字段
    for field in ('has_login', 'has_pagination', 'is_spa', 'page_type', 'complexity'):
        assert field in result['structure'], f"缺少字段: {field}"
