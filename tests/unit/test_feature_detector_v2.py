"""
FeatureDetector v2 单元测试

覆盖：
- CSS Modules/Tailwind 列表页识别
- React SPA 识别
- "next generation" 文本不误判分页
- rel="next" 识别分页
- detail、form 页识别
- 输出中无旧的 type 冲突字段
"""

import pytest
from src.core.smart_router import FeatureDetector


@pytest.fixture
def detector():
    return FeatureDetector()


# ---------------------------------------------------------------------------
# 返回字段完整性
# ---------------------------------------------------------------------------

def test_analyze_returns_all_required_fields(detector):
    """analyze() 必须包含全部原有字段以及新增 container_info"""
    html = "<html><body><p>Hello world</p></body></html>"
    result = detector.analyze(html)

    required = {'has_login', 'has_pagination', 'is_spa', 'anti_bot_level',
                 'page_type', 'complexity', 'container_info'}
    assert required.issubset(result.keys()), f"缺少字段: {required - result.keys()}"


def test_analyze_no_old_type_field(detector):
    """返回值中不应存在旧的 'type' 冲突字段"""
    html = "<html><body><p>test</p></body></html>"
    result = detector.analyze(html)
    assert 'type' not in result, "不应存在旧的 'type' 字段"


def test_page_type_valid_values(detector):
    """page_type 必须是规定的枚举值之一"""
    valid_types = {'list', 'detail', 'form', 'spa', 'interactive', 'other'}
    html = "<html><body><p>simple page</p></body></html>"
    result = detector.analyze(html)
    assert result['page_type'] in valid_types, f"page_type 值非法: {result['page_type']}"


# ---------------------------------------------------------------------------
# CSS Modules / Tailwind 列表页识别
# ---------------------------------------------------------------------------

def test_list_page_css_modules(detector):
    """使用 CSS Modules hash 类名的列表页应被识别为 list"""
    items = "\n".join(
        f'<li class="item_a1b2c3"><span class="title_xYzQ">Product {i}</span>'
        f'<span class="price_9k2m">${ i * 10}</span></li>'
        for i in range(6)
    )
    html = f"""
    <html><body>
      <ul class="list_pQrS7">
        {items}
      </ul>
    </body></html>
    """
    result = detector.analyze(html)
    assert result['page_type'] == 'list', f"应识别为列表页，实际: {result['page_type']}"
    assert result['container_info']['found'] is True


def test_list_page_tailwind(detector):
    """使用 Tailwind utility 类名的列表页应被识别为 list"""
    items = "\n".join(
        f'<div class="flex flex-col p-4 border rounded"><h3 class="text-lg font-bold">Item {i}</h3>'
        f'<p class="text-gray-500">Description {i}</p></div>'
        for i in range(5)
    )
    html = f"""
    <html><body>
      <div class="grid grid-cols-3 gap-4">
        {items}
      </div>
    </body></html>
    """
    result = detector.analyze(html)
    assert result['page_type'] == 'list', f"应识别为列表页，实际: {result['page_type']}"


# ---------------------------------------------------------------------------
# React SPA 识别
# ---------------------------------------------------------------------------

def test_spa_react_detection(detector):
    """React SPA 页面应被识别为 spa"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <script src="/static/js/vendor.chunk.js"></script>
      <script src="/static/js/main.chunk.js"></script>
    </head>
    <body>
      <div id="root"></div>
      <script src="/static/js/runtime-main.chunk.js"></script>
    </body>
    </html>
    """
    result = detector.analyze(html)
    assert result['is_spa'] is True, "应检测为 SPA"
    assert result['page_type'] == 'spa'


def test_spa_vue_detection(detector):
    """Vue SPA 页面（含 data-v- 属性和 bundle）应被识别为 spa"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <script src="/js/chunk-vendors.bundle.js"></script>
      <script src="/js/app.chunk.js"></script>
    </head>
    <body>
      <div id="app" data-v-app></div>
    </body>
    </html>
    """
    result = detector.analyze(html)
    assert result['is_spa'] is True, "应检测为 SPA"


# ---------------------------------------------------------------------------
# 分页检测
# ---------------------------------------------------------------------------

def test_pagination_rel_next(detector):
    """rel="next" 链接应触发分页检测"""
    html = """
    <html><body>
      <div class="products">
        <div class="product"><h2>Item 1</h2></div>
      </div>
      <a rel="next" href="/page/2">下一页</a>
    </body></html>
    """
    result = detector.analyze(html)
    assert result['has_pagination'] is True, "rel=next 应识别为有分页"


def test_pagination_not_triggered_by_next_generation(detector):
    """'next generation' 文本不应误判为分页"""
    html = """
    <html><body>
      <h1>Next Generation Technology</h1>
      <p>This is the next generation of our product.</p>
      <p>We offer next generation solutions.</p>
    </body></html>
    """
    result = detector.analyze(html)
    assert result['has_pagination'] is False, "'next generation' 不应误判为分页"


def test_pagination_container_class(detector):
    """含 pagination class 的容器应触发分页检测"""
    html = """
    <html><body>
      <div class="pagination">
        <a href="?page=1">1</a>
        <a href="?page=2">2</a>
        <a href="?page=3">3</a>
      </div>
    </body></html>
    """
    result = detector.analyze(html)
    assert result['has_pagination'] is True


# ---------------------------------------------------------------------------
# Detail 页识别
# ---------------------------------------------------------------------------

def test_detail_page_with_article(detector):
    """包含 <article> 标签的页面应识别为 detail"""
    html = """
    <html><body>
      <article>
        <h1>深度学习入门</h1>
        <p>深度学习是机器学习的一个子集...</p>
        <p>神经网络由多个层次组成...</p>
        <p>反向传播算法是训练神经网络的核心...</p>
      </article>
    </body></html>
    """
    result = detector.analyze(html)
    assert result['page_type'] == 'detail', f"应识别为详情页，实际: {result['page_type']}"


def test_detail_page_h1_plus_paragraphs(detector):
    """单个 h1 + 多个 p 结构应识别为 detail"""
    html = """
    <html><body>
      <h1>产品详情</h1>
      <p>这是一款高质量的产品，拥有卓越的性能表现。</p>
      <p>采用最新技术制造，经过严格的品质检验。</p>
      <p>提供一年质保服务，支持全国联保。</p>
    </body></html>
    """
    result = detector.analyze(html)
    assert result['page_type'] == 'detail', f"应识别为详情页，实际: {result['page_type']}"


# ---------------------------------------------------------------------------
# Form 页识别
# ---------------------------------------------------------------------------

def test_form_page_detection(detector):
    """含有登录表单的页面应识别为 form"""
    html = """
    <html><body>
      <form action="/login" method="post">
        <input type="text" name="username" placeholder="用户名">
        <input type="password" name="password" placeholder="密码">
        <button type="submit">登录</button>
      </form>
    </body></html>
    """
    result = detector.analyze(html)
    assert result['has_login'] is True
    assert result['page_type'] == 'form', f"应识别为表单页，实际: {result['page_type']}"


# ---------------------------------------------------------------------------
# _clean_class_name
# ---------------------------------------------------------------------------

def test_clean_class_name_removes_hash(detector):
    """_clean_class_name 应去掉含数字的 hash 后缀，保留有意义的后缀"""
    assert detector._clean_class_name('item_a1b2c3') == 'item'
    assert detector._clean_class_name('title_xY2Q') == 'title'    # hash 含数字
    assert detector._clean_class_name('container') == 'container'
    # 有意义后缀（纯字母）不应被去掉
    assert detector._clean_class_name('item_main') == 'item_main'
    assert detector._clean_class_name('item_test') == 'item_test'


# ---------------------------------------------------------------------------
# container_info 结构
# ---------------------------------------------------------------------------

def test_container_info_structure(detector):
    """container_info 应包含 found/tag/count/similarity 字段；简单页面返回 found=False"""
    html = "<html><body><p>simple</p></body></html>"
    result = detector.analyze(html)
    ci = result['container_info']
    assert 'found' in ci
    assert 'tag' in ci
    assert 'count' in ci
    assert 'similarity' in ci
    # 简单单段落页面不应被检测为列表容器
    assert ci['found'] is False
    assert ci['tag'] is None
    assert ci['count'] == 0
    assert ci['similarity'] == 0.0
