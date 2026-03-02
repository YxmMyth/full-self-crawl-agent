"""
ActAgent._sanitize_selector 单元测试

覆盖 LLM 生成的各种非法 CSS selector 格式的修复。
"""
import pytest


@pytest.fixture
def act_agent():
    from src.agents.base import ActAgent
    return ActAgent()


class TestSanitizeSelector:
    def test_at_attr_format(self, act_agent):
        """LLM 常生成 `a.card@href` 格式"""
        sel, attr = act_agent._sanitize_selector('a.mntl-card@href')
        assert sel == 'a.mntl-card'
        assert attr == 'href'

    def test_at_src_format(self, act_agent):
        sel, attr = act_agent._sanitize_selector('img.photo@src')
        assert sel == 'img.photo'
        assert attr == 'src'

    def test_at_data_attr(self, act_agent):
        sel, attr = act_agent._sanitize_selector('span.price@data-value')
        assert sel == 'span.price'
        assert attr == 'data-value'

    def test_scrapy_attr_format(self, act_agent):
        """Scrapy 风格 ::attr(name)"""
        sel, attr = act_agent._sanitize_selector('a.link::attr(href)')
        assert sel == 'a.link'
        assert attr == 'href'

    def test_normal_selector_unchanged(self, act_agent):
        sel, attr = act_agent._sanitize_selector('div.container > span.text')
        assert sel == 'div.container > span.text'
        assert attr is None

    def test_curly_braces_removed(self, act_agent):
        sel, attr = act_agent._sanitize_selector('div.item{color}')
        assert '{' not in sel
        assert '}' not in sel

    def test_empty_selector(self, act_agent):
        sel, attr = act_agent._sanitize_selector('')
        assert sel == ''
        assert attr is None

    def test_none_selector(self, act_agent):
        sel, attr = act_agent._sanitize_selector(None)
        assert sel == ''
        assert attr is None

    def test_at_without_attr(self, act_agent):
        """@ 后面为空"""
        sel, attr = act_agent._sanitize_selector('a.card@')
        assert sel == 'a.card'
        assert attr is None

    def test_multiple_at_signs(self, act_agent):
        """多个 @ 只拆最后一个"""
        sel, attr = act_agent._sanitize_selector('a@data-x@href')
        assert sel == 'a@data-x'
        assert attr == 'href'
