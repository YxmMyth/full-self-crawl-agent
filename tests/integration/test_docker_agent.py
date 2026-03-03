"""
test_docker_agent.py - Docker-as-Body 集成测试

测试 Docker 容器内的增强能力:
- execute_bash()
- execute_script()
- _extract_with_script()
- ReflectAgent 验证脚本
- PlanAgent script 策略升级
"""

import asyncio
import pytest

from src.utils.runtime import is_docker, get_runtime_info
from src.executors.executor import Sandbox


# ---- Phase 2: Sandbox 增强能力 ----

@pytest.mark.asyncio
async def test_sandbox_auto_detect():
    """Sandbox 在 Docker 内自动关闭 strict_mode"""
    from src.orchestrator import SelfCrawlingAgent
    agent = SelfCrawlingAgent()
    if is_docker():
        assert agent.sandbox.strict_mode is False
    else:
        assert agent.sandbox.strict_mode is True


@pytest.mark.asyncio
async def test_execute_bash_in_docker():
    """execute_bash() 在 Docker 内可执行"""
    sandbox = Sandbox(strict_mode=not is_docker())
    result = await sandbox.execute_bash('echo hello')
    if is_docker():
        assert result['success'] is True
        assert 'hello' in result['stdout']
    else:
        # 非 Docker 环境下 strict_mode=True 会拒绝
        assert result['success'] is False


@pytest.mark.asyncio
async def test_execute_bash_timeout():
    """execute_bash() 超时处理"""
    sandbox = Sandbox(strict_mode=False)
    result = await sandbox.execute_bash('sleep 10', timeout=1)
    assert result['success'] is False
    assert '超时' in result['stderr']


@pytest.mark.asyncio
async def test_execute_script_python():
    """execute_script() 执行 Python 脚本"""
    sandbox = Sandbox(strict_mode=False)
    code = 'import json; print(json.dumps({"status": "ok"}))'
    result = await sandbox.execute_script(code, language='python', timeout=10)
    assert result['success'] is True
    assert '"ok"' in result['stdout']


@pytest.mark.asyncio
async def test_execute_script_bash():
    """execute_script() 执行 bash 脚本"""
    sandbox = Sandbox(strict_mode=False)
    code = '#!/bin/bash\necho "bash works"'
    result = await sandbox.execute_script(code, language='bash', timeout=10)
    assert result['success'] is True
    assert 'bash works' in result['stdout']


@pytest.mark.asyncio
async def test_execute_script_strict_blocked():
    """strict_mode=True 非 Docker 环境下 execute_script 被阻止"""
    sandbox = Sandbox(strict_mode=True)
    result = await sandbox.execute_script('print(1)', language='python')
    if not is_docker():
        assert result['success'] is False


# ---- Phase 3: Script 提取路径 ----

@pytest.mark.asyncio
async def test_act_agent_extract_with_script():
    """ActAgent._extract_with_script() 基本功能"""
    from src.agents.act import ActAgent

    agent = ActAgent(sandbox=Sandbox(strict_mode=False))

    html = """<html><body>
    <div class="item"><h2>Title 1</h2><p>Desc 1</p></div>
    <div class="item"><h2>Title 2</h2><p>Desc 2</p></div>
    </body></html>"""

    # 不需要 LLM 时跳过
    result = await agent._extract_with_script(None, None, [], html)
    assert result == []  # 无 LLM 应返回空


@pytest.mark.asyncio
async def test_act_agent_execute_code():
    """ActAgent._execute_code() 激活测试"""
    from src.agents.act import ActAgent

    agent = ActAgent(sandbox=Sandbox(strict_mode=False))

    code = '''import sys, json
from bs4 import BeautifulSoup
html = sys.stdin.read()
soup = BeautifulSoup(html, 'html.parser')
items = soup.select('.item h2')
results = [{"title": i.get_text(strip=True)} for i in items]
print(json.dumps(results, ensure_ascii=False))
'''

    html = '<div class="item"><h2>Test Title</h2></div>'
    result = await agent._execute_code(code, html)
    assert len(result) == 1
    assert result[0]['title'] == 'Test Title'


# ---- Phase 3: PlanAgent script 策略 ----

def test_plan_agent_script_strategy_on_retry():
    """PlanAgent 在 Docker 模式重试时自动升级为 script 策略"""
    from src.agents.plan import PlanAgent

    plan = PlanAgent()
    structure = {'page_type': 'list', 'has_pagination': False}
    targets = [{'name': 'items', 'fields': [{'name': 'title'}]}]

    # retry_count=0 应仍为 css
    strategy = plan._determine_extraction_strategy(
        structure, targets, context={'retry_count': 0}
    )
    # 非 Docker 环境下不会升级
    if not is_docker():
        assert strategy['strategy_type'] == 'css'

    # retry_count=1 在 Docker 内应升级为 script
    strategy = plan._determine_extraction_strategy(
        structure, targets, context={'retry_count': 1}
    )
    if is_docker():
        assert strategy['strategy_type'] == 'script'


# ---- Phase 4: ReflectAgent 验证 ----

@pytest.mark.asyncio
async def test_reflect_verify_selectors():
    """ReflectAgent._verify_selectors_with_script() 验证选择器"""
    from src.agents.reflect import ReflectAgent

    agent = ReflectAgent()
    html = '<div class="item"><h2>Title</h2></div><div class="item"><h2>Title 2</h2></div>'

    # 有效选择器
    result = await agent._verify_selectors_with_script(
        {'title': '.item h2'}, html
    )
    assert result is True

    # 无效选择器
    result = await agent._verify_selectors_with_script(
        {'title': '.nonexistent'}, html
    )
    assert result is False


@pytest.mark.asyncio
async def test_reflect_verify_empty_input():
    """ReflectAgent 验证空输入"""
    from src.agents.reflect import ReflectAgent

    agent = ReflectAgent()
    assert await agent._verify_selectors_with_script({}, '') is False
    assert await agent._verify_selectors_with_script({'x': '.a'}, '') is False


# ---- Runtime detection ----

def test_runtime_info():
    """get_runtime_info() 返回正确结构"""
    info = get_runtime_info()
    assert 'is_docker' in info
    assert 'workspace' in info
    assert 'has_bash' in info
    assert 'has_curl' in info
    assert isinstance(info['is_docker'], bool)


def test_create_executor_for_environment():
    """create_executor_for_environment() 工厂函数"""
    from src.executors.executor import create_executor_for_environment

    # 容器配置
    executor = create_executor_for_environment({
        'is_containerized': True,
        'sandbox_config': {'use_strict_validation': False}
    })
    assert executor.sandbox.strict_mode is False

    # 非容器配置
    executor = create_executor_for_environment({
        'is_containerized': False,
        'sandbox_config': {}
    })
    assert executor.sandbox.strict_mode is True
