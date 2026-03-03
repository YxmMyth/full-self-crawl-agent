"""
test_containerization.py - 测试容器化功能
"""

import json
import tempfile
import os
from pathlib import Path

def create_test_spec():
    """创建一个测试规格文件"""
    spec = {
        "task_id": "test_task_container",
        "task_name": "Containerization Test",
        "target_url": "https://httpbin.org/html",
        "targets": [
            {
                "name": "title",
                "description": "Page title",
                "selector": "title",
                "type": "text"
            }
        ],
        "completion_criteria": {
            "min_items": 1,
            "quality_threshold": 0.8
        }
    }
    return spec

def test_container_detection():
    """测试容器环境检测功能"""
    print("测试容器环境检测功能...")

    from src.orchestrator import SelfCrawlingAgent

    try:
        agent = SelfCrawlingAgent()

        print(f"是否检测到容器环境: {agent.is_containerized}")
        print(f"容器配置: {agent.container_config}")
        print(f"使用的sandbox类型: {type(agent.sandbox).__name__}")

        # 验证属性类型
        assert isinstance(agent.is_containerized, bool)
        assert isinstance(agent.container_config, dict)
        assert 'is_containerized' in agent.container_config

        print("容器环境检测测试完成")

    except Exception as e:
        raise

def test_executor_creation():
    """测试根据环境创建合适的executor"""
    print("\n测试executor创建功能...")

    from src.executors.executor import create_executor_for_environment

    # 测试容器环境
    container_config = {
        'is_containerized': True,
        'sandbox_config': {'use_strict_validation': False}
    }

    executor_container = create_executor_for_environment(container_config)
    print(f"容器环境executor沙箱类型: {type(executor_container.sandbox).__name__}")
    assert executor_container.sandbox.strict_mode is False

    # 测试非容器环境
    non_container_config = {
        'is_containerized': False,
        'sandbox_config': {'use_strict_validation': True}
    }

    executor_local = create_executor_for_environment(non_container_config)
    print(f"本地环境executor沙箱类型: {type(executor_local.sandbox).__name__}")
    assert executor_local.sandbox.strict_mode is True

    print("executor创建测试完成")

def test_cli_options():
    """测试CLI参数中包含容器相关选项"""
    print("\n测试CLI参数功能...")

    import inspect
    import src.main

    source = inspect.getsource(src.main.main)

    # 检查容器模式参数存在
    has_container_param = '--container' in source

    print(f"CLI包含container参数: {has_container_param}")
    assert has_container_param, "CLI 应包含 --container 参数"

    print("CLI参数测试完成")

if __name__ == "__main__":
    print("开始测试容器化功能...\n")

    test_container_detection()
    test_executor_creation()
    test_cli_options()

    print("\n所有测试完成!")