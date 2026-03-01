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

    # 从修改后的main.py导入SelfCrawlingAgent
    from src.main import SelfCrawlingAgent

    # 临时创建一个测试规格文件
    spec = create_test_spec()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(spec, f)
        spec_path = f.name

    try:
        # 创建agent实例（这会触发容器检测）
        agent = SelfCrawlingAgent(spec_path)

        print(f"是否检测到容器环境: {agent.is_containerized}")
        print(f"容器配置: {agent.container_config}")
        print(f"使用的executor类型: {type(agent.executor.sandbox).__name__}")

        print("容器环境检测测试完成")

    finally:
        # 清理临时文件
        os.unlink(spec_path)

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

    # 测试非容器环境
    non_container_config = {
        'is_containerized': False,
        'sandbox_config': {'use_strict_validation': True}
    }

    executor_local = create_executor_for_environment(non_container_config)
    print(f"本地环境executor沙箱类型: {type(executor_local.sandbox).__name__}")

    print("executor创建测试完成")

def test_cli_options():
    """测试CLI参数"""
    print("\n测试CLI参数功能...")

    # 检查主程序是否包含docker相关参数
    import inspect
    import src.main

    # 查看main函数的源码中是否包含docker相关参数
    source = inspect.getsource(src.main.main)

    has_docker_params = (
        '--docker' in source and
        '--docker-image' in source and
        '--memory-limit' in source
    )

    print(f"CLI包含docker参数: {has_docker_params}")

    print("CLI参数测试完成")

if __name__ == "__main__":
    print("开始测试容器化功能...\n")

    test_container_detection()
    test_executor_creation()
    test_cli_options()

    print("\n所有测试完成!")