#!/usr/bin/env python3
"""
验证 Self-Crawling Agent 的容器化功能
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

def validate_environment_detection():
    """验证环境检测功能"""
    print("1. 验证环境检测功能...")

    # 测试代码
    test_code = '''# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

from src.main import SelfCrawlingAgent

# 创建一个最小的SelfCrawlingAgent实例（不实际运行）
# 检测环境
agent = SelfCrawlingAgent()
print(f"Container environment: {agent.is_containerized}")
print(f"Container config: {agent.container_config}")
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_file = f.name

    try:
        result = subprocess.run([sys.executable, temp_file],
                              capture_output=True, text=True, timeout=30)
        print(f"  返回码: {result.returncode}")
        print(f"  输出: {result.stdout.strip()}")

        if result.returncode == 0:
            print("  PASS - 环境检测功能正常")
            return True
        else:
            print(f"  FAIL - 环境检测失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  FAIL - 环境检测超时")
        return False
    except Exception as e:
        print(f"  FAIL - 环境检测异常: {e}")
        return False
    finally:
        os.unlink(temp_file)


def validate_simplified_sandbox():
    """验证简化沙箱功能"""
    print("\n2. 验证简化沙箱功能...")

    # 测试代码
    test_code = '''# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

from src.executors.simplified_sandbox import SimplifiedSandbox

# 创建简化沙箱
sandbox = SimplifiedSandbox({'use_strict_validation': False, 'timeout_buffer': 10})

# 测试简单代码执行
code = "result = 2 + 3\\nprint(f'Result: {result}')"
result = sandbox.execute(code, timeout=10)

print(f"Execution successful: {result['success']}")
print(f"Stdout: {result['stdout']}")
print(f"Stderr: {result['stderr']}")
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_file = f.name

    try:
        result = subprocess.run([sys.executable, temp_file],
                              capture_output=True, text=True, timeout=30)
        print(f"  返回码: {result.returncode}")
        print(f"  输出: {result.stdout.strip()}")

        if result.returncode == 0 and "执行成功: True" in result.stdout:
            print("  PASS - 简化沙箱功能正常")
            return True
        else:
            print(f"  FAIL - 简化沙箱失败")
            return False
    except subprocess.TimeoutExpired:
        print("  FAIL - 简化沙箱超时")
        return False
    except Exception as e:
        print(f"  FAIL - 简化沙箱异常: {e}")
        return False
    finally:
        os.unlink(temp_file)


def validate_cli_parameters():
    """验证CLI参数增强"""
    print("\n3. 验证CLI参数增强...")

    try:
        # 运行主程序帮助信息，检查是否有新的docker相关参数
        result = subprocess.run([sys.executable, '-m', 'src.main', '--help'],
                              capture_output=True, text=True, timeout=15)
        print(f"  返回码: {result.returncode}")

        if result.returncode == 0:
            # 检查是否有docker相关的参数
            has_docker_params = (
                '--docker' in result.stdout and
                '--docker-image' in result.stdout and
                '--memory-limit' in result.stdout and
                '--cpu-shares' in result.stdout
            )

            if has_docker_params:
                print("  PASS - CLI参数增强正常")
                print(f"  发现docker相关参数: {result.stdout.count('--docker') > 0}")
                return True
            else:
                print("  FAIL - CLI参数增强缺失")
                print(f"  输出: {result.stdout}")
                return False
        else:
            print(f"  FAIL - CLI帮助命令失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  FAIL - CLI帮助命令超时")
        return False
    except Exception as e:
        print(f"  FAIL - CLI参数验证异常: {e}")
        return False


def validate_run_task_script():
    """验证run_task.py脚本"""
    print("\n4. 验证run_task.py脚本...")

    try:
        result = subprocess.run([sys.executable, 'run_task.py', '--help'],
                              capture_output=True, text=True, timeout=15)
        print(f"  返回码: {result.returncode}")

        if result.returncode == 0:
            # 检查关键功能参数
            has_local_param = '--local' in result.stdout
            has_docker_param = '--docker' in result.stdout
            has_image_param = '--image' in result.stdout

            print(f"  发现参数 - 本地模式: {has_local_param}, Docker: {has_docker_param}, 镜像: {has_image_param}")

            if has_local_param and has_docker_param and has_image_param:
                print("  PASS - run_task.py脚本功能正常")
                return True
            else:
                print("  FAIL - run_task.py脚本参数缺失")
                return False
        else:
            print(f"  FAIL - run_task.py帮助命令失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  FAIL - run_task.py帮助命令超时")
        return False
    except Exception as e:
        print(f"  FAIL - run_task.py验证异常: {e}")
        return False


def validate_container_manager():
    """验证容器管理器模块"""
    print("\n5. 验证容器管理器模块...")

    # 测试代码
    test_code = '''# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

try:
    from src.container_manager import ContainerizedTaskManager, TaskConfig
    print("Container manager imported successfully")

    # 验证TaskConfig类
    config = TaskConfig(
        task_id="test_task",
        spec_dict={"test": "data"},
        timeout=300,
        cpu_shares=512,
        memory_limit="1g"
    )
    print(f"TaskConfig created successfully: {config.task_id}")

    # 验证基本属性
    assert hasattr(config, 'task_id')
    assert hasattr(config, 'timeout')
    assert hasattr(config, 'cpu_shares')
    assert hasattr(config, 'memory_limit')

    print("TaskConfig properties validated")

except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Validation failed: {e}")
    sys.exit(1)
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_file = f.name

    try:
        result = subprocess.run([sys.executable, temp_file],
                              capture_output=True, text=True, timeout=30)
        print(f"  返回码: {result.returncode}")
        print(f"  输出: {result.stdout.strip()}")

        if result.returncode == 0:
            print("  PASS - 容器管理器模块功能正常")
            return True
        else:
            print(f"  FAIL - 容器管理器模块失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  FAIL - 容器管理器模块超时")
        return False
    except Exception as e:
        print(f"  FAIL - 容器管理器模块异常: {e}")
        return False
    finally:
        os.unlink(temp_file)


def validate_executor_dynamic_selection():
    """验证执行器动态选择"""
    print("\n6. 验证执行器动态选择...")

    # 测试代码
    test_code = '''# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

from src.executors.executor import create_executor_for_environment

# 测试非容器环境配置
non_container_config = {
    'is_containerized': False,
    'sandbox_config': {'use_strict_validation': True}
}

executor1 = create_executor_for_environment(non_container_config)
print(f"Non-container executor type: {type(executor1).__name__}")

# 测试容器环境配置
container_config = {
    'is_containerized': True,
    'sandbox_config': {'use_strict_validation': False}
}

executor2 = create_executor_for_environment(container_config)
print(f"Container executor type: {type(executor2).__name__}")

# 验证沙箱类型不同
sandbox1_type = type(executor1.sandbox).__name__
sandbox2_type = type(executor2.sandbox).__name__

print(f"Non-container sandbox: {sandbox1_type}")
print(f"Container sandbox: {sandbox2_type}")

if "SimplifiedSandbox" in sandbox2_type:
    print("Executor dynamic selection validated")
else:
    print("Executor dynamic selection may have issues")
    sys.exit(1)
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_file = f.name

    try:
        result = subprocess.run([sys.executable, temp_file],
                              capture_output=True, text=True, timeout=30)
        print(f"  返回码: {result.returncode}")
        print(f"  输出: {result.stdout.strip()}")

        if result.returncode == 0:
            print("  PASS - 执行器动态选择功能正常")
            return True
        else:
            print(f"  FAIL - 执行器动态选择失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  FAIL - 执行器动态选择超时")
        return False
    except Exception as e:
        print(f"  FAIL - 执行器动态选择异常: {e}")
        return False
    finally:
        os.unlink(temp_file)


def validate_file_integrity():
    """验证文件完整性"""
    print("\n7. 验证文件完整性...")

    required_files = [
        'src/main.py',
        'src/executors/simplified_sandbox.py',
        'src/container_manager.py',
        'run_task.py',
        'Dockerfile',
        'build.sh',
        'CONTAINERIZATION_README.md',
        'QUICKSTART.md'
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"  FAIL - 缺少文件: {missing_files}")
        return False
    else:
        print("  PASS - 所有必需文件存在")
        return True


def main():
    """Main validation function"""
    print("=" * 60)
    print("Self-Crawling Agent Containerization Validation")
    print("=" * 60)

    tests = [
        ("Environment Detection", validate_environment_detection),
        ("Simplified Sandbox", validate_simplified_sandbox),
        ("CLI Parameter Enhancement", validate_cli_parameters),
        ("run_task.py Script", validate_run_task_script),
        ("Container Manager Module", validate_container_manager),
        ("Executor Dynamic Selection", validate_executor_dynamic_selection),
        ("File Integrity", validate_file_integrity)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n  FAIL - {test_name} test exception: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("Validation Results Summary:")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1

    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\nSUCCESS! All validations passed. Containerization is working properly.")
        return 0
    else:
        print(f"\nWARNING: {total - passed} validation(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())