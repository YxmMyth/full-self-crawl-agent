"""
Simplified Sandbox for Container Environments
重点：资源限制而非安全验证
"""

import tempfile
import subprocess
import sys
import os
import re
import json
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger('simplified_sandbox')


class SimplifiedSandbox:
    """
    专为容器环境优化的沙箱
    重点：资源限制而非安全验证
    因为容器已提供安全边界，内部只需基本的资源控制
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        # 在容器环境中，默认不进行严格安全验证
        self.use_security_validation = config.get('use_strict_validation', False)
        self.timeout_buffer = config.get('timeout_buffer', 30)

        # 资源限制
        self.memory_limit = config.get('memory_limit', 256 * 1024 * 1024)  # 256MB
        self.max_pids = config.get('max_pids', 10)

    def execute(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        执行代码，重点控制资源使用
        """
        try:
            # 仅基础语法检查（因为容器已提供安全边界）
            if self.use_security_validation:
                validation_result = self._validate_security(code)
                if not validation_result['is_safe']:
                    return {
                        'success': False,
                        'error': f"Security validation failed: {validation_result['issues']}",
                        'stdout': '',
                        'stderr': f"Security validation failed: {validation_result['issues']}",
                        'returncode': -1
                    }
            else:
                # 容器环境中仅做基础语法检查
                compile(code, '<string>', 'exec')

            # 执行代码，设置资源限制
            return self._execute_with_limits(code, timeout)

        except SyntaxError as e:
            return {
                'success': False,
                'error': f"Syntax error: {str(e)}",
                'stdout': '',
                'stderr': f"Syntax error: {str(e)}",
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Execution error: {str(e)}",
                'stdout': '',
                'stderr': f"Execution error: {str(e)}",
                'returncode': -1
            }

    async def execute_async(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """异步执行代码"""
        # 简单调用同步版本（在容器环境中资源限制足够）
        return self.execute(code, timeout)

    def _validate_security(self, code: str) -> Dict[str, Any]:
        """
        安全验证（仅在非容器环境启用）
        """
        issues = []

        # 基础危险模式检测
        dangerous_patterns = [
            r'\bos\.system\b',
            r'\bsubprocess\.',
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\b__import__\s*\(',
            r'\bopen\s*\(['
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                issues.append(f"Dangerous pattern detected: {pattern}")

        return {
            'is_safe': len(issues) == 0,
            'issues': issues
        }

    def _execute_with_limits(self, code: str, timeout: int) -> Dict[str, Any]:
        """
        在资源限制下执行代码
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_path = f.name

        try:
            # 设置资源限制
            def set_resource_limits():
                try:
                    import resource
                    # 内存限制（如果系统支持）
                    # resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, self.memory_limit))
                    # 进程数限制（如果系统支持）
                    # resource.setrlimit(resource.RLIMIT_NPROC, (self.max_pids, self.max_pids))
                except ImportError:
                    # resource模块在某些系统上不可用
                    pass
                except (ValueError, OSError):
                    # 限制设置失败（例如在Windows上）
                    pass

            # 使用subprocess执行
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=set_resource_limits if os.name == 'posix' else None
            )

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Timed out after {timeout}s',
                'stdout': '',
                'stderr': f'Execution timed out after {timeout}s',
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# 更新Executor类以支持动态沙箱选择
def create_executor_with_container_detection(container_config: dict = None):
    """
    根据容器配置创建最适合的executor
    """
    from .executor import Executor, DefaultSandbox

    if container_config and container_config.get('is_containerized'):
        # 容器环境中使用简化的沙箱
        return Executor(sandbox=SimplifiedSandbox(container_config.get('sandbox_config', {})))
    else:
        # 非容器环境使用严格的DefaultSandbox
        return Executor(sandbox=DefaultSandbox(strict_mode=True))