"""
代码执行器 — 精简版

Sandbox 类支持三种执行模式：
- strict_mode=True  (本地开发): 校验危险代码后再 subprocess 执行
- strict_mode=False (Docker容器): 只管超时，不限模块
- docker_env 设置时: 通过 DockerEnvironment 在容器内执行（最强能力）
"""

import asyncio
import base64
import subprocess
import tempfile
import os
import sys
import re
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger('executor')

# 危险模式（只在 strict_mode 下检查）
DANGEROUS_PATTERNS = [
    (r'\bos\.system\b', 'os.system'),
    (r'\bos\.popen\b', 'os.popen'),
    (r'\bsubprocess\.', 'subprocess'),
    (r'\b__import__\s*\(', '__import__'),
    (r'\beval\s*\(', 'eval'),
    (r'\bexec\s*\(', 'exec'),
]


class Sandbox:
    """
    唯一的沙箱实现

    本地开发: Sandbox(strict_mode=True)   → 校验 + subprocess
    Docker:  Sandbox(strict_mode=False)  → 只管超时 + subprocess
    Docker环境: Sandbox(docker_env=env)  → 在 Docker 容器内执行
    """

    def __init__(self, strict_mode: bool = True, default_timeout: int = 60,
                 policy_manager=None, docker_env=None):
        self.strict_mode = strict_mode
        self.default_timeout = default_timeout
        self.policy_manager = policy_manager
        self.docker_env = docker_env  # DockerEnvironment instance (optional)

    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """校验代码安全性（仅 strict_mode 生效）"""
        if not self.strict_mode:
            return True, []

        issues = []
        for pattern, name in DANGEROUS_PATTERNS:
            if re.search(pattern, code):
                issues.append(f"检测到危险操作: {name}")
        return len(issues) == 0, issues

    async def execute(self, code: str, stdin_data: str = None,
                      timeout: int = None) -> Dict[str, Any]:
        """执行 Python 代码"""
        timeout = timeout or self.default_timeout

        # Docker 模式：在容器内执行，跳过本地安全检查
        if self.docker_env:
            return await self._execute_in_docker(code, stdin_data, timeout)

        # PolicyManager 策略检查（优先于 strict_mode 检查）
        if self.policy_manager:
            policy_result = self.policy_manager.check_code(code)
            if not policy_result.get('allowed', True):
                violations = policy_result.get('violations', [])
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': "策略检查失败:\n" + "\n".join(
                        f"[{v.get('level', 'CRITICAL')}] {v.get('description', v.get('name', ''))}"
                        for v in violations
                    ),
                    'returncode': -1,
                }

        # strict 模式下做安全校验
        if self.strict_mode:
            is_safe, issues = self.validate_code(code)
            if not is_safe:
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': "安全检查失败:\n" + "\n".join(issues),
                    'returncode': -1,
                }

        # 写临时文件
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, temp_path,
                stdin=asyncio.subprocess.PIPE if stdin_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdin_bytes = stdin_data.encode('utf-8', errors='replace') if stdin_data else None

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=stdin_bytes),
                    timeout=timeout
                )
                return {
                    'success': proc.returncode == 0,
                    'stdout': stdout.decode('utf-8', errors='replace') if stdout else '',
                    'stderr': stderr.decode('utf-8', errors='replace') if stderr else '',
                    'returncode': proc.returncode,
                }
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f'执行超时 ({timeout}s)',
                    'returncode': -1,
                }

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def execute_bash(self, command: str, timeout: int = None) -> Dict[str, Any]:
        """执行 bash 命令（仅 Docker 模式可用）"""
        if not self.docker_env:
            return {
                'success': False,
                'stdout': '',
                'stderr': 'bash 执行需要 Docker 环境',
                'returncode': -1,
            }
        timeout = timeout or self.default_timeout
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.docker_env.execute(command, timeout=timeout)
        )
        return {
            'success': result['returncode'] == 0,
            'stdout': result['output'] if result['returncode'] == 0 else '',
            'stderr': result['output'] if result['returncode'] != 0 else '',
            'returncode': result['returncode'],
        }

    async def _execute_in_docker(self, code: str, stdin_data: str = None,
                                  timeout: int = 60) -> Dict[str, Any]:
        """在 Docker 容器内执行 Python 代码"""
        loop = asyncio.get_event_loop()

        # Write code to container via base64 (avoids quoting issues)
        encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
        write_result = await loop.run_in_executor(
            None,
            lambda: self.docker_env.execute(
                f"echo '{encoded_code}' | base64 -d > /workspace/_script.py",
                timeout=30,
            )
        )
        if write_result['returncode'] != 0:
            return {
                'success': False,
                'stdout': '',
                'stderr': f"写入脚本失败: {write_result['output']}",
                'returncode': -1,
            }

        # If stdin_data provided, write it to a file too
        run_cmd = "python3 /workspace/_script.py"
        if stdin_data:
            encoded_stdin = base64.b64encode(stdin_data.encode('utf-8')).decode('ascii')
            await loop.run_in_executor(
                None,
                lambda: self.docker_env.execute(
                    f"echo '{encoded_stdin}' | base64 -d > /workspace/_stdin.txt",
                    timeout=30,
                )
            )
            run_cmd = "python3 /workspace/_script.py < /workspace/_stdin.txt"

        # Execute the script
        result = await loop.run_in_executor(
            None,
            lambda: self.docker_env.execute(run_cmd, timeout=timeout)
        )

        return {
            'success': result['returncode'] == 0,
            'stdout': result['output'] if result['returncode'] == 0 else '',
            'stderr': result['output'] if result['returncode'] != 0 else '',
            'returncode': result['returncode'],
        }


class CrawlExecutor:
    """
    兼容执行器：对接旧测试入口 execute(spec, url)。
    """

    def __init__(self, agent_pool=None, llm_client=None):
        self.agent_pool = agent_pool
        self.llm_client = llm_client

    async def execute(self, spec: Dict[str, Any], url: str) -> Dict[str, Any]:
        from src.orchestrator import SelfCrawlingAgent

        agent = SelfCrawlingAgent()
        result = await agent.run(url, dict(spec or {}))

        extracted_data: List[Dict[str, Any]] = []
        if isinstance(result, dict):
            if isinstance(result.get('extracted_data'), list):
                extracted_data = result.get('extracted_data', [])
            elif isinstance(result.get('results'), list):
                for page in result.get('results', []):
                    page_result = page.get('result', {}) if isinstance(page, dict) else {}
                    items = page_result.get('extracted_data', [])
                    if isinstance(items, list):
                        extracted_data.extend(items)

        return {
            'success': bool(result.get('success')) if isinstance(result, dict) else False,
            'data': extracted_data,
            'plan_attempts': 1,
            'error': result.get('error') if isinstance(result, dict) else 'unknown error',
            'raw_result': result,
        }
