"""
代码执行器
在沙箱环境中安全执行代码
"""
from typing import Dict, Any, List, Optional, Tuple, Set
import asyncio
import subprocess
import tempfile
import os
import json
import sys
import re
import logging
import ast

logger = logging.getLogger('executor')


# 安全模块白名单（仅允许导入这些模块）
SAFE_MODULES: Set[str] = {
    'json', 're', 'math', 'datetime', 'collections',
    'itertools', 'functools', 'typing', 'dataclasses',
    'bs4', 'BeautifulSoup', 'lxml', 'html.parser',
    'time', 'random', 'statistics', 'decimal', 'fractions',
    'urllib.parse', 'urllib.request', 'requests',  # 注意：requests 在严格模式下仍需额外检查
    'csv', 'xml.etree.ElementTree', 'xml.dom.minidom',
    'hashlib', 'uuid', 'string', 'copy', 'enum',
    'operator', 'calendar', 'zoneinfo', 'zoneinfo.ZoneInfo'
}

# 危险模块和函数黑名单（用于非容器模式的严格沙箱）
DANGEROUS_MODULES: Set[str] = {
    'os', 'subprocess', 'sys', 'shutil', 'socket', 'requests',  # 'requests' 在此处是为了在某些情况下进一步限制
    'urllib', 'http', 'ftplib', 'telnetlib', 'smtplib', 'poplib',
    'imaplib', 'nntplib', 'popen2', 'commands', 'pexpect',
    'paramiko', 'fabric', 'cryptography', 'pickle', 'shelve',
    'marshal', 'imp', 'importlib', 'code', 'codeop', 'compile',
    'exec', 'eval', '__import__', 'builtins'
}

DANGEROUS_FUNCTIONS: Set[str] = {
    'eval', 'exec', 'compile', 'execfile', 'open', 'input',
    'raw_input', 'file', 'reload', '__import__', 'globals',
    'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr',
    'hasattr', 'type', 'object', 'base64.b64decode',
    'ctypes', 'multiprocessing', 'threading', 'exec'
}

DANGEROUS_PATTERNS = [
    r'\bos\.system\b',
    r'\bos\.popen\b',
    r'\bsubprocess\.',
    r'\beval\s*\(',
    r'\bexec\s*\(',
    r'\b__import__\s*\(',
    r'\bopen\s*\(["\']',
    r'\bcompile\s*\(',
    r'\bglobals\s*\(\)',
    r'\blocals\s*\(\)',
    r'\bgetattr\s*\(',
    r'\bsetattr\s*\(',
    r'\bdelattr\s*\(',
    r'\bhasattr\s*\(',
]

# AST-based dangerous nodes
DANGEROUS_AST_NODES = {
    ast.Import, ast.ImportFrom,  # Imports are checked separately
    ast.Call,  # Function calls are analyzed separately
    ast.Attribute,  # Attribute access could be dangerous (e.g., os.system)
    ast.Subscript,  # Could be used for dangerous operations
    ast.Slice,  # Slice operations
    ast.ListComp,  # List comprehensions could be used for side effects
    ast.GeneratorExp,  # Generator expressions
}


class ASTSecurityVisitor(ast.NodeVisitor):
    """
    AST 安全检查访问器
    分析代码 AST 结构，识别潜在危险操作
    """

    def __init__(self, allowed_modules=None):
        self.issues = []
        self.dangerous_functions_called = []
        self.dangerous_imports = []
        self.allowed_modules = allowed_modules or set()

    def visit_Import(self, node):
        """检查导入语句"""
        for alias in node.names:
            module_name = alias.name.split('.')[0]  # 只检查顶级模块
            if self.allowed_modules and module_name not in self.allowed_modules:
                if module_name in DANGEROUS_MODULES:
                    self.issues.append(f"Dangerous import detected: {module_name}")
                    self.dangerous_imports.append(module_name)
            elif module_name in DANGEROUS_MODULES:
                self.issues.append(f"Dangerous import detected: {module_name}")
                self.dangerous_imports.append(module_name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """检查 from...import 语句"""
        module_name = node.module.split('.')[0] if node.module else ""

        # 检查是否在允许的模块列表中
        if self.allowed_modules and module_name not in self.allowed_modules:
            if module_name in DANGEROUS_MODULES:
                self.issues.append(f"Dangerous from-import detected: {module_name}")
                self.dangerous_imports.append(module_name)
        elif module_name in DANGEROUS_MODULES:
            self.issues.append(f"Dangerous from-import detected: {module_name}")
            self.dangerous_imports.append(module_name)
        self.generic_visit(node)

    def visit_Call(self, node):
        """检查函数调用"""
        # 检查函数名
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle attribute calls like os.system
            attr_name = node.func.attr
            if attr_name in DANGEROUS_FUNCTIONS:
                full_call = f"{node.func.attr}"
                self.issues.append(f"Dangerous function call: {full_call}")
                self.dangerous_functions_called.append(full_call)

        if func_name and func_name in DANGEROUS_FUNCTIONS:
            self.issues.append(f"Dangerous function call: {func_name}")
            self.dangerous_functions_called.append(func_name)

        self.generic_visit(node)


class Executor:
    """
    代码执行器

    职责：
    - 在安全环境中执行代码
    - 捕获执行结果和错误
    - 限制资源使用
    - 防止恶意代码执行
    """

    def __init__(self, sandbox: Optional[Any] = None):
        self.sandbox = sandbox or DefaultSandbox()

    def execute_python(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        执行 Python 代码

        Args:
            code: Python 代码
            timeout: 超时时间（秒）

        Returns:
            执行结果字典
        """
        return self.sandbox.execute(code, timeout)

    async def execute_python_async(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        异步执行 Python 代码

        Args:
            code: Python 代码
            timeout: 超时时间（秒）

        Returns:
            执行结果字典
        """
        return await self.sandbox.execute_async(code, timeout)

    def execute_with_context(self, code: str, context: Dict[str, Any],
                            timeout: int = 30) -> Dict[str, Any]:
        """
        执行代码并提供上下文

        Args:
            code: 代码
            context: 上下文变量
            timeout: 超时时间（秒）

        Returns:
            执行结果
        """
        # 将上下文注入代码
        context_code = self._build_context_code(context, code)
        return self.execute_python(context_code, timeout)

    async def execute_with_context_async(self, code: str, context: Dict[str, Any],
                                        timeout: int = 30) -> Dict[str, Any]:
        """
        异步执行代码并提供上下文

        Args:
            code: 代码
            context: 上下文变量
            timeout: 超时时间（秒）

        Returns:
            执行结果
        """
        context_code = self._build_context_code(context, code)
        return await self.execute_python_async(context_code, timeout)

    def _build_context_code(self, context: Dict[str, Any], code: str) -> str:
        """构建带上下文的代码"""
        # 将上下文转换为 Python 代码
        context_lines = []
        for key, value in context.items():
            context_lines.append(f'{key} = {repr(value)}')

        return '\n'.join(context_lines) + '\n\n' + code


class DefaultSandbox:
    """
    默认沙箱实现

    使用 subprocess 执行代码，提供基础隔离
    包含代码安全检查和危险操作过滤
    """

    def __init__(self, allowed_modules: Optional[Set[str]] = None,
                 strict_mode: bool = True):
        """
        初始化沙箱

        Args:
            allowed_modules: 允许导入的模块白名单
            strict_mode: 严格模式，启用危险代码检测
        """
        self.allowed_modules = allowed_modules or {
            'json', 're', 'math', 'datetime', 'collections',
            'itertools', 'functools', 'typing', 'dataclasses',
            'bs4', 'BeautifulSoup', 'lxml', 'html.parser'
        }
        self.strict_mode = strict_mode

    def validate_code_ast(self, code: str) -> Tuple[bool, List[str]]:
        """
        使用 AST 分析验证代码安全性

        Args:
            code: 要验证的代码

        Returns:
            (是否安全, 检测到的问题列表)
        """
        issues = []

        try:
            # 解析代码为 AST
            tree = ast.parse(code)

            # 使用安全访问器遍历 AST
            visitor = ASTSecurityVisitor(allowed_modules=self.allowed_modules)
            visitor.visit(tree)

            # 合并问题
            issues.extend(visitor.issues)

        except SyntaxError as e:
            issues.append(f"Syntax error in code: {str(e)}")

        return len(issues) == 0, issues

    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """
        验证代码安全性 - 使用多种验证方式，优先使用白名单机制
        """
        issues = []

        if not self.strict_mode:
            return True, issues

        # 首先进行白名单检查（如果设置了允许模块）
        if self.allowed_modules:
            # 检查导入语句是否都在允许范围内
            import_match_patterns = [
                r'^\s*import\s+([a-zA-Z0-9_.]+)',
                r'^\s*from\s+([a-zA-Z0-9_.]+)\s+import',
            ]

            for pattern in import_match_patterns:
                for match in re.finditer(pattern, code, re.MULTILINE):
                    imported_module = match.group(1).split('.')[0]  # 只检查顶层模块
                    if imported_module not in self.allowed_modules:
                        issues.append(f"检测到不允许的模块导入: {imported_module}")

        # 正则表达式检查（保留黑名单检查作为后备）
        regex_issues = []

        # 检查危险模式
        for pattern in DANGEROUS_PATTERNS:
            matches = re.findall(pattern, code)
            if matches:
                regex_issues.append(f"检测到危险操作: {pattern}")

        # 检查危险模块导入
        import_patterns = [
            r'^\s*import\s+(\w+)',
            r'^\s*from\s+(\w+)\s+import',
        ]

        for pattern in import_patterns:
            for match in re.finditer(pattern, code, re.MULTILINE):
                module = match.group(1)
                if module in DANGEROUS_MODULES:
                    regex_issues.append(f"检测到危险模块导入: {module}")

        # 检查危险函数调用（更精确的模式）
        for func in DANGEROUS_FUNCTIONS:
            # 避免误报，只匹配函数调用形式，而非变量名或注释
            pattern = rf'\b{re.escape(func)}\s*\('
            if re.search(pattern, code):
                regex_issues.append(f"检测到危险函数调用: {func}")

        # 额外的安全检查：检测动态导入
        dynamic_import_pattern = r'__import__|importlib\.import_module|exec\(|eval\('
        if re.search(dynamic_import_pattern, code):
            regex_issues.append("检测到动态导入或执行函数，可能存在安全风险")

        # 额外的安全检查：检测文件系统访问
        file_access_pattern = r'open\(|\.write\(|\.read\(|shutil\.|os\.path'
        if re.search(file_access_pattern, code):
            regex_issues.append("检测到文件系统访问函数")

        issues.extend(regex_issues)

        # AST 分析检查
        ast_is_safe, ast_issues = self.validate_code_ast(code)
        issues.extend(ast_issues)

        return len(issues) == 0, issues

    def execute(self, code: str, timeout: int = 30, skip_validation: bool = False) -> Dict[str, Any]:
        """
        执行代码（同步版本）

        Args:
            code: Python 代码
            timeout: 超时时间（秒）
            skip_validation: 跳过安全验证

        Returns:
            执行结果字典
        """
        # 安全验证
        if not skip_validation and self.strict_mode:
            is_safe, issues = self.validate_code(code)
            if not is_safe:
                logger.warning(f"代码安全检查失败: {issues}")
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f"安全检查失败:\n" + "\n".join(issues),
                    'returncode': -1,
                    'error': 'security_violation'
                }

        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            # 执行代码
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'execution_time': result.returncode
            }

        except subprocess.TimeoutExpired:
            logger.warning(f"代码执行超时 ({timeout}s)")
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Timeout after {timeout} seconds',
                'returncode': -1,
                'error': 'timeout'
            }

        except Exception as e:
            logger.error(f"代码执行错误: {e}")
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1,
                'error': 'execution_error'
            }

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def execute_async(self, code: str, timeout: int = 30, skip_validation: bool = False) -> Dict[str, Any]:
        """
        执行代码（异步版本）

        使用 asyncio.create_subprocess_exec 避免阻塞事件循环

        Args:
            code: Python 代码
            timeout: 超时时间（秒）
            skip_validation: 跳过安全验证

        Returns:
            执行结果字典
        """
        # 安全验证
        if not skip_validation and self.strict_mode:
            is_safe, issues = self.validate_code(code)
            if not is_safe:
                logger.warning(f"代码安全检查失败: {issues}")
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f"安全检查失败:\n" + "\n".join(issues),
                    'returncode': -1,
                    'error': 'security_violation'
                }

        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            # 使用异步子进程
            proc = await asyncio.create_subprocess_exec(
                sys.executable, temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout
                )

                return {
                    'success': proc.returncode == 0,
                    'stdout': stdout.decode() if stdout else '',
                    'stderr': stderr.decode() if stderr else '',
                    'returncode': proc.returncode,
                    'execution_time': timeout
                }

            except asyncio.TimeoutError:
                # 超时时终止进程
                logger.warning(f"代码执行超时 ({timeout}s)")
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass

                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f'Timeout after {timeout} seconds',
                    'returncode': -1,
                    'error': 'timeout'
                }

        except Exception as e:
            logger.error(f"代码执行错误: {e}")
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1,
                'error': 'execution_error'
            }

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class ContainerSandbox:
    """
    容器内沙箱 - 简化版

    因为整个 Agent 已经在 Docker 容器中运行，
    这里只需要基本的超时和资源控制，不需要严格的模块限制

    设计理念：
    - 预装常用库（parsel, lxml, bs4, pyquery, selectolax 等）
    - LLM 可以自由选择工具
    - 容器本身提供安全边界
    """

    def __init__(self, default_timeout: int = 60, max_timeout: int = 300):
        """
        初始化容器沙箱

        Args:
            default_timeout: 默认超时时间（秒）
            max_timeout: 最大允许超时时间（秒）
        """
        self.default_timeout = default_timeout
        self.max_timeout = min(max_timeout, 300)  # 最多 5 分钟

    def execute(self, code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        在容器内执行代码（同步版本）

        Args:
            code: Python 代码
            timeout: 超时时间（秒）

        Returns:
            执行结果字典
        """
        timeout = min(timeout or self.default_timeout, self.max_timeout)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                timeout=timeout,
                cwd=os.getcwd()
            )

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout.decode('utf-8', errors='replace'),
                'stderr': result.stderr.decode('utf-8', errors='replace'),
                'returncode': result.returncode
            }

        except subprocess.TimeoutExpired:
            logger.warning(f"代码执行超时 ({timeout}s)")
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Timeout after {timeout}s',
                'returncode': -1,
                'error': 'timeout'
            }

        except Exception as e:
            logger.error(f"代码执行错误: {e}")
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1,
                'error': 'execution_error'
            }

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def execute_async(self, code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        在容器内执行代码（异步版本）

        使用 asyncio.create_subprocess_exec 避免阻塞事件循环

        Args:
            code: Python 代码
            timeout: 超时时间（秒）

        Returns:
            执行结果字典
        """
        timeout = min(timeout or self.default_timeout, self.max_timeout)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout
                )

                return {
                    'success': proc.returncode == 0,
                    'stdout': stdout.decode('utf-8', errors='replace') if stdout else '',
                    'stderr': stderr.decode('utf-8', errors='replace') if stderr else '',
                    'returncode': proc.returncode
                }

            except asyncio.TimeoutError:
                logger.warning(f"代码执行超时 ({timeout}s)")
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass

                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f'Timeout after {timeout}s',
                    'returncode': -1,
                    'error': 'timeout'
                }

        except Exception as e:
            logger.error(f"代码执行错误: {e}")
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1,
                'error': 'execution_error'
            }

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class DockerSandbox:
    """
    Docker 沙箱实现（需要 Docker 环境）

    提供更高级的隔离，推荐用于生产环境
    """

    def __init__(self, image: str = 'python:3.10-slim'):
        self.image = image

    def execute(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """在 Docker 容器中执行代码"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            # 构建 Docker 命令
            docker_cmd = [
                'docker', 'run',
                '--rm',
                '--network', 'none',  # 禁用网络
                '--memory', '512m',  # 限制内存
                '--cpus', '1.0',  # 限制 CPU
                '-v', f'{temp_path}:/app/code.py:ro',
                self.image,
                'python', '/app/code.py'
            ]

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 5  # Docker 启动需要额外时间
            )

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'execution_time': timeout
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Timeout after {timeout} seconds',
                'returncode': -1,
                'error': 'timeout'
            }

        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1,
                'error': 'execution_error'
            }

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# ==================== 代码生成器 ====================

class CodeGenerator:
    """
    代码生成器

    根据策略生成可执行的爬虫代码
    """

    def __init__(self):
        pass

    def generate_extractor_code(self, selectors: Dict[str, str],
                               strategy: Dict[str, Any]) -> str:
        """
        生成数据提取代码

        Args:
            selectors: 字段选择器
            strategy: 提取策略

        Returns:
            Python 代码字符串
        """
        approach = strategy.get('approach', 'direct')

        if approach == 'direct':
            return self._generate_direct_code(selectors)
        elif approach == 'pagination':
            return self._generate_pagination_code(selectors, strategy)
        elif approach == 'scroll':
            return self._generate_scroll_code(selectors, strategy)
        else:
            return self._generate_direct_code(selectors)

    def _generate_direct_code(self, selectors: Dict[str, str]) -> str:
        """生成直接提取代码"""
        code = '''
from bs4 import BeautifulSoup
import json

# 获取 HTML（由外部提供）
html = HTML_CONTENT

# 解析 HTML
soup = BeautifulSoup(html, 'html.parser')

# 提取数据
results = []
items = soup.select('TODO_ITEM_SELECTOR')

for item in items:
    data = {}
    {selector_code}
    results.append(data)

# 输出结果
print(json.dumps(results, ensure_ascii=False))
'''.strip()

        # 生成选择器代码
        selector_lines = []
        for field_name, selector in selectors.items():
            selector_lines.append(
                f"    data['{field_name}'] = item.select_one('{selector}').get_text().strip()"
                f" if item.select_one('{selector}') else ''"
            )

        selector_code = '\n'.join(selector_lines)
        return code.replace('{selector_code}', selector_code)

    def _generate_pagination_code(self, selectors: Dict[str, str],
                                  strategy: Dict[str, Any]) -> str:
        """生成分页提取代码"""
        code = '''
from bs4 import BeautifulSoup
import json

# 配置
max_pages = {max_pages}
base_url = "{base_url}"

all_results = []

for page in range(1, max_pages + 1):
    # 构建页面 URL
    url = base_url.format(page=page)

    # 获取 HTML（由外部提供）
    html = fetch_html(url)

    # 解析并提取
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.select('TODO_ITEM_SELECTOR')

    for item in items:
        data = {{}}
        {selector_code}
        all_results.append(data)

# 输出结果
print(json.dumps(all_results, ensure_ascii=False))
'''.strip()

        selector_lines = []
        for field_name, selector in selectors.items():
            selector_lines.append(
                f"        data['{field_name}'] = item.select_one('{selector}').get_text().strip()"
                f" if item.select_one('{selector}') else ''"
            )

        selector_code = '\n'.join(selector_lines)
        max_pages = strategy.get('max_pages', 10)
        base_url = strategy.get('base_url', '')

        return (code
                .replace('{selector_code}', selector_code)
                .replace('{max_pages}', str(max_pages))
                .replace('{base_url}', base_url))

    def _generate_scroll_code(self, selectors: Dict[str, str],
                             strategy: Dict[str, Any]) -> str:
        """生成滚动提取代码"""
        code = '''
from bs4 import BeautifulSoup
import json
import time

# 配置
max_scrolls = {max_scrolls}
scroll_delay = {scroll_delay}

all_results = []
seen_items = set()

for scroll in range(max_scrolls):
    # 滚动页面（由外部提供）
    scroll_page()
    time.sleep(scroll_delay)

    # 获取 HTML
    html = get_html()

    # 解析并提取
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.select('TODO_ITEM_SELECTOR')

    for item in items:
        {selector_code}
        # 去重
        item_key = json.dumps(data, sort_keys=True)
        if item_key not in seen_items:
            seen_items.add(item_key)
            all_results.append(data)

    # 检查是否还有更多内容
    if not has_more_content():
        break

# 输出结果
print(json.dumps(all_results, ensure_ascii=False))
'''.strip()

        selector_lines = []
        for field_name, selector in selectors.items():
            selector_lines.append(
                f"        data['{field_name}'] = item.select_one('{selector}').get_text().strip()"
                f" if item.select_one('{selector}') else ''"
            )

        selector_code = '\n'.join(selector_lines)
        max_scrolls = strategy.get('max_scrolls', 10)
        scroll_delay = strategy.get('scroll_delay', 2)

        return (code
                .replace('{selector_code}', selector_code)
                .replace('{max_scrolls}', str(max_scrolls))
                .replace('{scroll_delay}', str(scroll_delay)))