"""
Agent 能力定义
执行层的 7 种智能体能力
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime
import json
import re


class DegradationTracker:
    """
    降级追踪器

    追踪 LLM 调用降级情况，提供警告和统计
    """

    def __init__(self, warning_threshold: int = 3):
        self.degradation_count = 0
        self.warning_threshold = warning_threshold
        self.degradation_history: List[Dict[str, Any]] = []

    def record_degradation(self, agent_name: str, operation: str, error: str) -> Dict[str, Any]:
        """
        记录降级事件

        Returns:
            包含 is_degraded 和 should_warn 的字典
        """
        self.degradation_count += 1
        event = {
            'agent': agent_name,
            'operation': operation,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.degradation_history.append(event)

        return {
            'is_degraded': True,
            'should_warn': self.degradation_count >= self.warning_threshold,
            'degradation_count': self.degradation_count,
            'message': f"LLM {operation} 降级 (总计: {self.degradation_count}次)"
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取降级统计"""
        return {
            'total_degradations': self.degradation_count,
            'warning_threshold': self.warning_threshold,
            'history': self.degradation_history[-10:]  # 最近10条
        }


class AgentCapability(str, Enum):
    """智能体能力"""
    SENSE = "sense"  # 感知页面结构和特征
    PLAN = "plan"  # 规划提取策略
    ACT = "act"  # 执行提取操作
    VERIFY = "verify"  # 验证数据质量
    JUDGE = "judge"  # 做出决策判断
    EXPLORE = "explore"  # 探索页面链接
    REFLECT = "reflect"  # 反思和优化策略


class AgentInterface:
    """智能体接口"""

    def __init__(self, name: str, capability: AgentCapability):
        self.name = name
        self.capability = capability

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行智能体

        Args:
            context: 执行上下文

        Returns:
            执行结果
        """
        raise NotImplementedError()

    def get_description(self) -> str:
        """获取智能体描述"""
        raise NotImplementedError()

    def can_handle(self, context: Dict[str, Any]) -> bool:
        """判断是否能处理"""
        raise NotImplementedError()


# ==================== 感知智能体 ====================

class SenseAgent(AgentInterface):
    """感知智能体 - 分析页面结构"""

    def __init__(self, degradation_tracker: Optional[DegradationTracker] = None):
        super().__init__("SenseAgent", AgentCapability.SENSE)
        self.degradation_tracker = degradation_tracker or DegradationTracker()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        browser = context.get('browser')
        spec = context.get('spec')
        llm_client = context.get('llm_client')
        degradation_info = None

        # 检查浏览器是否可用
        if not browser:
            return {
                'success': False,
                'error': 'Browser not available',
                'structure': {},
                'features': {},
                'anti_bot_detected': False
            }

        # 1. 获取页面内容
        html = await browser.get_html()
        screenshot = await browser.take_screenshot()

        # 2. 程序快速分析（复用 FeatureDetector）
        from src.core.smart_router import FeatureDetector
        detector = FeatureDetector()
        features = detector.analyze(html)

        # 3. 分析页面结构
        structure = self._analyze_structure(html)
        features.update(structure)

        # 4. LLM 深度分析（如有 LLM 客户端）
        if llm_client:
            try:
                deep_analysis = await self._llm_analyze(html, spec, llm_client)
                features.update(deep_analysis)
            except Exception as e:
                error_msg = str(e)
                print(f"LLM 分析失败: {error_msg}")
                # 记录降级
                degradation_info = self.degradation_tracker.record_degradation(
                    self.name, 'llm_analyze', error_msg
                )
                if degradation_info.get('should_warn'):
                    print(f"警告: {degradation_info['message']}")

        # 5. 检测反爬
        anti_bot = self._detect_anti_bot(html)

        result = {
            'success': True,
            'structure': features,
            'features': features,
            'anti_bot_detected': anti_bot,
            'html_snapshot': html[:50000],  # 限制大小
            'screenshot': screenshot
        }

        # 添加降级信息
        if degradation_info:
            result['degradation'] = degradation_info

        return result

    def _analyze_structure(self, html: str) -> Dict[str, Any]:
        """分析页面结构"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        structure = {
            'type': 'unknown',
            'complexity': 'medium',
            'has_dynamic_content': False,
            'main_content_selector': None,
            'pagination_type': 'none',
            'pagination_selector': None,
            'data_fields': [],
            'estimated_items': 0
        }

        # 检测页面类型
        # 列表页特征
        list_indicators = ['<ul', '<ol', '<table', 'class="list"', 'class="item"', 'class="article']
        article_indicators = ['<article', 'class="article"', 'class="post"', 'class="news']

        list_count = sum(1 for ind in list_indicators if ind in html.lower())
        article_count = sum(1 for ind in article_indicators if ind in html.lower())

        if list_count > 2:
            structure['type'] = 'list'
        elif article_count > 0:
            structure['type'] = 'detail'
        elif '<form' in html.lower():
            structure['type'] = 'form'
        else:
            structure['type'] = 'other'

        # 检测分页类型
        if 'page=' in html.lower() or '/page/' in html.lower():
            structure['pagination_type'] = 'url'
        elif 'next' in html.lower() and ('<a' in html.lower() or '<button' in html.lower()):
            structure['pagination_type'] = 'click'
        elif 'scroll' in html.lower() or 'load-more' in html.lower():
            structure['pagination_type'] = 'scroll'

        # 查找分页选择器
        next_selectors = ['a.next', '.next-page', '.pagination a:last-child', '[rel="next"]']
        for selector in next_selectors:
            if soup.select_one(selector):
                structure['pagination_selector'] = selector
                break

        # 估算数据项数量
        common_containers = ['.article', '.item', '.post', '.news-item', 'article', 'tr']
        for container in common_containers:
            items = soup.select(container)
            if len(items) > 3:
                structure['estimated_items'] = len(items)
                structure['main_content_selector'] = container
                break

        return structure

    async def _llm_analyze(self, html: str, spec: Any, llm_client) -> Dict:
        """使用 LLM 增强分析 - 推理任务，使用 DeepSeek"""
        goal = spec.get('goal', '未知') if spec else '未知'

        prompt = f"""分析以下 HTML 页面，提取关键信息：

目标：{goal}

HTML 片段：
```
{html[:3000]}
```

请输出 JSON 格式：
{{
    "page_type": "list|detail|form|other",
    "main_content_selector": "CSS选择器",
    "pagination_type": "none|click|scroll|url",
    "pagination_selector": "CSS选择器或空",
    "data_fields": ["字段1", "字段2"],
    "special_handling": ["login", "captcha", "spa"]
}}"""
        try:
            # 推理任务 - 使用 reason() 方法
            if hasattr(llm_client, 'reason'):
                response = await llm_client.chat(
                    [{"role": "user", "content": prompt}],
                    task_type='reasoning'
                )
            else:
                response = await llm_client.chat([{"role": "user", "content": prompt}])
            # 解析 JSON
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0]
            else:
                json_str = response
            return json.loads(json_str)
        except Exception as e:
            print(f"LLM 分析解析失败: {e}")
            return {}

    def _detect_anti_bot(self, html: str) -> bool:
        """检测反爬机制"""
        anti_bot_keywords = ['cloudflare', 'recaptcha', 'challenge', 'cf-', 'turnstile']
        return any(keyword in html.lower() for keyword in anti_bot_keywords)

    def get_description(self) -> str:
        return "感知页面结构和特征，识别页面类型和反爬机制"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'browser' in context and context.get('browser') is not None


# ==================== 规划智能体 ====================

class PlanAgent(AgentInterface):
    """规划智能体 - 生成提取策略"""

    def __init__(self, llm_client=None, degradation_tracker: Optional[DegradationTracker] = None):
        super().__init__("PlanAgent", AgentCapability.PLAN)
        self.llm_client = llm_client
        self.degradation_tracker = degradation_tracker or DegradationTracker()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        page_structure = context.get('page_structure', {})
        spec = context.get('spec')
        llm_client = context.get('llm_client') or self.llm_client
        degradation_info = None

        # 1. 使用 LLM 生成策略（如有）
        if llm_client:
            try:
                strategy = await self._generate_with_llm(page_structure, spec, llm_client)
            except Exception as e:
                error_msg = str(e)
                print(f"LLM 策略生成失败: {error_msg}")
                strategy = self._fallback_strategy(page_structure, spec)
                # 记录降级
                degradation_info = self.degradation_tracker.record_degradation(
                    self.name, 'generate_strategy', error_msg
                )
                if degradation_info.get('should_warn'):
                    print(f"警告: {degradation_info['message']}")
        else:
            strategy = self._fallback_strategy(page_structure, spec)

        # 2. 生成提取代码
        # 如果 LLM 已返回代码，直接使用；否则生成或使用 LLM 编码
        code = strategy.get('extraction_code')
        if not code:
            # 尝试使用 LLM 生成代码（编码任务，使用 GLM）
            if llm_client and hasattr(llm_client, 'code'):
                try:
                    code = await self._generate_code_with_llm(strategy, spec, llm_client)
                except Exception as e:
                    print(f"LLM 代码生成失败: {e}")
                    code = self._generate_code(strategy, spec)
            else:
                code = self._generate_code(strategy, spec)

        result = {
            'success': True,
            'strategy': strategy,
            'selectors': strategy.get('selectors', {}),
            'generated_code': code
        }

        # 添加降级信息
        if degradation_info:
            result['degradation'] = degradation_info

        return result

    async def _generate_with_llm(self, structure: Dict, spec: Any, llm_client) -> Dict:
        """使用 LLM 生成策略 - 推理任务，使用 DeepSeek"""
        targets = spec.get('targets', []) if spec else []

        prompt = f"""基于页面结构分析，生成数据提取策略。

页面分析结果：
{json.dumps(structure, ensure_ascii=False, indent=2)}

需要提取的字段：
{json.dumps(targets, ensure_ascii=False, indent=2)}

请输出 JSON 格式的提取策略：
{{
    "strategy_type": "css|xpath|regex|api",
    "selectors": {{"field_name": "选择器"}},
    "container_selector": "数据容器选择器",
    "extraction_code": "完整的 Python 提取代码",
    "pagination_strategy": "none|click|scroll|url",
    "pagination_selector": "下一页按钮选择器",
    "estimated_items": 100
}}"""
        # 推理任务 - 使用 reason() 方法
        if hasattr(llm_client, 'reason'):
            response = await llm_client.chat(
                [{"role": "user", "content": prompt}],
                task_type='reasoning'
            )
        else:
            response = await llm_client.chat([{"role": "user", "content": prompt}])
        # 解析并返回
        try:
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0]
            else:
                json_str = response
            return json.loads(json_str)
        except:
            return self._fallback_strategy(structure, spec)

    def _fallback_strategy(self, structure: Dict[str, Any], spec: Any) -> Dict[str, Any]:
        """降级策略"""
        selectors = {}
        container_selector = structure.get('main_content_selector', 'body')

        if spec and 'targets' in spec:
            for target in spec['targets']:
                for field in target.get('fields', []):
                    field_name = field.get('name', '')
                    field_selector = field.get('selector', '')
                    if field_name and field_selector:
                        selectors[field_name] = field_selector

        return {
            'strategy_type': 'css',
            'selectors': selectors,
            'container_selector': container_selector,
            'pagination_strategy': structure.get('pagination_type', 'none'),
            'pagination_selector': structure.get('pagination_selector'),
            'estimated_items': structure.get('estimated_items', 10)
        }

    async def _generate_code_with_llm(self, strategy: Dict, spec: Any, llm_client) -> str:
        """使用 LLM 生成代码 - 编码任务，使用 GLM"""
        targets = spec.get('targets', []) if spec else []
        container_selector = strategy.get('container_selector', '.item')
        selectors = strategy.get('selectors', {})

        prompt = f"""生成 Python 数据提取代码。

目标字段：
{json.dumps(targets, ensure_ascii=False, indent=2)}

选择器配置：
- 容器选择器: {container_selector}
- 字段选择器: {json.dumps(selectors, ensure_ascii=False, indent=2)}

请生成完整的 Python 代码，使用 BeautifulSoup 进行数据提取。

代码要求：
1. 函数名 extract_data(html_content)
2. 返回列表，每个元素是字典
3. 使用 json.dumps 输出 JSON
4. 处理空值情况

只输出代码，不要解释。"""

        try:
            # 编码任务 - 使用 code() 方法
            if hasattr(llm_client, 'code'):
                response = await llm_client.chat(
                    [{"role": "user", "content": prompt}],
                    task_type='coding'
                )
            else:
                response = await llm_client.chat([{"role": "user", "content": prompt}])

            # 提取代码块
            if '```python' in response:
                code = response.split('```python')[1].split('```')[0]
            elif '```' in response:
                code = response.split('```')[1].split('```')[0]
            else:
                code = response

            return code.strip()
        except Exception as e:
            print(f"LLM 代码生成失败: {e}")
            return self._generate_code(strategy, spec)

    def _generate_code(self, strategy: Dict, spec: Any) -> str:
        """生成可执行的 Python 代码"""
        selectors = strategy.get('selectors', {})
        container_selector = strategy.get('container_selector', '.item') or '.item'

        code = '''from bs4 import BeautifulSoup
import json

def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []

    # 主容器选择器
    containers = soup.select('CONTAINER_SELECTOR')

    for container in containers:
        item = {}
'''
        # 根据选择器生成提取逻辑
        for field_name, field_selector in selectors.items():
            code += f'''
        # 提取 {field_name}
        {field_name}_elem = container.select_one('{field_selector}')
        item['{field_name}'] = {field_name}_elem.get_text(strip=True) if {field_name}_elem else ''
'''
        code += '''
        if any(item.values()):  # 只保留有内容的项
            results.append(item)

    return results

if __name__ == "__main__":
    import sys
    html = sys.stdin.read()
    result = extract_data(html)
    print(json.dumps(result, ensure_ascii=False))
'''
        return code.replace('CONTAINER_SELECTOR', container_selector)

    def get_description(self) -> str:
        return "基于页面结构和契约生成智能提取策略"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'spec' in context or 'page_structure' in context


# ==================== 执行智能体 ====================

class ExtractionMetrics:
    """提取质量指标"""

    def __init__(self):
        self.total_items = 0
        self.successful_items = 0
        self.failed_selectors: Dict[str, int] = {}  # 选择器 -> 失败次数
        self.missing_fields: Dict[str, int] = {}  # 字段名 -> 缺失次数
        self.empty_fields: Dict[str, int] = {}  # 字段名 -> 空值次数
        self.required_missing: Dict[str, int] = {}  # 必填字段缺失次数

    def record_selector_result(self, field_name: str, selector: str,
                                found: bool, has_value: bool, required: bool = False):
        """记录选择器匹配结果"""
        if not found:
            self.failed_selectors[selector] = self.failed_selectors.get(selector, 0) + 1
            self.missing_fields[field_name] = self.missing_fields.get(field_name, 0) + 1
            if required:
                self.required_missing[field_name] = self.required_missing.get(field_name, 0) + 1
        elif not has_value:
            self.empty_fields[field_name] = self.empty_fields.get(field_name, 0) + 1

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        return {
            'total_items': self.total_items,
            'successful_items': self.successful_items,
            'success_rate': self.successful_items / self.total_items if self.total_items > 0 else 0,
            'failed_selectors': self.failed_selectors,
            'missing_fields': self.missing_fields,
            'empty_fields': self.empty_fields,
            'required_missing': self.required_missing,
            'quality_score': self._calculate_quality()
        }

    def _calculate_quality(self) -> float:
        """计算质量分数"""
        if self.total_items == 0:
            return 0.0

        # 基础分数：成功提取的项目比例
        base_score = self.successful_items / self.total_items

        # 惩罚必填字段缺失
        required_penalty = 0
        if self.required_missing:
            total_required_missing = sum(self.required_missing.values())
            required_penalty = min(total_required_missing / (self.total_items + 1), 0.5)

        return max(0, base_score - required_penalty)


class ActAgent(AgentInterface):
    """执行智能体 - 执行提取操作"""

    def __init__(self):
        super().__init__("ActAgent", AgentCapability.ACT)

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        browser = context.get('browser')
        selectors = context.get('selectors', {})
        strategy = context.get('strategy', {})
        generated_code = context.get('generated_code')
        spec = context.get('spec')

        # 提取必填字段
        required_fields = set()
        if spec and 'targets' in spec:
            for target in spec['targets']:
                for field in target.get('fields', []):
                    if field.get('required', False):
                        required_fields.add(field.get('name'))

        # 1. 获取 HTML
        html = await browser.get_html()

        # 初始化指标追踪
        metrics = ExtractionMetrics()

        # 2. 执行提取
        if generated_code and strategy.get('strategy_type') == 'css':
            # 使用生成的代码提取
            extracted_data = await self._execute_code(generated_code, html)
        else:
            # 使用选择器直接提取
            extracted_data = await self._extract_with_selectors(
                browser, selectors, strategy, metrics, required_fields
            )

        # 3. 处理分页
        pagination_type = strategy.get('pagination_strategy', 'none')
        if pagination_type != 'none' and len(extracted_data) > 0:
            all_data = await self._handle_pagination(
                browser, extracted_data, strategy, selectors, metrics, required_fields
            )
            extracted_data = all_data

        # 获取指标
        metrics.total_items = len(extracted_data)
        metrics.successful_items = sum(1 for item in extracted_data if any(item.values()))

        return {
            'success': True,
            'extracted_data': extracted_data,
            'count': len(extracted_data),
            'extraction_metrics': metrics.get_metrics()
        }

    async def _extract_with_selectors(self, browser, selectors: Dict, strategy: Dict,
                                       metrics: Optional[ExtractionMetrics] = None,
                                       required_fields: Optional[set] = None) -> List[Dict]:
        """使用选择器提取数据"""
        from bs4 import BeautifulSoup
        html = await browser.get_html()
        soup = BeautifulSoup(html, 'html.parser')

        # 找到所有数据项容器
        container_selector = strategy.get('container_selector') or '.item'
        containers = soup.select(container_selector)

        results = []
        required_fields = required_fields or set()

        for container in containers:
            item = {}
            item_has_data = False

            for field_name, field_selector in selectors.items():
                if field_name.startswith('_'):
                    continue
                try:
                    elem = container.select_one(field_selector)

                    if metrics:
                        # 记录选择器匹配结果
                        found = elem is not None
                        has_value = found and bool(elem.get_text(strip=True))
                        is_required = field_name in required_fields
                        metrics.record_selector_result(field_name, field_selector,
                                                       found, has_value, is_required)

                    if elem:
                        text = elem.get_text(strip=True)
                        item[field_name] = text
                        if text:
                            item_has_data = True
                    else:
                        # 区分"元素不存在"和"元素为空"
                        item[field_name] = None  # None 表示元素不存在
                except Exception as e:
                    item[field_name] = None
                    if metrics:
                        metrics.failed_selectors[field_selector] = \
                            metrics.failed_selectors.get(field_selector, 0) + 1

            if item_has_data:  # 只保留有内容的项
                # 将 None 转换为空字符串以便兼容
                for key in item:
                    if item[key] is None:
                        item[key] = ''
                results.append(item)

        return results

    async def _execute_code(self, code: str, html: str) -> List[Dict]:
        """在沙箱中执行生成的代码（异步版本）"""
        import asyncio
        import json
        import tempfile
        import os

        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            script_path = f.name

        try:
            # 使用异步子进程执行
            proc = await asyncio.create_subprocess_exec(
                'python', script_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(html.encode()),
                timeout=60
            )

            if proc.returncode == 0 and stdout:
                try:
                    return json.loads(stdout.decode())
                except json.JSONDecodeError as e:
                    print(f"JSON 解析失败: {e}")
                    return []
            else:
                print(f"代码执行失败: {stderr.decode()}")
                return []
        except asyncio.TimeoutError:
            print(f"代码执行超时 (60秒)")
            return []
        except Exception as e:
            print(f"代码执行异常: {e}")
            return []
        finally:
            os.unlink(script_path)

    async def _handle_pagination(self, browser, initial_data, strategy, selectors,
                                  metrics: Optional[ExtractionMetrics] = None,
                                  required_fields: Optional[set] = None):
        """处理分页（改进版：支持去重和滚动检测）"""
        all_data = initial_data.copy()
        max_pages = strategy.get('max_pages', 5)
        pagination_type = strategy.get('pagination_strategy', 'click')
        next_selector = strategy.get('pagination_selector', 'a.next')

        # 用于去重 - 使用关键字段组合
        def get_item_key(item: Dict) -> str:
            """生成唯一键用于去重"""
            # 使用所有非空值组合作为键
            key_parts = [f"{k}:{v}" for k, v in sorted(item.items()) if v]
            return "|".join(key_parts) if key_parts else str(id(item))

        seen_keys = {get_item_key(item) for item in all_data}
        required_fields = required_fields or set()

        for page_num in range(max_pages - 1):
            try:
                prev_data_count = len(all_data)

                if pagination_type == 'click':
                    next_btn = await browser.page.query_selector(next_selector)
                    if next_btn:
                        await next_btn.click()
                        await browser.page.wait_for_load_state('networkidle')
                        await browser.page.wait_for_timeout(1000)  # 等待内容加载
                    else:
                        break
                elif pagination_type == 'scroll':
                    # 滚动前记录位置
                    prev_scroll_height = await browser.page.evaluate('document.body.scrollHeight')

                    await browser.scroll_to_bottom()
                    await browser.page.wait_for_timeout(1000)

                    # 检测是否到底（滚动后高度没变化）
                    new_scroll_height = await browser.page.evaluate('document.body.scrollHeight')
                    if new_scroll_height == prev_scroll_height:
                        print("已到达页面底部")
                        break

                elif pagination_type == 'url':
                    # URL 分页逻辑
                    current_url = await browser.get_current_url()
                    next_url = self._get_next_page_url(current_url, page_num + 2)
                    if next_url:
                        await browser.navigate(next_url)
                    else:
                        break

                # 提取新数据
                html = await browser.get_html()
                new_data = await self._extract_with_selectors(
                    browser, selectors, strategy, metrics, required_fields
                )

                # 去重添加
                for item in new_data:
                    item_key = get_item_key(item)
                    if item_key not in seen_keys:
                        all_data.append(item)
                        seen_keys.add(item_key)

                # 如果没有新数据，可能已到底
                if len(all_data) == prev_data_count:
                    print(f"分页第 {page_num + 2} 页无新数据，停止翻页")
                    break

            except Exception as e:
                print(f"分页处理失败 (第 {page_num + 2} 页): {e}")
                break

        return all_data

    def _get_next_page_url(self, current_url: str, next_page: int) -> Optional[str]:
        """
        生成下一页 URL

        支持:
        - ?page=N 格式
        - /page/N 格式
        - /N 格式
        """
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

        parsed = urlparse(current_url)

        # 尝试 ?page=N 格式
        query_params = parse_qs(parsed.query)
        if 'page' in query_params:
            query_params['page'] = [str(next_page)]
            new_query = urlencode(query_params, doseq=True)
            return urlunparse(parsed._replace(query=new_query))

        # 尝试 /page/N 或 /N 格式
        path_parts = parsed.path.rstrip('/').split('/')
        if path_parts and path_parts[-1].isdigit():
            # 替换最后的数字
            path_parts[-1] = str(next_page)
            new_path = '/'.join(path_parts)
            return urlunparse(parsed._replace(path=new_path))

        # 尝试添加 /page/N
        if 'page' in parsed.path.lower():
            # 已有 page，替换数字
            import re
            new_path = re.sub(r'/page/\d+', f'/page/{next_page}', parsed.path)
            return urlunparse(parsed._replace(path=new_path))

        return None

    def get_description(self) -> str:
        return "执行实际的数据提取操作"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        # 需要browser和至少一个提取参数
        browser_ok = 'browser' in context and context.get('browser') is not None
        has_params = 'selectors' in context or 'strategy' in context or 'generated_code' in context
        return browser_ok and has_params


# ==================== 验证智能体 ====================

class VerifyAgent(AgentInterface):
    """验证智能体 - 验证数据质量"""

    def __init__(self, verifier=None):
        super().__init__("VerifyAgent", AgentCapability.VERIFY)
        self.verifier = verifier

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        extracted_data = context.get('extracted_data', [])
        spec = context.get('spec')
        extraction_metrics = context.get('extraction_metrics', {})

        # 使用外部验证器或内置验证逻辑
        if self.verifier:
            try:
                verification_result = self.verifier.verify(extracted_data, context)
                quality_score = self._calculate_quality_from_result(verification_result)
            except Exception as e:
                print(f"验证器执行失败: {e}")
                quality_score = self._calculate_quality(extracted_data, spec)
                verification_result = {'status': 'partial', 'error': str(e)}
        else:
            # 内置验证逻辑
            quality_score = self._calculate_quality(extracted_data, spec)
            verification_result = self._build_verification_result(
                extracted_data, spec, quality_score, extraction_metrics
            )

        return {
            'success': True,
            'verification_result': verification_result,
            'valid_items': verification_result.get('valid_items', len(extracted_data)),
            'total_items': len(extracted_data),
            'quality_score': quality_score
        }

    def _calculate_quality(self, data: List, spec: Any) -> float:
        """计算质量分数"""
        if not data:
            return 0.0

        # 获取必填字段
        required_fields = set()
        if spec and 'targets' in spec:
            for target in spec['targets']:
                for field in target.get('fields', []):
                    if field.get('required', False):
                        required_fields.add(field.get('name'))

        if not required_fields:
            # 没有必填字段，检查数据非空率
            non_empty = sum(1 for item in data if any(item.values()))
            return non_empty / len(data) if data else 0.0

        # 检查必填字段完整性
        complete_count = 0
        for item in data:
            if all(item.get(f) for f in required_fields):
                complete_count += 1

        return complete_count / len(data) if data else 0.0

    def _calculate_quality_from_result(self, result: Dict) -> float:
        """从验证结果计算质量分数"""
        scores = result.get('scores', {})
        if scores:
            return sum(scores.values()) / len(scores)
        valid = result.get('valid_items', 0)
        total = result.get('total_items', 1)
        return valid / total if total > 0 else 0.0

    def _build_verification_result(self, data: List, spec: Any, quality_score: float,
                                    extraction_metrics: Optional[Dict] = None) -> Dict:
        """构建验证结果"""
        issues = []

        # 检查必填字段
        required_fields = []
        if spec and 'targets' in spec:
            for target in spec['targets']:
                for field in target.get('fields', []):
                    if field.get('required', False):
                        required_fields.append(field.get('name'))

        missing_count = 0
        for item in data:
            for field in required_fields:
                if not item.get(field):
                    missing_count += 1

        if missing_count > len(data) * 0.5:
            issues.append(f"超过50%的数据缺少必填字段")

        # 添加提取指标相关的问题
        if extraction_metrics:
            failed_selectors = extraction_metrics.get('failed_selectors', {})
            if failed_selectors:
                issues.append(f"部分选择器匹配失败: {list(failed_selectors.keys())[:3]}")

            required_missing = extraction_metrics.get('required_missing', {})
            if required_missing:
                issues.append(f"必填字段缺失: {dict(list(required_missing.items())[:3])}")

        result = {
            'status': 'passed' if quality_score >= 0.8 else 'partial' if quality_score >= 0.5 else 'failed',
            'total_items': len(data),
            'valid_items': int(len(data) * quality_score),
            'quality_score': quality_score,
            'issues': issues,
            'scores': {
                'completeness': quality_score,
                'consistency': self._check_consistency(data)
            }
        }

        # 包含提取指标
        if extraction_metrics:
            result['extraction_metrics'] = extraction_metrics

        return result

    def _check_consistency(self, data: List) -> float:
        """检查数据一致性"""
        if len(data) < 2:
            return 1.0

        first_keys = set(data[0].keys())
        inconsistent = 0
        for item in data[1:10]:  # 检查前10个
            if set(item.keys()) != first_keys:
                inconsistent += 1

        return 1.0 - (inconsistent / min(len(data) - 1, 9))

    def get_description(self) -> str:
        return "验证提取数据的质量和完整性"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'extracted_data' in context


# ==================== 决策智能体 ====================

class JudgeAgent(AgentInterface):
    """决策智能体 - 做出关键决策"""

    def __init__(self, llm_client=None, degradation_tracker: Optional[DegradationTracker] = None):
        super().__init__("JudgeAgent", AgentCapability.JUDGE)
        self.llm_client = llm_client
        self.degradation_tracker = degradation_tracker or DegradationTracker()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        quality_score = context.get('quality_score', 0)
        iteration = context.get('iteration', 0)
        max_iterations = context.get('max_iterations', 10)
        errors = context.get('errors', [])
        spec = context.get('spec')
        extracted_data_count = len(context.get('extracted_data', []))
        degradation_info = None

        # 1. 程序快速判断
        decision, reasoning = self._quick_decision(
            quality_score, iteration, max_iterations, errors, extracted_data_count
        )

        # 2. LLM 增强判断（如有 LLM 且决策不是 complete）
        llm_client = context.get('llm_client') or self.llm_client
        if llm_client and decision != 'complete':
            try:
                enhanced_decision = await self._llm_judge(context, llm_client)
                if enhanced_decision:
                    decision = enhanced_decision.get('decision', decision)
                    reasoning = enhanced_decision.get('reasoning', reasoning)
            except Exception as e:
                error_msg = str(e)
                print(f"LLM 决策失败: {error_msg}")
                # 记录降级
                degradation_info = self.degradation_tracker.record_degradation(
                    self.name, 'llm_judge', error_msg
                )
                if degradation_info.get('should_warn'):
                    print(f"警告: {degradation_info['message']}")

        result = {
            'success': True,
            'decision': decision,
            'reasoning': reasoning,
            'confidence': quality_score,
            'suggestions': self._get_suggestions(decision, context)
        }

        # 添加降级信息
        if degradation_info:
            result['degradation'] = degradation_info

        return result

    def _quick_decision(self, quality_score: float, iteration: int, max_iterations: int,
                        errors: List, data_count: int) -> Tuple[str, str]:
        """快速决策"""
        # 成功条件
        if quality_score >= 0.8 and data_count >= 5:
            return 'complete', f"质量分数 {quality_score:.2f} >= 0.8，数据量 {data_count}，任务完成"

        # 失败条件
        if iteration >= max_iterations:
            return 'terminate', f"已达到最大迭代次数 {max_iterations}"

        if len(errors) >= 5:
            return 'terminate', f"错误过多 ({len(errors)} 个错误)"

        # 可提升
        if quality_score >= 0.3:
            return 'reflect_and_retry', f"质量分数 {quality_score:.2f} 可提升，继续迭代 {iteration + 1}/{max_iterations}"

        return 'terminate', f"质量分数过低 {quality_score:.2f}"

    async def _llm_judge(self, context: Dict, llm_client) -> Optional[Dict]:
        """使用 LLM 增强决策 - 推理任务，使用 DeepSeek"""
        prompt = f"""分析爬取任务的执行情况，决定下一步行动：

质量分数：{context.get('quality_score', 0)}
迭代次数：{context.get('iteration', 0)}/{context.get('max_iterations', 10)}
错误列表：{context.get('errors', [])[:5]}
目标：{context.get('spec', {}).get('goal', '未知') if context.get('spec') else '未知'}
数据量：{len(context.get('extracted_data', []))}

可选决策：
- complete: 任务完成，质量达标
- reflect_and_retry: 反思并重试
- terminate: 终止任务

请输出 JSON：
{{"decision": "complete|reflect_and_retry|terminate", "reasoning": "原因"}}"""

        try:
            # 推理任务 - 使用 reason() 方法
            if hasattr(llm_client, 'reason'):
                response = await llm_client.chat(
                    [{"role": "user", "content": prompt}],
                    task_type='reasoning'
                )
            else:
                response = await llm_client.chat([{"role": "user", "content": prompt}])
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0]
            else:
                json_str = response
            return json.loads(json_str)
        except Exception as e:
            print(f"LLM 决策解析失败: {e}")
            return None

    def _get_suggestions(self, decision: str, context: Dict) -> List[str]:
        """获取改进建议"""
        if decision == 'complete':
            return ["任务已完成，可以结束"]

        suggestions = []
        errors = context.get('errors', [])
        quality_score = context.get('quality_score', 0)

        if any('selector' in str(e).lower() for e in errors):
            suggestions.append("考虑重新分析页面结构，更新选择器")
        if any('timeout' in str(e).lower() for e in errors):
            suggestions.append("增加等待时间或使用更稳定的选择器")
        if quality_score < 0.5:
            suggestions.append("质量分数过低，检查目标字段是否正确")
        if len(context.get('extracted_data', [])) < 5:
            suggestions.append("数据量不足，检查分页或数据容器选择器")

        return suggestions if suggestions else ["继续优化"]

    def get_description(self) -> str:
        return "在多个选项间做出最优决策"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'quality_score' in context or 'iteration' in context


# ==================== 探索智能体 ====================

class ExploreAgent(AgentInterface):
    """探索智能体 - 探索页面链接和结构"""

    def __init__(self):
        super().__init__("ExploreAgent", AgentCapability.EXPLORE)

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        browser = context.get('browser')
        current_url = context.get('current_url', '')
        depth = context.get('depth', 0)
        max_depth = context.get('max_depth', 2)
        base_url = context.get('base_url', current_url)

        if depth >= max_depth:
            return {'success': True, 'links': [], 'message': '已达到最大探索深度'}

        # 1. 提取所有链接
        all_links = await self._extract_links(browser)

        # 2. 过滤相关链接
        relevant_links = self._filter_links(all_links, base_url, context)

        # 3. 分类链接
        categorized = self._categorize_links(relevant_links, context)

        return {
            'success': True,
            'links': relevant_links,
            'categorized': categorized,
            'count': len(relevant_links),
            'next_depth': depth + 1
        }

    async def _extract_links(self, browser) -> List[str]:
        """提取页面链接"""
        html = await browser.get_html()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            # 过滤无效链接
            if href and not href.startswith(('#', 'javascript:', 'mailto:')):
                links.append(href)

        return list(set(links))

    def _filter_links(self, links: List[str], base_url: str, context: Dict) -> List[str]:
        """过滤相关链接"""
        from urllib.parse import urljoin, urlparse

        filtered = []
        base_domain = urlparse(base_url).netloc

        for link in links:
            try:
                absolute = urljoin(base_url, link)
                parsed = urlparse(absolute)

                # 只保留同域名链接
                if parsed.netloc == base_domain:
                    filtered.append(absolute)
            except Exception:
                continue

        return filtered

    def _categorize_links(self, links: List[str], context: Dict) -> Dict[str, List]:
        """分类链接"""
        categories = {
            'detail': [],   # 详情页
            'list': [],     # 列表页
            'other': []     # 其他
        }

        # 简单分类逻辑
        for link in links:
            link_lower = link.lower()
            if any(k in link_lower for k in ['detail', 'item', 'article', 'product', 'news', 'post']):
                categories['detail'].append(link)
            elif any(k in link_lower for k in ['list', 'page', 'category', 'search']):
                categories['list'].append(link)
            else:
                categories['other'].append(link)

        return categories

    def get_description(self) -> str:
        return "探索页面链接，发现新的数据源"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'browser' in context


# ==================== 反思智能体 ====================

class ReflectAgent(AgentInterface):
    """反思智能体 - 反思和优化策略"""

    def __init__(self, llm_client=None, degradation_tracker: Optional[DegradationTracker] = None):
        super().__init__("ReflectAgent", AgentCapability.REFLECT)
        self.llm_client = llm_client
        self.degradation_tracker = degradation_tracker or DegradationTracker()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        execution_history = context.get('execution_history', [])
        errors = context.get('errors', [])
        quality_score = context.get('quality_score', 0)
        spec = context.get('spec')
        llm_client = context.get('llm_client') or self.llm_client
        degradation_info = None

        # 1. 分析错误模式
        error_analysis = self._analyze_errors(errors)

        # 2. 分析执行历史
        history_analysis = self._analyze_history(execution_history)

        # 3. 生成改进建议
        if llm_client:
            try:
                improvements = await self._llm_reflect(
                    error_analysis, history_analysis, spec, llm_client
                )
            except Exception as e:
                error_msg = str(e)
                print(f"LLM 反思失败: {error_msg}")
                improvements = self._fallback_improvements(error_analysis)
                # 记录降级
                degradation_info = self.degradation_tracker.record_degradation(
                    self.name, 'llm_reflect', error_msg
                )
                if degradation_info.get('should_warn'):
                    print(f"警告: {degradation_info['message']}")
        else:
            improvements = self._fallback_improvements(error_analysis)

        result = {
            'success': True,
            'analysis': {
                'error_patterns': error_analysis,
                'history_summary': history_analysis
            },
            'improvements': improvements,
            'suggested_action': improvements.get('action', 'retry'),
            'new_selectors': improvements.get('selectors', {}),
            'new_strategy': improvements.get('strategy', None)
        }

        # 添加降级信息
        if degradation_info:
            result['degradation'] = degradation_info

        return result

    def _analyze_errors(self, errors: List[str]) -> Dict[str, Any]:
        """分析错误模式"""
        patterns = {}
        for error in errors:
            # 分类错误
            error_str = str(error).lower()
            if 'selector' in error_str or 'element' in error_str:
                key = 'selector_error'
            elif 'timeout' in error_str:
                key = 'timeout_error'
            elif 'network' in error_str or 'connection' in error_str:
                key = 'network_error'
            elif 'anti' in error_str or 'captcha' in error_str or 'block' in error_str:
                key = 'anti_bot_error'
            else:
                key = 'unknown_error'

            patterns[key] = patterns.get(key, 0) + 1

        return {
            'patterns': patterns,
            'most_common': max(patterns.items(), key=lambda x: x[1])[0] if patterns else 'none',
            'total_errors': len(errors)
        }

    def _analyze_history(self, history: List[Dict]) -> Dict[str, Any]:
        """分析执行历史"""
        if not history:
            return {'summary': '无历史记录'}

        return {
            'total_attempts': len(history),
            'stages': [h.get('stage') for h in history if h.get('stage')],
            'last_stage': history[-1].get('stage') if history else None,
            'quality_trend': [h.get('quality_score') for h in history if h.get('quality_score')]
        }

    async def _llm_reflect(self, error_analysis, history_analysis, spec, llm_client) -> Dict:
        """使用 LLM 进行反思 - 推理任务，使用 DeepSeek"""
        goal = spec.get('goal', '未知') if spec else '未知'

        prompt = f"""分析爬取任务失败的原因并生成改进建议：

错误分析：{json.dumps(error_analysis, ensure_ascii=False)}
历史记录：{json.dumps(history_analysis, ensure_ascii=False)}
目标：{goal}

请输出 JSON 格式的改进建议：
{{
    "action": "retry|change_strategy|abort",
    "reasoning": "原因分析",
    "selectors": {{"field_name": "新选择器"}},
    "strategy": "新策略描述",
    "precautions": ["注意事项"]
}}"""

        try:
            # 推理任务 - 使用 reason() 方法
            if hasattr(llm_client, 'reason'):
                response = await llm_client.chat(
                    [{"role": "user", "content": prompt}],
                    task_type='reasoning'
                )
            else:
                response = await llm_client.chat([{"role": "user", "content": prompt}])
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0]
            else:
                json_str = response
            return json.loads(json_str)
        except Exception as e:
            print(f"LLM 反思解析失败: {e}")
            return self._fallback_improvements(error_analysis)

    def _fallback_improvements(self, error_analysis: Dict) -> Dict:
        """降级改进建议"""
        most_common = error_analysis.get('most_common', 'unknown')

        suggestions = {
            'selector_error': {
                'action': 'retry',
                'reasoning': '选择器可能已更新，建议重新分析页面',
                'selectors': {},
                'strategy': 'reanalyze'
            },
            'timeout_error': {
                'action': 'retry',
                'reasoning': '增加等待时间',
                'selectors': {},
                'strategy': 'increase_timeout'
            },
            'network_error': {
                'action': 'retry',
                'reasoning': '网络问题，稍后重试',
                'selectors': {},
                'strategy': 'delay_retry'
            },
            'anti_bot_error': {
                'action': 'change_strategy',
                'reasoning': '检测到反爬机制，需要更换策略',
                'selectors': {},
                'strategy': 'use_stealth'
            },
            'unknown_error': {
                'action': 'abort',
                'reasoning': '未知错误，建议人工介入',
                'selectors': {},
                'strategy': None
            }
        }

        return suggestions.get(most_common, suggestions['unknown_error'])

    def get_description(self) -> str:
        return "分析失败原因，生成优化建议"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        # 可以在没有错误时也工作（用于分析改进）
        return True


# ==================== 智能体池 ====================

class AgentPool:
    """
    智能体池

    管理 7 种能力的智能体：
    - Sense: 感知页面结构
    - Plan: 规划提取策略
    - Act: 执行提取操作
    - Verify: 验证数据质量
    - Judge: 做出决策判断
    - Explore: 探索页面链接
    - Reflect: 反思和优化策略
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        # 共享的降级追踪器
        self.degradation_tracker = DegradationTracker()

        self.agents = {
            AgentCapability.SENSE: SenseAgent(self.degradation_tracker),
            AgentCapability.PLAN: PlanAgent(llm_client, self.degradation_tracker),
            AgentCapability.ACT: ActAgent(),
            AgentCapability.VERIFY: VerifyAgent(),
            AgentCapability.JUDGE: JudgeAgent(llm_client, self.degradation_tracker),
            AgentCapability.EXPLORE: ExploreAgent(),
            AgentCapability.REFLECT: ReflectAgent(llm_client, self.degradation_tracker)
        }

    def get_agent(self, capability: AgentCapability) -> Optional[AgentInterface]:
        """获取指定能力的智能体"""
        return self.agents.get(capability)

    async def execute_capability(self, capability: AgentCapability,
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """执行指定能力（异步版本）"""
        agent = self.get_agent(capability)
        if not agent:
            return {'success': False, 'error': f'Unknown capability: {capability}'}

        if not agent.can_handle(context):
            return {'success': False, 'error': f'Agent cannot handle this context'}

        return await agent.execute(context)

    def get_all_capabilities(self) -> List[AgentCapability]:
        """获取所有能力"""
        return list(self.agents.keys())

    def get_capability_description(self, capability: AgentCapability) -> str:
        """获取能力描述"""
        agent = self.get_agent(capability)
        return agent.get_description() if agent else ''

    def set_verifier(self, verifier):
        """设置验证器"""
        self.agents[AgentCapability.VERIFY] = VerifyAgent(verifier)

    def get_degradation_stats(self) -> Dict[str, Any]:
        """获取降级统计"""
        return self.degradation_tracker.get_stats()