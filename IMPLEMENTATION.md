# Full Self-Crawling Agent - 微观实现（五层治理架构 - 优化版）

## 1. 核心契约定义

### 1.1 Spec 契约

```python
from typing import TypedDict, List, Dict, Any, Optional, Literal
from datetime import datetime


class SpecContract(TypedDict, total=False):
    """Spec契约结构定义"""

    # ===== 基本信息 =====
    version: str  # 版本号，如 "v1"
    freeze: bool  # 是否冻结，冻结后不可修改
    created_at: str  # 创建时间
    updated_at: str  # 更新时间

    # ===== 任务目标 =====
    goal: str  # 用户目标描述
    target_url: str  # 目标URL

    # ===== 约束条件 =====
    constraints: List[str]  # 约束列表
    max_execution_time: int  # 最大执行时间（秒）
    max_retries: int  # 最大重试次数
    max_iterations: int  # 最大迭代次数

    # ===== 完成门禁条件 =====
    completion_gate: List[str]
    # 支持的门禁条件：
    # - "html_snapshot_exists" - HTML快照存在
    # - "sense_analysis_valid" - 感知分析有效
    # - "code_syntax_valid" - 代码语法正确
    # - "execution_success" - 执行成功
    # - "quality_score >= 0.6" - 质量分数 >= 阈值
    # - "sample_count >= 5" - 样本数 >= 阈值

    # ===== 证据要求 =====
    evidence: Dict[str, List[str]]
    # {
    #     "required": ["spec.yaml", "sense_report.json", ...],
    #     "optional": ["screenshots/", "reflection_memory.json"]
    # }

    # ===== 能力需求 =====
    capabilities: List[str]
    # ["sense", "plan", "act", "verify", "judge", "explore", "reflect"]
```

### 1.2 State 契约

```python
class StateContract(TypedDict, total=False):
    """状态契约"""

    # ===== 任务信息 =====
    task_id: str
    url: str
    goal: str

    # ===== Spec契约 =====
    spec: SpecContract

    # ===== 执行状态 =====
    stage: str  # 'initialized', 'sensing', 'planning', 'acting', 'verifying', 'judging'
    iteration: int
    routing_decision: Optional[Dict[str, Any]]

    # ===== 时间戳 =====
    created_at: str
    updated_at: str
    started_at: Optional[str]
    completed_at: Optional[str]

    # ===== 数据 =====
    html_snapshot: Optional[str]
    sense_analysis: Optional[Dict[str, Any]]
    generated_code: Optional[str]
    execution_result: Optional[Dict[str, Any]]
    sample_data: Optional[List[Any]]
    quality_score: Optional[float]

    # ===== 验证 =====
    syntax_valid: Optional[bool]
    gate_passed: Optional[bool]
    passed_gates: Optional[List[str]]
    failed_gates: Optional[List[str]]

    # ===== 性能 =====
    performance_data: Dict[str, Any]
    # {
    #     "sense_duration": float,
    #     "plan_duration": float,
    #     "act_duration": float,
    #     "total_duration": float,
    #     "llm_calls": int
    # }

    # ===== 历史 =====
    failure_history: List[Dict[str, Any]]
    evidence_collected: Dict[str, Any]
```

### 1.3 RoutingDecision 契约（新增）

```python
class RoutingDecision(TypedDict, total=False):
    """路由决策契约"""

    # ===== 策略信息 =====
    strategy: str  # 策略名称
    capabilities: List[str]  # 需要的能力列表
    expected_success_rate: float  # 预期成功率 0.0-1.0

    # ===== 分析信息 =====
    complexity: Literal['simple', 'medium', 'complex', 'extremely_complex']
    page_type: Literal['static', 'dynamic', 'spa', 'interactive', 'unknown']
    special_requirements: List[str]  # ['login', 'javascript', 'pagination', 'anti-bot']

    # ===== 执行参数 =====
    execution_params: Dict[str, Any]

    # ===== 备选方案 =====
    fallback_strategies: List[str]

    # ===== 元数据 =====
    decided_at: str
    decision_duration: float  # 秒
```

### 1.4 Evidence 契约

```python
class EvidenceContract(TypedDict, total=False):
    """证据契约"""

    # ===== 感知证据 =====
    html_snapshot: str
    sense_analysis: Dict[str, Any]

    # ===== 规划证据 =====
    generated_code: str
    reasoning: str

    # ===== 执行证据 =====
    execution_log: Dict[str, Any]
    screenshots: List[str]  # base64编码
    result: Dict[str, Any]

    # ===== 验证证据 =====
    sample_data: List[Any]
    quality_report: Dict[str, Any]
    issues: List[str]

    # ===== 评判证据 =====
    judge_decision: Dict[str, Any]

    # ===== 元数据 =====
    collected_at: str
    evidence_size: int  # 字节数
```

---

## 2. 战略层实现

### 2.1 SpecLoader 实现

```python
import yaml
import json
import os
from pathlib import Path


class SpecLoader:
    """Spec加载器"""

    def __init__(self, spec_path: Optional[str] = None):
        self.spec_path = spec_path
        self.spec: Optional[SpecContract] = None

    def load(self, spec_path: Optional[str] = None) -> SpecContract:
        """加载并验证Spec契约"""
        if spec_path is None:
            spec_path = self.spec_path

        if spec_path is None:
            raise ValueError("Spec path not provided")

        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Spec file not found: {spec_path}")

        # 读取Spec文件
        with open(spec_path, 'r', encoding='utf-8') as f:
            if spec_path.endswith('.yaml') or spec_path.endswith('.yml'):
                spec = yaml.safe_load(f)
            elif spec_path.endswith('.json'):
                spec = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {spec_path}")

        # 验证Spec
        self._validate(spec)

        # 设置时间戳
        spec.setdefault('created_at', datetime.now().isoformat())
        spec['updated_at'] = datetime.now().isoformat()

        self.spec = spec
        return spec

    def _validate(self, spec: dict) -> None:
        """验证Spec合法性"""
        # 基本字段验证
        if not spec.get('version'):
            raise ValueError("Spec must have 'version' field")

        if spec.get('freeze') is not True:
            raise ValueError("Spec must be frozen (freeze=true)")

        if not spec.get('goal'):
            raise ValueError("Spec must have 'goal' field")

        if not spec.get('completion_gate'):
            raise ValueError("Spec must have 'completion_gate' field")

        # 门禁条件格式验证
        for gate in spec['completion_gate']:
            if not isinstance(gate, str):
                raise ValueError(f"Invalid gate condition: {gate}")

        # 证据要求验证
        if 'evidence' in spec:
            evidence = spec['evidence']
            if not isinstance(evidence, dict):
                raise ValueError("Evidence must be a dictionary")

            if 'required' in evidence:
                if not isinstance(evidence['required'], list):
                    raise ValueError("Evidence required must be a list")

    def get_spec(self) -> SpecContract:
        """获取当前Spec"""
        if self.spec is None:
            raise ValueError("Spec not loaded")
        return self.spec

    def save(self, spec_path: str, spec: Optional[SpecContract] = None):
        """保存Spec到文件"""
        if spec is None:
            spec = self.spec

        if spec is None:
            raise ValueError("No spec to save")

        spec['updated_at'] = datetime.now().isoformat()

        Path(spec_path).parent.mkdir(parents=True, exist_ok=True)

        with open(spec_path, 'w', encoding='utf-8') as f:
            if spec_path.endswith('.yaml') or spec_path.endswith('.yml'):
                yaml.dump(spec, f, allow_unicode=True, default_flow_style=False)
            elif spec_path.endswith('.json'):
                json.dump(spec, f, ensure_ascii=False, indent=2)
```

### 2.2 CompletionGate 实现

```python
class CompletionGate:
    """完成门禁检查器"""

    def __init__(self):
        self.failed_gates = []
        self.passed_gates = []

    def check(self, state: dict, spec: SpecContract) -> bool:
        """检查是否满足完成门禁"""
        self.failed_gates = []
        self.passed_gates = []

        for gate_condition in spec.get('completion_gate', []):
            if self._evaluate(gate_condition, state):
                self.passed_gates.append(gate_condition)
            else:
                self.failed_gates.append(gate_condition)

        if self.failed_gates:
            state['gate_failed'] = True
            state['failed_gates'] = self.failed_gates
            state['passed_gates'] = self.passed_gates
            return False

        state['gate_passed'] = True
        state['passed_gates'] = self.passed_gates
        return True

    def _evaluate(self, condition: str, state: dict) -> bool:
        """评估门禁条件"""
        if condition == 'html_snapshot_exists':
            return state.get('html_snapshot') is not None

        elif condition == 'sense_analysis_valid':
            analysis = state.get('sense_analysis')
            return analysis is not None and len(analysis) > 0

        elif condition == 'code_syntax_valid':
            return state.get('syntax_valid', False)

        elif condition == 'execution_success':
            result = state.get('execution_result', {})
            return result.get('success', False) and result.get('data') is not None

        elif condition.startswith('quality_score >='):
            try:
                threshold = float(condition.split('>=')[1].strip())
                quality = state.get('quality_score', 0)
                return quality >= threshold
            except:
                return False

        elif condition.startswith('sample_count >='):
            try:
                threshold = int(condition.split('>=')[1].strip())
                data = state.get('sample_data', [])
                return len(data) >= threshold
            except:
                return False

        raise ValueError(f"Unknown gate condition: {condition}")
```

---

## 3. 管理层实现

### 3.1 SmartRouter 实现（核心优化）

```python
import re
import json


class FeatureDetector:
    """特征检测器 - 程序快速分析"""

    def analyze(self, html: str, url: Optional[str] = None) -> dict:
        """快速分析页面特征"""
        features = {
            'has_login': self._detect_login_form(html),
            'has_pagination': self._detect_pagination(html),
            'is_spa': self._detect_spa(html),
            'anti_bot_level': self._detect_anti_bot(html),
        }

        # 分析页面类型和复杂度
        features['page_type'] = self._classify_page_type(features)
        features['complexity'] = self._assess_complexity(features)

        return features

    def _detect_login_form(self, html: str) -> bool:
        """检测是否有登录表单"""
        login_patterns = [
            r'<input[^>]+type=["\']?password["\']?[^>]*>',
            r'<form[^>]+action=["\']?login["\']?[^>]*>',
        ]
        for pattern in login_patterns:
            if re.search(pattern, html, re.IGNORECASE | re.MULTILINE):
                return True
        return False

    def _detect_pagination(self, html: str) -> bool:
        """检测是否有分页"""
        pagination_patterns = [
            r'page=\d+',
            r'下一页|next\s*page',
        ]
        for pattern in pagination_patterns:
            if re.search(pattern, html, re.IGNORECASE):
                return True
        return False

    def _detect_spa(self, html: str) -> bool:
        """检测是否是SPA"""
        spa_indicators = ['fetch(', 'XMLHttpRequest', 'div id="app"', 'div id="root"']
        return any(indicator.lower() in html.lower() for indicator in spa_indicators)

    def _detect_anti_bot(self, html: str) -> str:
        """检测反爬等级"""
        lower_html = html.lower()
        if any(keyword in lower_html for keyword in ['cloudflare', 'recaptcha', 'captcha']):
            return 'medium'
        return 'none'

    def _classify_page_type(self, features: dict) -> str:
        """分类页面类型"""
        if features.get('is_spa'):
            return 'spa'
        elif features.get('has_login'):
            return 'interactive'
        else:
            return 'static'

    def _assess_complexity(self, features: dict) -> str:
        """评估复杂度"""
        score = 0
        if features.get('is_spa'):
            score += 2
        if features.get('has_login'):
            score += 2
        if features.get('anti_bot_level') == 'medium':
            score += 1

        if score >= 3:
            return 'complex'
        elif score >= 1:
            return 'medium'
        else:
            return 'simple'


class SmartRouter:
    """智能路由 - 混合判断核心"""

    STRATEGY_LIBRARY = {
        'direct_crawl': {
            'capabilities': ['sense', 'plan', 'act', 'verify'],
            'expected_success_rate': 0.95,
            'complexity': 'simple',
        },
        'pagination_crawl': {
            'capabilities': ['sense', 'plan', 'act', 'verify', 'handle_pagination'],
            'expected_success_rate': 0.85,
            'complexity': 'medium',
        },
        'spa_crawl': {
            'capabilities': ['sense', 'handle_spa', 'api_extract', 'verify'],
            'expected_success_rate': 0.70,
            'complexity': 'complex',
        },
    }

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client
        self.feature_detector = FeatureDetector()

    async def route(
        self,
        url: str,
        goal: str,
        html: Optional[str] = None,
        use_llm: bool = True
    ) -> RoutingDecision:
        """路由决策 - 三层决策模式"""
        from datetime import datetime
        start_time = datetime.now()

        # ========== 第1级：程序快速分析 ==========
        features = self.feature_detector.analyze(html or '', url)

        # 简单场景直接返回
        if features['complexity'] == 'simple' and not use_llm:
            strategy = self.STRATEGY_LIBRARY['direct_crawl'].copy()
        else:
            # ========== 第2级：策略生成 ==========
            if use_llm and self.llm_client is not None:
                strategy = await self._generate_with_llm(features, goal, html)
            else:
                strategy = self._match_from_library(features, goal)

        # ========== 第3级：程序验证 ==========
        if not self._validate_strategy(strategy):
            strategy = self._fallback_strategy(features)

        # 记录决策
        decision_duration = (datetime.now() - start_time).total_seconds()

        decision: RoutingDecision = {
            'strategy': strategy['name'],
            'capabilities': strategy['capabilities'],
            'expected_success_rate': strategy['expected_success_rate'],
            'complexity': features.get('complexity', 'simple'),
            'page_type': features.get('page_type', 'unknown'),
            'special_requirements': self._extract_requirements(features),
            'execution_params': strategy.get('params', {}),
            'decided_at': datetime.now().isoformat(),
            'decision_duration': decision_duration,
        }

        return decision

    async def _generate_with_llm(self, features: dict, goal: str, html: Optional[str]) -> dict:
        """使用LLM生成策略"""
        if self.llm_client is None:
            return self._match_from_library(features, goal)

        html_sample = (html or '')[:5000] if html else ''

        prompt = f"""
# 任务分析
{json.dumps(features, indent=2, ensure_ascii=False)}

# 用户目标
{goal}

# 页面片段
```html
{html_sample}
```

# 请生成最适合的爬取策略

考虑以下方面：
1. 推荐使用哪些能力（从7种中选择：sense, plan, act, verify, judge, explore, reflect）
2. 具体的执行步骤
3. 可能遇到的挑战
4. 应对策略
5. 预期成功率（0.0-1.0）

# 输出格式（JSON）
```json
{{
    "strategy": "策略名称",
    "capabilities": ["能力1", "能力2"],
    "steps": ["步骤1", "步骤2"],
    "considerations": ["注意事项"],
    "expected_success_rate": 0.8
}}
```
"""

        try:
            response = await self.llm_client.chat(prompt)

            if isinstance(response, str):
                if '```json' in response:
                    json_str = response.split('```json')[1].split('```')[0]
                elif '```' in response:
                    json_str = response.split('```')[1].split('```')[0]
                else:
                    json_str = response
                strategy_dict = json.loads(json_str)
            else:
                strategy_dict = response

            return {
                'name': strategy_dict.get('strategy', 'custom'),
                'capabilities': strategy_dict.get('capabilities', ['sense', 'plan', 'act', 'verify']),
                'expected_success_rate': strategy_dict.get('expected_success_rate', 0.7),
                'params': {
                    'steps': strategy_dict.get('steps', []),
                    'considerations': strategy_dict.get('considerations', []),
                }
            }

        except Exception as e:
            print(f"LLM生成策略失败: {e}")
            return self._match_from_library(features, goal)

    def _match_from_library(self, features: dict, goal: str) -> dict:
        """从策略库中匹配策略"""
        if features.get('has_login'):
            return self.STRATEGY_LIBRARY['login_crawl'].copy()

        if features.get('is_spa'):
            return self.STRATEGY_LIBRARY['spa_crawl'].copy()

        if features.get('has_pagination'):
            return self.STRATEGY_LIBRARY['pagination_crawl'].copy()

        return self.STRATEGY_LIBRARY['direct_crawl'].copy()

    def _validate_strategy(self, strategy: dict) -> bool:
        """验证策略的有效性"""
        if not strategy.get('name'):
            return False

        if not strategy.get('capabilities'):
            return False

        if not (0 <= strategy.get('expected_success_rate', 0) <= 1):
            return False

        valid_capabilities = {'sense', 'plan', 'act', 'verify', 'judge', 'explore', 'reflect'}

        return all(cap in valid_capabilities for cap in strategy.get('capabilities', []))

    def _fallback_strategy(self, features: dict) -> dict:
        """备选策略"""
        return {
            'name': 'fallback',
            'capabilities': ['sense', 'plan', 'act', 'verify'],
            'expected_success_rate': 0.5,
            'params': {'retry_count': 3, 'timeout': 60}
        }

    def _extract_requirements(self, features: dict) -> List[str]:
        """提取特殊需求"""
        requirements = []
        if features.get('has_login'):
            requirements.append('login')
        if features.get('is_spa'):
            requirements.append('javascript')
        if features.get('has_pagination'):
            requirements.append('pagination')
        if features.get('anti_bot_level') == 'medium':
            requirements.append('anti-bot')
        return requirements
```

### 3.2 StateManager 实现

```python
import uuid


class StateManager:
    """全局状态管理器"""

    def __init__(self):
        self.state = {}
        self.history = []
        self.snapshots = {}

    def initialize(self, url: str, goal: str, spec: SpecContract) -> dict:
        """初始化状态"""
        self.state = {
            'task_id': str(uuid.uuid4()),
            'url': url,
            'goal': goal,
            'spec': spec,
            'stage': 'initialized',
            'iteration': 0,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'performance_data': {},
            'failure_history': [],
            'evidence_collected': {}
        }
        return self.state.copy()

    def update(self, updates: dict):
        """更新状态"""
        self.state.update(updates)
        self.state['updated_at'] = datetime.now().isoformat()

        self.history.append({
            'stage': self.state['stage'],
            'timestamp': self.state['updated_at'],
            'updates': updates
        })

    def create_snapshot(self, tag: str):
        """创建状态快照"""
        self.snapshots[tag] = {
            'state': self.state.copy(),
            'timestamp': datetime.now().isoformat()
        }

    def restore_snapshot(self, tag: str):
        """恢复状态快照"""
        if tag in self.snapshots:
            self.state = self.snapshots[tag]['state'].copy()

    def get_state(self) -> dict:
        """获取当前状态"""
        return self.state.copy()

    def get_history(self) -> list:
        """获取历史记录"""
        return self.history.copy()
```

### 3.3 ContextCompressor 实现

```python
import hashlib


class ContextCompressor:
    """上下文压缩器"""

    def compress(self, context: dict) -> dict:
        """
        压缩策略：
        1. 移除冗余信息
        2. 摘要历史记录
        3. 保留关键状态
        4. 压缩失败历史
        """
        compressed = {}

        # 保留关键状态
        compressed['stage'] = context['stage']
        compressed['task_id'] = context['task_id']
        compressed['spec'] = context['spec']

        # 摘要历史记录
        if 'failure_history' in context:
            compressed['failure_summary'] = self.summarize_failures(
                context['failure_history']
            )

        # 保留最新证据
        if 'evidence' in context:
            compressed['latest_evidence'] = context['evidence'][-1]

        # 压缩性能数据
        if 'performance_data' in context:
            compressed['perf_summary'] = self.summarize_perf(
                context['performance_data']
            )

        # 移除已验证的中间数据
        if 'html_snapshot' in context and 'sense_analysis' in context:
            compressed['html_snapshot_hash'] = hashlib.md5(
                context['html_snapshot'].encode()
            ).hexdigest()

        return compressed

    def summarize_failures(self, failures: list) -> dict:
        """摘要失败历史"""
        error_types = {}
        for failure in failures:
            et = failure.get('error_type', 'unknown')
            error_types[et] = error_types.get(et, 0) + 1

        return {
            'total_failures': len(failures),
            'error_distribution': error_types,
            'latest_failure': failures[-1] if failures else None
        }

    def summarize_perf(self, perf_data: dict) -> dict:
        """摘要性能数据"""
        return {
            'total_duration': perf_data.get('total_duration', 0),
            'llm_calls': perf_data.get('llm_calls', 0),
            'iterations': perf_data.get('iterations', 0)
        }
```

### 3.4 RiskMonitor 实现

```python
class RiskMonitor:
    """风险监控器"""

    def __init__(self):
        self.thresholds = {
            "max_retries": 3,
            "max_iterations": 10,
            "quality_threshold": 0.4,
            "max_context_size": 10 * 1024 * 1024,
            "max_execution_time": 300
        }

    def check_risk(self, state: dict) -> dict:
        """检查风险"""
        warnings = []
        actions = []

        iteration = state.get("iteration", 0)
        if iteration > self.thresholds["max_retries"]:
            warnings.append(f"迭代次数超过阈值: {iteration}")
            actions.append("trigger_reflection")

        quality = state.get("quality_score", 1.0)
        if quality < self.thresholds["quality_threshold"]:
            warnings.append(f"质量分数太低: {quality}")
            actions.append("terminate_task")

        context_size = len(json.dumps(state))
        if context_size > self.thresholds["max_context_size"]:
            warnings.append(f"上下文大小超过阈值: {context_size / 1024 / 1024:.2f}MB")
            actions.append("compress_context")

        return {
            "risk_level": "high" if "terminate" in str(actions) else
                         "medium" if warnings else "low",
            "warnings": warnings,
            "recommended_actions": actions,
            "terminate": "terminate_task" in actions,
            "timestamp": datetime.now().isoformat()
        }
```

---

## 4. 验证层实现

### 4.1 Verifier 实现

```python
import ast


class Verifier:
    """结果验证器"""

    def verify_sense(self, analysis: dict) -> dict:
        """验证感知分析"""
        issues = []

        if not analysis.get('page_type'):
            issues.append("缺少页面类型")

        if not analysis.get('selectors'):
            issues.append("缺少选择器")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'score': max(0.0, 1.0 - len(issues) * 0.1)
        }

    def verify_code(self, code: str) -> dict:
        """验证生成的代码"""
        try:
            ast.parse(code)
            syntax_valid = True
            issues = []
        except SyntaxError as e:
            syntax_valid = False
            issues = [str(e)]

        return {
            'valid': syntax_valid,
            'syntax_valid': syntax_valid,
            'issues': issues
        }

    def verify_execution(self, result: dict) -> dict:
        """验证执行结果"""
        data = result.get('data', [])

        return {
            'valid': len(data) > 0,
            'has_data': len(data) > 0,
            'data_count': len(data),
            'issues': [] if len(data) > 0 else ["没有返回数据"]
        }

    def verify_quality(self, data: list, expected_fields: list) -> dict:
        """验证数据质量"""
        if not data:
            return {'valid': False, 'score': 0.0, 'issues': ['无数据']}

        # 检查字段完整性
        missing_fields = []
        for item in data[:5]:  # 检查前5个样本
            for field in expected_fields:
                if field not in item:
                    missing_fields.append(field)

        completeness = 1.0 - (len(set(missing_fields)) / max(len(expected_fields), 1))

        # 检查数据一致性
        consistency = self._check_consistency(data)

        score = (completeness + consistency) / 2

        return {
            'valid': score >= 0.6,
            'score': score,
            'completeness': completeness,
            'consistency': consistency,
            'issues': missing_fields if missing_fields else []
        }

    def _check_consistency(self, data: list) -> float:
        """检查数据一致性"""
        if len(data) < 2:
            return 1.0

        # 检查字段结构是否一致
        first_keys = set(data[0].keys())
        inconsistent_count = 0

        for item in data[1:5]:
            if set(item.keys()) != first_keys:
                inconsistent_count += 1

        return 1.0 - (inconsistent_count / min(len(data) - 1, 4))
```

### 4.2 EvidenceCollector 实现

```python
from pathlib import Path
import base64


class EvidenceCollector:
    """证据收集器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_index = {}

    def collect_sense(self, html: str, analysis: dict):
        """收集感知阶段证据"""
        sense_dir = self.output_dir / 'sense'
        sense_dir.mkdir(exist_ok=True)

        with open(sense_dir / 'html_snapshot.html', 'w', encoding='utf-8') as f:
            f.write(html[:100000])

        with open(sense_dir / 'analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        self.evidence_index['sense'] = str(sense_dir)

    def collect_plan(self, code: str, reasoning: str):
        """收集规划阶段证据"""
        plan_dir = self.output_dir / 'plan'
        plan_dir.mkdir(exist_ok=True)

        with open(plan_dir / 'generated_code.py', 'w', encoding='utf-8') as f:
            f.write(code)

        with open(plan_dir / 'reasoning.md', 'w', encoding='utf-8') as f:
            f.write(reasoning)

        self.evidence_index['plan'] = str(plan_dir)

    def collect_act(self, result: dict, logs: list, screenshots: list):
        """收集执行阶段证据"""
        act_dir = self.output_dir / 'act'
        act_dir.mkdir(exist_ok=True)

        with open(act_dir / 'result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        with open(act_dir / 'execution_log.json', 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

        screenshot_dir = act_dir / 'screenshots'
        screenshot_dir.mkdir(exist_ok=True)
        for i, screenshot in enumerate(screenshots):
            img_data = base64.b64decode(screenshot)
            with open(screenshot_dir / f'screenshot_{i}.png', 'wb') as f:
                f.write(img_data)

        self.evidence_index['act'] = str(act_dir)

    def collect_verify(self, sample_data: list, quality_report: dict, issues: list):
        """收集验证阶段证据"""
        verify_dir = self.output_dir / 'verify'
        verify_dir.mkdir(exist_ok=True)

        with open(verify_dir / 'sample_data.json', 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)

        with open(verify_dir / 'quality_report.json', 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)

        with open(verify_dir / 'issues.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(issues))

        self.evidence_index['verify'] = str(verify_dir)

    def save_index(self):
        """保存证据索引"""
        with open(self.output_dir / 'evidence_index.json', 'w', encoding='utf-8') as f:
            json.dump(self.evidence_index, f, ensure_ascii=False, indent=2)

    def get_evidence_path(self) -> Path:
        """获取证据包路径"""
        return self.output_dir
```

### 4.3 GateDecision 实现

```python
class GateDecision:
    """门禁决策器"""

    def __init__(self):
        self.completion_gate = CompletionGate()

    def decide(self, state: dict, spec: SpecContract) -> str:
        """门禁决策"""
        gate_passed = self.completion_gate.check(state, spec)

        if not gate_passed:
            failed_gates = state.get("failed_gates", [])

            if "execution_success" in failed_gates:
                return "soal_repair"

            elif any("quality_score" in str(gate) for gate in failed_gates):
                quality = state.get("quality_score", 0)
                if quality >= 0.4:
                    return "reflect_and_retry"
                else:
                    return "terminate"

            elif "html_snapshot_exists" in failed_gates:
                return "retry_with_delay"

            else:
                return "terminate"

        return "complete"

    def get_decision_reason(self, decision: str, state: dict) -> str:
        """获取决策原因"""
        reasons = {
            "complete": "所有门禁条件满足",
            "soal_repair": "执行失败，进入SOOAL修复流程",
            "reflect_and_retry": "质量不达标但可修复，反思后重试",
            "terminate": "无法修复，终止任务",
            "retry_with_delay": "网络问题，延迟后重试"
        }
        return reasons.get(decision, "未知决策")
```

---

## 5. 执行层实现

### 5.1 AgentPool 实现

```python
class AgentPool:
    """智能体池 - 提议模式"""

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client
        self.agents = {}

    async def propose(self, capability: str, context: dict) -> dict:
        """生成提议"""
        if capability == 'sense':
            return await self._sense_propose(context)
        elif capability == 'plan':
            return await self._plan_propose(context)
        elif capability == 'act':
            return await self._act_propose(context)
        elif capability == 'verify':
            return await self._verify_propose(context)
        elif capability == 'judge':
            return await self._judge_propose(context)
        else:
            raise ValueError(f"Unknown capability: {capability}")

    async def _sense_propose(self, context: dict) -> dict:
        """感知提议"""
        html = context.get('html_snapshot', '')
        goal = context.get('goal', '')

        prompt = f"""
        分析以下HTML页面：

        ```html
        {html[:5000]}
        ```

        用户目标：{goal}

        请分析：
        1. 页面类型（静态/动态/SPA/交互式）
        2. 需要提取的数据字段
        3. 推荐的CSS选择器或XPath
        4. 是否有特殊需求（登录/分页/反爬）

        输出格式（JSON）：
        {{
            "page_type": "static",
            "fields": ["title", "content"],
            "selectors": {{"title": "h1", "content": ".article"}},
            "special_requirements": []
        }}
        """

        response = await self.llm_client.chat(prompt)

        # 解析响应
        if '```json' in response:
            json_str = response.split('```json')[1].split('```')[0]
        else:
            json_str = response
        analysis = json.loads(json_str)

        return {
            'type': 'sense_proposal',
            'analysis': analysis,
            'reasoning': f"基于页面分析，推荐使用以下选择器提取{goal}"
        }

    async def _plan_propose(self, context: dict) -> dict:
        """规划提议"""
        analysis = context.get('sense_analysis', {})
        goal = context.get('goal', '')

        selectors = analysis.get('selectors', {})
        fields = analysis.get('fields', [])

        code_template = f"""
import json
import re
from bs4 import BeautifulSoup

def crawl(html_content: str):
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []

    # 根据分析结果提取数据
"""

        for field, selector in selectors.items():
            code_template += f"""
    {field}_elements = soup.select('{selector}')
    for element in {field}_elements:
        # TODO: 提取逻辑
        pass
"""

        code_template += """
    return results

if __name__ == '__main__':
    import sys
    html = sys.stdin.read()
    result = crawl(html)
    print(json.dumps(result, ensure_ascii=False, indent=2))
"""

        return {
            'type': 'plan_proposal',
            'code': code_template,
            'language': 'python',
            'reasoning': f"基于感知分析生成爬虫代码，使用BeautifulSoup提取{', '.join(fields)}"
        }

    async def _act_propose(self, context: dict) -> dict:
        """行动提议"""
        code = context.get('generated_code', '')
        return {
            'type': 'act_proposal',
            'action': 'execute_code',
            'code': code,
            'timeout': 60,
            'sandbox': True
        }

    async def _verify_propose(self, context: dict) -> dict:
        """验证提议"""
        result = context.get('execution_result', {})
        data = result.get('data', [])

        return {
            'type': 'verify_proposal',
            'action': 'evaluate_quality',
            'data_sample': data[:5] if data else [],
            'data_count': len(data),
            'has_errors': result.get('success') is False
        }

    async def _judge_propose(self, context: dict) -> dict:
        """评判提议"""
        quality_score = context.get('quality_score', 0)
        iteration = context.get('iteration', 0)
        max_iterations = context.get('spec', {}).get('max_iterations', 10)

        if quality_score >= 0.8:
            decision = 'complete'
        elif quality_score >= 0.4 and iteration < max_iterations:
            decision = 'reflect_and_retry'
        else:
            decision = 'terminate'

        return {
            'type': 'judge_proposal',
            'decision': decision,
            'reasoning': f"质量分数={quality_score}, 迭代次数={iteration}/{max_iterations}",
            'confidence': quality_score
        }
```

### 5.2 Executor 实现

```python
import subprocess
import tempfile
import os


class Executor:
    """代码执行器"""

    def __init__(self, sandbox: Optional[Any] = None):
        self.sandbox = sandbox

    async def execute(self, code: str, timeout: int = 60, html: Optional[str] = None) -> dict:
        """执行代码"""
        try:
            if self.sandbox:
                result = await self.sandbox.run(code, timeout=timeout, html=html)
            else:
                result = await self._execute_in_subprocess(code, timeout, html)

            return {
                'success': True,
                'data': result.get('data', []),
                'logs': result.get('logs', []),
                'execution_time': result.get('execution_time', 0),
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'data': [],
                'logs': [],
                'execution_time': 0,
                'error': str(e),
                'error_type': type(e).__name__
            }

    async def _execute_in_subprocess(self, code: str, timeout: int, html: Optional[str]) -> dict:
        """在子进程中执行代码"""
        import asyncio
        from datetime import datetime

        start_time = datetime.now()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            script_path = f.name

        try:
            if html:
                process = await asyncio.create_subprocess_exec(
                    'python', script_path,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=html.encode()),
                    timeout=timeout
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    'python', script_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

            execution_time = (datetime.now() - start_time).total_seconds()

            if process.returncode != 0:
                raise RuntimeError(f"Execution failed: {stderr.decode()}")

            try:
                data = json.loads(stdout.decode())
            except json.JSONDecodeError:
                data = []

            return {
                'data': data,
                'logs': [stderr.decode()],
                'execution_time': execution_time
            }

        finally:
            if os.path.exists(script_path):
                os.unlink(script_path)
```

---

## 6. 工具层实现

### 6.1 Browser 实现

```python
from playwright.async_api import async_playwright


class Browser:
    """浏览器工具"""

    def __init__(self):
        self.playwright = None
        self.browser = None

    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def browse(self, url: str, wait_for: str = 'body', timeout: int = 30000) -> dict:
        """访问网页"""
        page = await self.browser.new_page()

        try:
            await page.goto(url, wait_until='domcontentloaded', timeout=timeout)

            if wait_for:
                await page.wait_for_selector(wait_for, timeout=timeout)

            html = await page.content()
            status = await page.evaluate('() => document.readyState')

            return {
                'html': html,
                'status': status,
                'url': page.url,
                'title': await page.title()
            }

        finally:
            await page.close()
```

### 6.2 LLMClient 实现

```python
import httpx


class LLMClient:
    """大模型客户端"""

    def __init__(self, api_key: str, base_url: str = "https://open.bigmodel.cn/api/paas/v4"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def chat(
        self,
        prompt: str,
        model: str = "glm-4",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """调用LLM"""
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )

            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
```

---

**总结**：优化后的实现将6层架构简化为5层，核心是SmartRouter的三层决策模式（程序→规则→LLM），通过混合判断实现效率与灵活性的平衡。
