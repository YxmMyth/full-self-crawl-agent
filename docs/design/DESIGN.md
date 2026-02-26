# Full Self-Crawling Agent - 中观设计（五层治理架构 - 优化版）

## 1. 治理组件结构（重构）

### 1.1 战略层组件

#### 1.1.1 SpecLoader（规范加载器）

**职责**：加载并验证Spec契约，确保契约合法性和完整性

```python
class SpecLoader:
    """Spec加载器"""

    def __init__(self, spec_path: Optional[str] = None):
        self.spec_path = spec_path
        self.spec: Optional[SpecContract] = None

    def load(self, spec_path: Optional[str] = None) -> SpecContract:
        """加载并验证Spec契约"""
        # 读取Spec文件（YAML或JSON）
        # 验证Spec合法性
        # 设置时间戳
        # 返回验证通过的Spec
        pass

    def _validate(self, spec: dict) -> None:
        """验证Spec合法性"""
        # 基本字段验证
        assert spec.get('version'), "Spec必须有version字段"
        assert spec.get('freeze') is True, "Spec必须冻结"
        assert spec.get('goal'), "Spec必须有goal字段"
        assert spec.get('completion_gate'), "Spec必须有门禁条件"

        # 门禁条件格式验证
        for gate in spec['completion_gate']:
            assert isinstance(gate, str), f"无效的门禁条件: {gate}"
```

**输入**：Spec文件路径（YAML/JSON）
**处理**：读取、验证、冻结
**输出**：验证通过的Spec契约

#### 1.1.2 PolicyManager（策略管理器）

**职责**：管理边界和约束策略

```python
class PolicyManager:
    """策略管理器"""

    def __init__(self):
        self.policies = {
            # 执行限制
            'max_execution_time': 300,  # 5分钟
            'max_retries': 3,
            'max_iterations': 10,

            # 质量阈值
            'min_quality_score': 0.4,
            'min_sample_count': 5,

            # 资源限制
            'max_memory_mb': 1024,
            'max_context_size_mb': 10,

            # 安全限制
            'sandbox_enabled': True,
            'network_isolation': True,
        }
```

**策略类型**：
- 执行限制：最大执行时间、重试次数、迭代次数
- 质量阈值：最小质量分数、样本数量
- 资源限制：内存、上下文大小
- 安全限制：沙箱、网络隔离

#### 1.1.3 CompletionGate（完成门禁检查器）

**职责**：基于证据验证任务是否完成

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

        return len(self.failed_gates) == 0

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
            return result.get('success', False)

        elif condition.startswith('quality_score >='):
            threshold = float(condition.split('>=')[1].strip())
            quality = state.get('quality_score', 0)
            return quality >= threshold

        elif condition.startswith('sample_count >='):
            threshold = int(condition.split('>=')[1].strip())
            data = state.get('sample_data', [])
            return len(data) >= threshold
```

**门禁条件类型**：
- 存在性检查：`html_snapshot_exists`
- 有效性检查：`sense_analysis_valid`
- 语法检查：`code_syntax_valid`
- 执行检查：`execution_success`
- 阈值检查：`quality_score >= 0.6`, `sample_count >= 5`

### 1.2 管理层组件

#### 1.2.1 Orchestrator（工作流编排器）

**职责**：编排工作流，调度执行顺序

```python
class Orchestrator:
    """工作流编排器"""

    def __init__(
        self,
        smart_router: 'SmartRouter',
        state_manager: 'StateManager',
        context_compressor: 'ContextCompressor',
        risk_monitor: 'RiskMonitor'
    ):
        self.smart_router = smart_router
        self.state_manager = state_manager
        self.context_compressor = context_compressor
        self.risk_monitor = risk_monitor

    async def orchestrate(
        self,
        url: str,
        goal: str,
        spec: SpecContract
    ) -> dict:
        """编排完整工作流"""

        # 1. 初始化状态
        state = self.state_manager.initialize(url, goal, spec)

        # 2. 智能路由决策
        routing_decision = await self.smart_router.route(
            url=url,
            goal=goal,
            html=state.get('html_snapshot', '')
        )

        state['routing_decision'] = routing_decision

        # 3. 执行主循环
        max_iterations = spec.get('max_iterations', 10)

        for iteration in range(max_iterations):
            # 执行当前阶段
            result = await self._execute_stage(state)

            # 检查门禁
            gate_passed = CompletionGate().check(state, spec)

            if gate_passed:
                return self._build_success_result(state)

            # 检查风险
            risk_result = self.risk_monitor.check_risk(state)

            if risk_result.get('terminate', False):
                return self._build_failure_result(state, "风险过高，终止任务")

            # 压缩上下文
            state = self.context_compressor.compress(state)

        return self._build_failure_result(state, "超过最大迭代次数")
```

**工作流阶段**：
1. 初始化：加载Spec、初始化状态
2. 路由决策：SmartRouter生成策略
3. 执行循环：Sense → Plan → Act → Verify → Judge
4. 门禁检查：检查完成条件
5. 风险监控：检查风险阈值
6. 上下文压缩：压缩状态
7. 结果返回：成功或失败

#### 1.2.2 StateManager（全局状态管理）

**职责**：维护全局状态，提供状态访问和更新

```python
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
        return self.state

    def update(self, updates: dict):
        """更新状态"""
        self.state.update(updates)
        self.state['updated_at'] = datetime.now().isoformat()

        # 记录历史
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
```

**状态字段**：
- 任务信息：task_id, url, goal
- Spec契约：spec
- 执行状态：stage, iteration
- 时间戳：created_at, updated_at
- 性能数据：performance_data
- 失败历史：failure_history
- 证据收集：evidence_collected

#### 1.2.3 SmartRouter（智能路由）← 核心优化

**职责**：程序+LLM混合判断，动态选择最佳策略

```python
class SmartRouter:
    """智能路由 - 混合判断核心"""

    def __init__(self, llm_client: 'LLMClient'):
        self.llm = llm_client
        self.feature_detector = FeatureDetector()

    async def route(self, url: str, goal: str, html: str) -> RoutingDecision:
        """
        三层决策模式：
        1. 程序快速分析（<50ms）
        2. LLM生成策略（核心灵活性）
        3. 程序验证可行性
        """

        # === 第1级：程序快速分析 ===
        features = self.feature_detector.analyze(html)

        # 简单场景直接返回
        if features['complexity'] == 'simple':
            return self._simple_strategy(features)

        # === 第2级：LLM生成策略 ===
        strategy = await self._llm_generate(features, goal, html[:5000])

        # === 第3级：程序验证 ===
        if self._validate(strategy):
            return strategy

        # === 备选方案 ===
        return self._fallback(features)

    def _program_analysis(self, html: str) -> dict:
        """程序快速分析 - 规则明确、效率优先"""
        return {
            'has_login': bool(re.search(r'password|login', html, re.I)),
            'has_pagination': bool(re.search(r'page=\d+|下一页', html, re.I)),
            'is_spa': 'fetch(' in html or 'div id="app"' in html,
            'anti_bot_level': self._detect_anti_bot(html)
        }

    async def _llm_generate(self, features: dict, goal: str, html_sample: str) -> dict:
        """LLM生成策略 - 语义复杂、灵活性强"""
        prompt = f"""
        任务分析：{json.dumps(features, indent=2)}
        用户目标：{goal}
        HTML片段：{html_sample}

        请生成最适合的爬取策略：
        1. 推荐使用哪些能力（从7种中选择）
        2. 具体的执行步骤
        3. 需要注意什么
        4. 预期成功率

        （你的建议不受预定义策略限制，可以创新）
        """
        return await self.llm.chat(prompt)

    def _validate(self, strategy: dict) -> bool:
        """程序验证 - 安全兜底"""
        # 验证必需字段
        if not strategy.get('name'):
            return False

        if not (0 <= strategy.get('expected_success_rate', 0) <= 1):
            return False

        # 验证能力合理性
        valid_capabilities = {'sense', 'plan', 'act', 'verify', 'judge',
                             'explore', 'reflect'}
        capabilities = strategy.get('capabilities', [])

        return all(cap in valid_capabilities for cap in capabilities)
```

**三层决策模式**：
```
第1级：程序快速分析（<50ms）
    ↓
    ├─ 明确匹配（简单场景） → 直接执行
    └─ 需要语义理解 →

第2级：LLM生成策略（2-3秒）
    ↓

第3级：程序验证（<10ms）
    ↓

生成最终策略 + 执行
```

**判断边界**：

| 判断类型 | 适用场景 | 响应时间 | 示例 |
|---------|---------|---------|------|
| **程序判断** | 格式验证、数值比较、简单特征 | <50ms | 密码框检测、分页检测 |
| **规则引擎** | 多条件组合、模式匹配 | <500ms | 复杂规则判断 |
| **LLM判断** | 语义理解、策略生成、权衡决策 | 2-3秒 | 页面类型识别、选择器推荐 |

#### 1.2.4 ContextCompressor（上下文压缩器）

**职责**：压缩长期运行的上下文，避免无限累积

```python
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
        compressed['spec'] = context['spec']  # Spec必须保留

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
```

**压缩策略**：
- 短期任务：不压缩，保留完整上下文
- 长期任务：每5个阶段压缩一次
- 关键阶段：创建状态快照（tag）
- 回滚机制：基于快照回滚

#### 1.2.5 RiskMonitor（风险监控器）

**职责**：实时监控风险，自动预警和处理

```python
class RiskMonitor:
    """风险监控器"""

    def __init__(self):
        self.thresholds = {
            "max_retries": 3,
            "max_iterations": 10,
            "quality_threshold": 0.4,
            "max_context_size": 10 * 1024 * 1024,  # 10MB
            "max_execution_time": 300  # 5分钟
        }

    def check_risk(self, state: dict) -> dict:
        """检查风险并返回预警"""
        warnings = []
        actions = []

        # 检查重试次数
        if state.get("iteration", 0) > self.thresholds["max_retries"]:
            warnings.append("重试次数超过阈值")
            actions.append("trigger_reflection")

        # 检查质量分数
        quality = state.get("quality_score", 1.0)
        if quality < self.thresholds["quality_threshold"]:
            warnings.append(f"质量分数太低: {quality}")
            actions.append("terminate_task")

        # 检查上下文大小
        context_size = len(json.dumps(state))
        if context_size > self.thresholds["max_context_size"]:
            warnings.append("上下文大小超过阈值")
            actions.append("compress_context")

        return {
            "risk_level": "high" if "terminate" in str(actions) else
                         "medium" if warnings else "low",
            "warnings": warnings,
            "recommended_actions": actions,
            "terminate": "terminate_task" in actions
        }
```

**风险分类**：
- 目标偏移风险：Spec不清晰或被修改
- 结构越权风险：执行层尝试修改规则
- 累积技术风险：日志未落地或上下文膨胀
- 质量风险：数据质量持续低于阈值
- 安全风险：恶意代码或越权访问

### 1.3 验证层组件

#### 1.3.1 Verifier（结果验证器）

**职责**：独立验证执行结果的正确性

```python
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
            'score': 1.0 - len(issues) * 0.1
        }

    def verify_code(self, code: str) -> dict:
        """验证生成的代码"""
        try:
            ast.parse(code)
            syntax_valid = True
        except SyntaxError as e:
            syntax_valid = False

        return {
            'valid': syntax_valid,
            'syntax_valid': syntax_valid,
            'issues': [] if syntax_valid else [str(e)]
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
```

#### 1.3.2 EvidenceCollector（证据收集器）

**职责**：收集各阶段的结构化证据

```python
class EvidenceCollector:
    """证据收集器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.evidence_index = {}

    def collect_sense(self, html: str, analysis: dict):
        """收集感知阶段证据"""
        sense_dir = self.output_dir / 'sense'
        sense_dir.mkdir(exist_ok=True)

        with open(sense_dir / 'html_snapshot.html', 'w') as f:
            f.write(html[:100000])  # 限制大小

        with open(sense_dir / 'analysis.json', 'w') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        self.evidence_index['sense'] = str(sense_dir)

    def collect_plan(self, code: str, reasoning: str):
        """收集规划阶段证据"""
        plan_dir = self.output_dir / 'plan'
        plan_dir.mkdir(exist_ok=True)

        with open(plan_dir / 'generated_code.py', 'w') as f:
            f.write(code)

        with open(plan_dir / 'reasoning.md', 'w') as f:
            f.write(reasoning)

        self.evidence_index['plan'] = str(plan_dir)

    def save_index(self):
        """保存证据索引"""
        with open(self.output_dir / 'evidence_index.json', 'w') as f:
            json.dump(self.evidence_index, f, ensure_ascii=False, indent=2)
```

#### 1.3.3 GateDecision（门禁决策器）

**职责**：基于证据做门禁决策

```python
class GateDecision:
    """门禁决策器"""

    def decide(self, state: dict, spec: SpecContract) -> str:
        """门禁决策"""
        completion_gate = CompletionGate()
        gate_passed = completion_gate.check(state, spec)

        if not gate_passed:
            failed_gates = state.get("failed_gates", [])

            # 分类处理
            if "execution_success" in failed_gates:
                return "soal_repair"  # 执行失败，进入修复

            elif "quality_score" in str(failed_gates):
                quality = state.get("quality_score", 0)
                if quality >= 0.4:
                    return "reflect_and_retry"  # 反思重试
                else:
                    return "terminate"  # 质量太低，终止

            else:
                return "terminate"

        return "complete"
```

### 1.4 执行层组件

#### 1.4.1 AgentPool（智能体池）

**职责**：管理7种能力的智能体

```python
class AgentPool:
    """智能体池 - 提议模式"""

    def __init__(self, llm_client: 'LLMClient'):
        self.llm_client = llm_client
        self.agents = {
            'sense': SenseAgent(llm_client),
            'plan': PlanAgent(llm_client),
            'act': ActAgent(llm_client),
            'verify': VerifyAgent(llm_client),
            'judge': JudgeAgent(llm_client),
            'explore': ExploreAgent(llm_client),
            'reflect': ReflectAgent(llm_client)
        }

    async def propose(self, capability: str, context: dict) -> dict:
        """生成提议"""
        if capability not in self.agents:
            raise ValueError(f"Unknown capability: {capability}")

        return await self.agents[capability].propose(context)

    def get_agent(self, capability: str):
        """获取智能体"""
        return self.agents.get(capability)
```

**7种能力**：
1. **Sense（感知）**：分析页面结构，识别特征
2. **Plan（规划）**：生成爬虫代码
3. **Act（行动）**：安全执行代码
4. **Verify（验证）**：评估数据质量
5. **Judge（评判）**：决策是否继续
6. **Explore（探索）**：全站深度探索
7. **Reflect（反思）**：分析失败原因

#### 1.4.2 Executor（代码执行器）

**职责**：在沙箱中安全执行代码

```python
class Executor:
    """代码执行器"""

    def __init__(self, sandbox: 'Sandbox'):
        self.sandbox = sandbox

    async def execute(self, code: str, timeout: int = 60) -> dict:
        """执行代码"""
        try:
            # 在沙箱中执行
            result = await self.sandbox.run(code, timeout=timeout)

            return {
                'success': True,
                'data': result.get('data', []),
                'logs': result.get('logs', []),
                'execution_time': result.get('execution_time', 0)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
```

#### 1.4.3 Sandbox（安全沙箱）

**职责**：提供安全的代码执行环境

```python
class Sandbox:
    """安全沙箱"""

    def __init__(self):
        self.restrictions = {
            'network_access': False,  # 禁止网络访问
            'file_system': 'read_only',  # 只读文件系统
            'max_memory_mb': 512,  # 最大内存
            'max_execution_time': 60  # 最大执行时间
        }

    async def run(self, code: str, timeout: int = 60) -> dict:
        """在沙箱中运行代码"""
        # 1. 代码审查
        self._review_code(code)

        # 2. 资源限制
        self._apply_restrictions()

        # 3. 执行代码
        result = await self._execute_in_isolation(code, timeout)

        return result
```

### 1.5 工具层组件

#### 1.5.1 Browser（浏览器工具）

**职责**：提供网页访问能力

```python
class Browser:
    """浏览器工具 - Playwright封装"""

    async def browse(self, url: str, wait_for: str = 'body', timeout: int = 30000) -> dict:
        """访问网页"""
        pass
```

#### 1.5.2 LLMClient（大模型客户端）

**职责**：提供LLM调用能力

```python
class LLMClient:
    """大模型客户端"""

    async def chat(self, prompt: str, model: str = 'glm-4') -> str:
        """调用LLM"""
        pass
```

#### 1.5.3 Parser（HTML解析器）

**职责**：解析HTML内容

```python
class Parser:
    """HTML解析器 - BeautifulSoup封装"""

    def parse(self, html: str) -> dict:
        """解析HTML"""
        pass
```

#### 1.5.4 Storage（证据存储）

**职责**：存储证据包

```python
class Storage:
    """证据存储"""

    def save(self, data: dict, path: str):
        """保存数据"""
        pass
```

---

## 2. 能力模块化设计

### 2.1 7种基础能力

#### Sense（感知能力）
**输入**：URL, HTML
**输出**：页面类型、选择器、反爬等级
**职责**：分析页面结构、识别特征
**可复用**：是

#### Plan（规划能力）
**输入**：页面分析、选择器
**输出**：Python代码
**职责**：生成爬虫代码
**可复用**：是

#### Act（执行能力）
**输入**：Python代码
**输出**：JSON数据
**职责**：安全执行代码
**可复用**：是

#### Verify（验证能力）
**输入**：执行结果
**输出**：质量分数、问题列表
**职责**：评估数据质量
**可复用**：是

#### Judge（评判能力）
**输入**：质量分数、历史记录
**输出**：是否继续、改进方向
**职责**：决策是否重试
**可复用**：是

#### Explore（探索能力）
**输入**：站点地图、链接队列
**输出**：所有数据
**职责**：深度探索站点
**可复用**：是

#### Reflect（反思能力）
**输入**：失败历史
**输出**：修复策略
**职责**：分析失败原因
**可复用**：是

### 2.2 能力动态组合

```python
CAPABILITY_MATRIX = {
    'simple_static': ['sense', 'plan', 'act', 'verify'],
    'paginated_list': ['sense', 'plan', 'act', 'verify', 'handle_pagination'],
    'spa_application': ['sense', 'handle_spa', 'api_extract', 'verify'],
    'login_required': ['detect_login', 'handle_login', 'sense', 'plan', 'act', 'verify'],
    'strong_anti_bot': ['sense', 'handle_anti_bot', 'slow_plan', 'act', 'verify'],
    'full_site': ['scan', 'explore', 'sense', 'plan', 'act', 'verify'],
}

def compose_capabilities(task_analysis: dict) -> list:
    """动态组合能力"""
    capabilities = ['sense', 'plan', 'act', 'verify']

    requirements = task_analysis.get('special_requirements', [])

    if 'login' in requirements:
        capabilities.insert(0, 'handle_login')

    if 'pagination' in requirements:
        capabilities.append('handle_pagination')

    if 'javascript' in requirements:
        capabilities.insert(1, 'handle_spa')

    return capabilities
```

---

## 3. 流程编排设计

### 3.1 主工作流（单页面爬取）

```
用户输入 (URL + Spec)
    ↓
战略层：加载Spec → 验证合法性 → 冻结契约
    ↓
管理层：创建工作流 → 智能路由决策 → 初始化状态
    ↓
执行层：Sense（感知）→ 生成页面分析
    ↓
验证层：验证感知 → 门禁检查
    ↓
执行层：Plan（规划）→ 生成代码
    ↓
验证层：验证代码 → 语法检查 → 门禁检查
    ↓
执行层：Act（行动）→ 沙箱执行
    ↓
验证层：验证执行 → 检查结果 → 门禁检查
    ↓
执行层：Verify（验证）→ 质量评估
    ↓
验证层：质量门禁 → 检查分数/样本数
    ↓
    ├─ 门禁通过 → 收集证据 → 完成
    └─ 门禁失败 → Judge（评判）→ 决策
                ↓
                ├─ 可修复 → Reflect（反思）→ 回到Plan
                └─ 不可修复 → 终止任务
```

### 3.2 智能路由工作流

```
用户请求 (URL + Goal)
    ↓
┌─────────────────────────────────────┐
│  第1级：程序快速分析（0.1秒）       │
│  - 格式验证                          │
│  - 数值比较                          │
│  - 简单特征识别                      │
└─────────────────────────────────────┘
    ↓
    ├─ 明确匹配（简单场景） → 直接返回策略
    └─ 需要语义理解 →
    ↓
┌─────────────────────────────────────┐
│  第2级：LLM生成策略（2-3秒）        │
│  - 语义理解                          │
│  - 因果推理                          │
│  - 策略生成                          │
│  - 权衡判断                          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  第3级：程序验证（0.01秒）          │
│  - 验证必需字段                      │
│  - 验证能力合理性                    │
│  - 安全兜底                          │
└─────────────────────────────────────┘
    ↓
生成最终策略 + 执行
```

---

## 4. 数据契约设计

### 4.1 Spec 契约

```yaml
spec:
  version: v1
  freeze: true

  goal: "提取科技新闻网站的所有文章"

  constraints:
    - anti_bot_level: medium
    - max_execution_time: 120
    - max_retries: 3

  completion_gate:
    - html_snapshot_exists
    - sense_analysis_valid
    - code_syntax_valid
    - execution_success
    - quality_score >= 0.6
    - sample_count >= 5

  evidence:
    required:
      - spec.yaml
      - sense_report.json
      - generated_code.py
      - execution_log.json
      - quality_report.json
      - final_result.json
```

### 4.2 State 契约

```python
StateContract = {
    'task_id': str,
    'url': str,
    'goal': str,
    'spec': dict,
    'stage': str,
    'iteration': int,
    'created_at': str,
    'updated_at': str,
    'performance_data': dict,
    'failure_history': list,
    'evidence_collected': dict,
    'quality_score': float,
    'routing_decision': dict,
    'gate_passed': bool,
    'failed_gates': list
}
```

---

**总结**：优化后的设计将6层架构简化为5层，通过SmartRouter实现程序+LLM混合判断，既保证效率又保持灵活性，避免agent"死板"的问题。
