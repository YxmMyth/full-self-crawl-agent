# Full Self-Crawling Agent - 宏观架构（五层治理架构 - 优化版）

## 一、重构背景与目标

### 1.1 核心挑战

**现有架构的问题：**
- **过度设计**：元能力层7种策略+5个组件，实际收益低
- **性能损耗**：6层嵌套导致启动慢、调用链长
- **维护复杂**：20+组件、10000+行代码，调试困难
- **扩展受限**：层间耦合深，新增功能成本高

**关键疑问：**
> "很多时候我不知道应该让程序判断还是LLM判断，挺害怕这个agent很死板没有变通性"

### 1.2 优化目标

**保持核心不变：**
- ✅ 契约驱动（Spec-Driven）
- ✅ 证据验证（Evidence-Based）
- ✅ 7种基础能力（Sense/Plan/Act/Verify/Judge/Explore/Reflect）

**核心优化方向：**
1. **简化架构**：从6层→5层，去掉过度复杂的元能力层
2. **平衡判断**：程序判断（规则明确）+ LLM判断（语义复杂）+ 混合决策（最佳实践）
3. **保持灵活**：保留LLM的策略生成能力，不让agent"死板"
4. **提升性能**：减少组件数量，优化调用链

---

## 二、顶层架构设计（优化版）

### 2.1 五层治理架构（精简版）

```
┌─────────────────────────────────────────────────────────────────┐
│                    战略层 (Strategy Layer)                      │
│  职责：定义规则、冻结契约、控制边界                            │
│                                                                 │
│  • SpecLoader       - 规范加载器                               │
│  • PolicyManager    - 策略管理器（边界与约束）                 │
│  • CompletionGate   - 完成门禁检查器                           │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                  管理层 (Orchestration Layer)                   │
│  职责：任务拆解、调度执行、状态维护                            │
│                                                                 │
│  • Orchestrator        - 工作流编排器                          │
│  • StateManager        - 全局状态管理                          │
│  • SmartRouter         - 智能路由（简化元能力层）             │
│  • ContextCompressor   - 上下文压缩器                          │
│  • RiskMonitor         - 风险监控器                            │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                 验证层 (Verification Layer)                     │
│  职责：独立检查、证据验证、自动门禁                            │
│                                                                 │
│  • Verifier           - 结果验证器                             │
│  • EvidenceCollector  - 证据收集器                             │
│  • GateDecision       - 门禁决策器                             │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   执行层 (Execution Layer)                      │
│  职责：生成方案、安全执行、独立评判                            │
│                                                                 │
│  • AgentPool          - 智能体池（提议模式）                   │
│    • SenseAgent       - 感知提议                               │
│    • PlanAgent        - 规划提议                               │
│    • ActAgent         - 行动提议                               │
│    • JudgeAgent       - 评判提议（独立）                       │
│  • Executor           - 代码执行器                             │
│  • Sandbox            - 安全沙箱                               │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     工具层 (Tool Layer)                         │
│  职责：提供原子能力                                            │
│                                                                 │
│  • Browser            - 浏览器工具（Playwright）                │
│  • LLMClient          - 大模型客户端（智谱API）                 │
│  • Parser             - HTML解析器（BeautifulSoup）             │
│  • Storage            - 证据存储（EvidenceBundle）              │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 智能路由设计（SmartRouter）

**核心思想**：将元能力层简化为1个智能路由组件

```python
class SmartRouter:
    """轻量级智能路由（<300行代码）"""

    async def route(self, url: str, goal: str, html: str) -> dict:
        """
        混合判断机制（程序+LLM）：
        1. 程序快速分析（<50ms）
        2. LLM生成策略（核心灵活性）
        3. 程序验证可行性
        """

        # === 第1步：程序快速分析（规则明确）===
        quick_analysis = self._program_analysis(url, html)
        # - 是否有登录表单（正则）
        # - 是否有分页（正则）
        # - 是否是SPA（简单特征）

        # === 第2步：LLM生成策略（语义复杂）===
        strategy = await self._llm_generate_strategy(
            analysis=quick_analysis,
            goal=goal,
            html_sample=html[:5000]
        )
        # LLM根据上下文生成最适合的策略

        # === 第3步：程序验证（安全兜底）===
        if self._validate_strategy(strategy):
            return strategy

        # === 第4步：备选方案 ===
        return self._fallback_strategy()

    def _program_analysis(self, url: str, html: str) -> dict:
        """程序判断：快速特征提取"""
        return {
            'has_login': self._detect_login_form(html),
            'has_pagination': self._detect_pagination(html),
            'is_spa': self._detect_spa(html),
            'anti_bot_level': self._detect_anti_bot(html)
        }

    async def _llm_generate_strategy(self, analysis: dict, goal: str, html_sample: str) -> dict:
        """LLM判断：生成最佳策略"""
        prompt = f"""
        分析结果：{analysis}
        用户目标：{goal}
        HTML片段：{html_sample}

        请生成最适合的爬取策略，包括：
        - 使用哪些能力（从7种能力中选择）
        - 执行步骤
        - 需要注意的事项
        - 预期成功率

        输出格式：JSON
        """
        return await self.llm.chat(prompt)
```

### 2.3 判断边界标准

#### 程序判断（规则明确、效率优先）

```python
# ✅ 适合程序判断的场景：
def should_retry(error_type: str, count: int) -> bool:
    retryable = ['timeout', 'network']
    return error_type in retryable and count < 3

def is_high_risk(quality: float, failures: int) -> bool:
    return quality < 0.4 or failures > 3

def validate_code_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except:
        return False
```

#### LLM判断（语义复杂、需要推理）

```python
# ✅ 适合LLM判断的场景：
async def analyze_page_structure(html: str, goal: str) -> dict:
    prompt = f"分析这个页面，为{goal}推荐选择器..."
    return await llm.chat(prompt)

async def decide_fix_strategy(failure: dict, history: list) -> str:
    prompt = f"分析失败：{failure}\n历史：{history}\n推荐修复方案..."
    return await llm.chat(prompt)

async def judge_continue_or_stop(state: dict) -> str:
    prompt = f"综合评估：{state}\n是否继续？为什么？"
    return await llm.chat(prompt)
```

#### 混合判断（最佳实践）

```python
async def adaptive_decision(context: dict) -> str:
    # 第1级：程序快速过滤
    if context['quality_score'] >= 0.8:
        return 'complete'

    if context['retry_count'] >= 3:
        return 'terminate'

    # 第2级：规则引擎
    if self._rule_check(context):
        return self._rule_decision(context)

    # 第3级：LLM深度分析
    return await self._llm_decision(context)
```

### 2.4 核心设计原则（优化版）

| 原则 | 优化说明 | 实现方式 |
|------|---------|---------|
| **契约驱动** | 保持不变 | Spec冻结、门禁检查 |
| **证据验证** | 保持不变 | 结构化证据、可追溯 |
| **分层制衡** | 简化为5层 | 去掉元能力层的过度设计 |
| **智能路由** | 新增核心 | 程序+LLM混合判断 |
| **上下文管理** | 保持不变 | 压缩算法、状态快照 |
| **风险控制** | 保持不变 | 门禁阈值、监控器 |

---

## 三、优化重点详解

### 3.1 智能路由的三层决策模式

```
┌─────────────────────────────────────────┐
│        用户请求 (URL + Goal)            │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│    第1级：程序快速判断（0.1秒）          │
│    - 格式验证                            │
│    - 数值比较                            │
│    - 简单特征识别                        │
└─────────────────────────────────────────┘
                    ↓
    ├─ 明确匹配 → 直接执行
    └─ 需要语义理解 →
                    ↓
┌─────────────────────────────────────────┐
│    第2级：规则引擎（0.5秒）              │
│    - 多条件组合                          │
│    - 历史模式匹配                        │
│    - 知识库检索                          │
└─────────────────────────────────────────┘
                    ↓
    ├─ 规则匹配 → 执行规则决策
    └─ 复杂推理 →
                    ↓
┌─────────────────────────────────────────┐
│    第3级：LLM深度分析（2-3秒）           │
│    - 语义理解                            │
│    - 因果推理                            │
│    - 策略生成                            │
│    - 权衡判断                            │
└─────────────────────────────────────────┘
                    ↓
         生成最终策略 + 执行
```

### 3.2 7种能力的动态组合

```python
# 能力组合策略矩阵
CAPABILITY_MATRIX = {
    # 简单静态页面
    'simple_static': ['sense', 'plan', 'act', 'verify'],

    # 标准列表页（带分页）
    'paginated_list': ['sense', 'plan', 'act', 'verify', 'handle_pagination'],

    # SPA应用
    'spa_application': ['sense', 'handle_spa', 'api_extract', 'verify'],

    # 需要登录
    'login_required': ['detect_login', 'handle_login', 'sense', 'plan', 'act', 'verify'],

    # 强反爬
    'strong_anti_bot': ['sense', 'handle_anti_bot', 'slow_plan', 'act', 'verify'],

    # 全站探索
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

### 3.3 渐进式探索机制

```python
class ProgressiveExplorer:
    """渐进式探索（从简单到复杂）"""

    STRATEGY_ORDER = [
        ('direct_crawl', 0.95),       # 简单页面
        ('pagination_crawl', 0.85),   # 列表页
        ('api_reverse', 0.70),        # SPA
        ('login_required', 0.40),     # 需登录
        ('headless_browser', 0.50),   # 复杂JS
    ]

    async def explore(self, url: str, goal: str):
        for strategy, expected_rate in self.STRATEGY_ORDER:
            if not self._is_applicable(strategy, url):
                continue

            result = await self._execute_strategy(strategy, url, goal)

            if result.success and result.quality >= 0.6:
                return result

            # 分析失败，决定是否继续
            if not await self._should_continue(result):
                break

        return FailureResult("所有策略失败")
```

---

## 四、核心设计原则

### 4.1 契约驱动（Spec-Driven）
- **所有行为基于Spec定义**：任务目标、约束、门禁条件全部在Spec中明确定义
- **Spec冻结机制**：Spec一旦加载即冻结，不可修改，保证一致性
- **代码生成必须遵循Spec**：生成的代码必须满足Spec中的约束条件

### 4.2 证据验证（Evidence-Based）
- **完成 = 证据满足门禁条件**：不再依赖主观判断，而是客观证据
- **每个阶段输出结构化证据**：HTML快照、分析报告、代码、执行日志、质量报告
- **证据可追溯、可回滚**：所有证据保存在证据包中，支持审计和回滚

### 4.3 分层制衡（Layered Governance）
- **战略层定义规则**：不参与执行，只定义边界
- **管理层拆解调度**：不生成代码，只编排流程
- **验证层独立检查**：不修改规则，只验证证据
- **执行层专注实现**：不定义规则，只生成方案
- **工具层提供能力**：不参与决策，只提供原子操作

### 4.4 智能路由（Smart Routing）
- **程序判断**：规则明确、效率优先（<50ms）
- **LLM判断**：语义复杂、灵活性强（2-3秒）
- **混合判断**：三层决策、平衡最优
- **渐进式探索**：从简单到复杂尝试策略

### 4.5 上下文管理（Context Management）
- **长期任务上下文压缩**：避免上下文无限累积
- **记忆摘要机制**：失败历史、性能数据摘要存储
- **状态快照管理**：关键阶段自动打快照，支持回滚
- **有界上下文**：上下文大小限制，超限时自动压缩

### 4.6 风险控制（Risk Control）
- **阶段性门禁检查**：每个阶段结束时自动检查门禁条件
- **自动风险阈值**：重试次数、迭代次数、质量分数等自动监控
- **强制回滚点**：关键阶段创建快照，支持快速回滚
- **风险预警机制**：实时监控风险，提前预警

---

## 五、系统边界

### 5.1 适用场景

**非常适合**：
- ✅ 公开网站的数据采集
- ✅ 结构化数据提取（新闻、商品、论文等）
- ✅ 需要绕过基础反爬的网站
- ✅ 网站结构频繁变化的场景
- ✅ 多页面自动探索需求
- ✅ 需要可追溯、可审计的场景

**不适合**：
- ❌ 需要登录的网站（特别是验证码）
- ❌ 高度动态的SPA（需要复杂交互）
- ❌ 实时性要求极高的场景
- ❌ 超大规模数据采集（成本高）
- ❌ 需要人工审核的敏感数据

### 5.2 技术边界

**依赖项**：
- LLM API服务（智谱、OpenAI等）
- Docker（沙箱执行环境）
- Playwright（浏览器自动化）
- LangGraph（状态机编排）

**约束条件**：
- 单个任务执行时间：30秒 - 10分钟
- 内存占用：200MB - 1GB
- 网络带宽：取决于页面大小
- LLM调用次数：5-15次/任务
- 证据包大小：10MB - 100MB（取决于截图和日志）

### 5.3 质量边界

**预期成功率**：
| 场景 | 成功率 | 说明 |
|------|--------|------|
| 简单静态页面 | 95%+ | 博客、文档类 |
| 标准列表页 | 85%+ | 新闻、商品列表 |
| 复杂动态页面 | 70%+ | 需要JavaScript渲染 |
| 强反爬网站 | 50%+ | 需要高级反爬策略 |

**失败的主要原因**：
1. LLM生成的代码有逻辑错误（~30%）
2. 网站结构过于复杂（~25%）
3. 反爬措施过于严格（~20%）
4. 网络或超时问题（~15%）
5. 其他（~10%）

---

## 六、关键创新

### 6.1 智能路由（新增）
- **三层决策模式**：程序→规则→LLM
- **渐进式探索**：从简单到复杂尝试
- **能力动态组合**：根据需求精准匹配
- **灵活性保障**：保留LLM的策略生成权

### 6.2 CodeAct 模式
- **从调用工具到创造工具**：LLM生成完整代码，而不是调用预定义函数
- **从有限能力到无限可能**：可以生成任意复杂度的解决方案
- **从固定流程到灵活定制**：每个任务都有定制化的代码

### 6.3 全站探索能力
- **智能站点探索**：自动扫描站点地图，按优先级探索页面
- **动态更新**：发现新链接后动态更新探索队列
- **智能终止**：LLM判断是否满足目标，避免过度探索

### 6.4 LLM 驱动的质量判断
- **上下文感知**：考虑网站类型、用户目标、历史记录
- **语义化判断**：返回"是否需要继续"的语义化决策
- **可解释性**：提供判断理由和改进建议

### 6.5 内生治理结构
- **五层治理架构**：战略层、管理层、验证层、执行层、工具层
- **契约驱动**：Spec定义规则，冻结后不可修改
- **证据验证**：完成基于证据，而非主观判断
- **分层制衡**：各层职责分离，相互制衡

---

## 七、价值与演进

### 7.1 核心价值

**对开发者**：
- 减少80%以上的爬虫开发时间
- 无需维护爬虫代码
- 降低技术门槛
- 可追溯、可审计

**对业务方**：
- 快速获取数据
- 适应网站变化
- 降低运营成本
- 质量可控

**对系统**：
- 自主决策能力
- 持续学习优化
- 可扩展架构
- 内生治理

### 7.2 演进路径

**阶段1：基础自动化**（已完成）
- 单页面爬取
- 基础反爬处理
- 简单质量判断
- 三层架构

**阶段2：契约驱动**（当前）
- **五层治理架构**
- **Spec契约定义**
- **证据验证机制**
- **门禁自动检查**

**阶段3：智能优化**（规划中）
- 智能路由（程序+LLM混合判断）
- 上下文压缩
- 风险自动监控
- 多页面探索

**阶段4：自主学习**（远景）
- 跨任务知识共享
- 自动策略优化
- 预测性数据采集
- 自适应治理

### 7.3 未来方向

**技术层面**：
- 多模态理解（图片、视频内容）
- 实时自适应（动态调整策略）
- 分布式执行（并行爬取）
- 智能Spec生成（基于目标自动生成Spec）

**应用层面**：
- 垂直领域专家（电商、新闻、学术）
- 知识图谱构建
- 数据质量认证
- 合规审计支持

---

**总结**：优化后的架构本质是**用Spec契约代替主观描述，用证据验证代替语言判断，用分层治理代替单智能体，用智能路由实现灵活决策**。通过程序+LLM混合判断，既保证效率又保持灵活性，避免agent"死板"的问题。
