"""
完整项目结构说明
"""

PROJECT_STRUCTURE = '''
full-self-crawl-agent/
├── src/                          # 源代码
│   ├── __init__.py
│   ├── main.py                   # CLI 入口
│   ├── orchestrator.py           # 核心编排器
│   ├── pipeline.py               # 单页处理流水线
│   ├── run_mode.py               # 运行模式检测
│   ├── config/                   # 契约配置
│   │   ├── __init__.py
│   │   ├── contracts.py          # 契约定义（SpecContract, StateContract）
│   │   ├── loader.py             # 契约加载器
│   │   └── validator.py          # 契约验证器
│   │
│   ├── core/                     # 核心组件（战略层 + 管理层 + 验证层）
│   │   ├── __init__.py
│   │   ├── policy_manager.py     # 策略管理器（战略层）
│   │   ├── completion_gate.py    # 完成门禁检查器（战略层）
│   │   ├── smart_router.py       # 智能路由（管理层）
│   │   ├── state_manager.py      # 状态管理器（管理层）
│   │   ├── context_compressor.py # 上下文压缩器（管理层）
│   │   ├── risk_monitor.py       # 风险监控器（管理层）
│   │   ├── crawl_frontier.py     # 爬取前沿队列
│   │   ├── spec_inferrer.py      # Spec 自动推断
│   │   ├── verifier.py           # 验证器 + 证据收集器（验证层）
│   │   └── logging.py            # 日志配置
│   │
│   ├── agents/                   # 智能体（执行层）
│   │   ├── __init__.py
│   │   ├── base.py               # AgentPool + 7种智能体能力
│   │   ├── sense.py              # SenseAgent - 感知页面
│   │   ├── plan.py               # PlanAgent - 规划策略
│   │   ├── act.py                # ActAgent - 执行提取
│   │   ├── verify.py             # VerifyAgent - 验证质量
│   │   ├── judge.py              # JudgeAgent - 做出决策
│   │   ├── explore.py            # ExploreAgent - 探索链接
│   │   ├── reflect.py            # ReflectAgent - 反思优化
│   │   └── spa_handler.py        # SPA 页面处理
│   │
│   ├── executors/                # 执行器（执行层）
│   │   ├── __init__.py
│   │   └── executor.py           # 代码执行器 + 沙箱
│   │
│   ├── tools/                    # 工具层
│   │   ├── __init__.py
│   │   ├── browser.py            # 浏览器工具（Playwright）
│   │   ├── llm_client.py         # LLM 客户端
│   │   ├── multi_llm_client.py   # 多 LLM 客户端
│   │   ├── llm_circuit_breaker.py # LLM 断路器
│   │   ├── api_gateway_client.py # API 网关客户端
│   │   ├── parser.py             # HTML 解析器（BeautifulSoup）
│   │   ├── downloader.py         # 文件下载器
│   │   └── storage.py            # 存储工具（证据/状态/配置）
│   │
│   ├── api/                      # API 接口
│   │   └── monitoring_api.py     # 监控 API
│   │
│   └── monitoring/               # 监控组件
│       ├── metrics_collector.py  # 指标收集
│       └── progress_tracker.py   # 进度追踪
│
├── tests/                        # 测试
│   ├── unit/                     # 单元测试
│   ├── integration/              # 集成测试
│   ├── e2e/                      # 端到端测试
│   └── specs/                    # 测试用 Spec
│
├── docs/                         # 文档
│   ├── architecture/             # 架构文档
│   ├── design/                   # 设计文档
│   ├── guides/                   # 使用指南
│   ├── implementation/           # 实现文档
│   ├── release/                  # 发布说明
│   └── testing/                  # 测试文档
│
├── config/                       # 运行时配置文件
│   ├── settings.json            # 系统配置
│   └── policies.json            # 策略配置
│
├── specs/                        # Spec 契约定义
│   ├── ecommerce/
│   ├── news/
│   ├── templates/
│   └── test_sites/
│
├── examples/                     # 示例契约
│   ├── example_ecommerce.yaml
│   ├── example_news.yaml
│   └── example_real_estate.yaml
│
├── scripts/                      # 工具脚本
│   ├── initialize.py            # 项目初始化
│   ├── setup.sh                 # 安装脚本
│   └── ...                      # 验证和测试运行脚本
│
├── run_task.py                  # 一键执行脚本
├── docker-compose.yml           # Docker 编排
├── Dockerfile                   # 生产镜像
├── Dockerfile.sandbox           # 沙箱环境
├── Dockerfile.dev               # 开发环境
├── pyproject.toml               # 项目配置
├── pytest.ini                   # 测试配置
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境配置模板
├── .gitignore
├── LICENSE
└── README.md
'''

CORE_COMPONENTS = '''
五层架构组件说明：

┌─────────────────────────────────────────────────────┐
│                   战略层 (Strategy)                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. SpecLoader (src/config/loader.py)              │
│     - 加载 Spec 契约文件（JSON/YAML）               │
│     - 验证契约完整性                                │
│     - 返回冻结的 SpecContract                       │
│                                                     │
│  2. PolicyManager (src/core/policy_manager.py)     │
│     - 管理代码安全策略                              │
│     - 管理资源使用策略                              │
│     - 防止越界行为（如无限递归、代码注入等）         │
│                                                     │
│  3. CompletionGate (src/core/completion_gate.py)   │
│     - 基于证据验证任务完成情况                      │
│     - 评估数据质量                                  │
│     - 独立判断（不依赖代理的主观陈述）               │
│                                                     │
├─────────────────────────────────────────────────────┤
│                   管理层 (Management)                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  4. SmartRouter (src/core/smart_router.py)         │
│     - 三层决策模式：程序→规则→LLM                   │
│     - 路由到 7 种智能体或验证层/执行层组件           │
│     - 平衡效率与灵活性                              │
│                                                     │
│  5. StateManager (src/core/state_manager.py)       │
│     - 维护全局任务状态                              │
│     - 提供状态快照和回滚                            │
│     - 管理状态变更历史                              │
│                                                     │
│  6. ContextCompressor (src/core/context_compressor.py)
│     - 压缩上下文以适应 LLM token 限制               │
│     - 删除冗余信息，保留核心证据                    │
│     - 提取摘要和统计信息                            │
│                                                     │
│  7. RiskMonitor (src/core/risk_monitor.py)         │
│     - 监控代码安全风险                              │
│     - 监控资源使用风险                              │
│     - 实时告警和阈值检查                            │
│                                                     │
├─────────────────────────────────────────────────────┤
│                   验证层 (Verification)              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  8. Verifier (src/core/verifier.py)                │
│     - 验证数据完整性                                │
│     - 检查数据格式                                  │
│     - 评估数据质量                                  │
│                                                     │
│  9. EvidenceCollector (src/core/verifier.py)       │
│     - 收集各种类型的证据                            │
│     - 截图、HTML 快照、提取的数据、性能指标等        │
│     - 持久化存储和导出                              │
│                                                     │
│  10. GateDecision (src/core/verifier.py)           │
│     - 基于验证结果做出决策                          │
│     - 决定继续/重试/终止                            │
│                                                     │
├─────────────────────────────────────────────────────┤
│                   执行层 (Execution)                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  11. AgentPool (src/agents/base.py)                │
│     - 管理 7 种智能体能力                           │
│     - 统一接口执行                                  │
│                                                     │
│  12. Executor (src/executors/executor.py)          │
│     - 在沙箱中安全执行代码                          │
│     - 捕获执行结果和错误                            │
│     - 限制资源使用                                  │
│                                                     │
├─────────────────────────────────────────────────────┤
│                   工具层 (Tools)                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  13. BrowserTool (src/tools/browser.py)            │
│     - Playwright 封装                               │
│     - 页面导航、截图、获取 HTML、点击等              │
│                                                     │
│  14. LLMClient (src/tools/llm_client.py)           │
│     - 智谱 GLM API 客户端                           │
│     - 支持 GLM-4 / GLM-4-Air / GLM-4-Flash          │
│     - 缓存机制                                      │
│                                                     │
│  15. HTMLParser (src/tools/parser.py)              │
│     - BeautifulSoup 封装                            │
│     - CSS 选择器、提取表格/链接/图片等              │
│                                                     │
│  16. Storage (src/tools/storage.py)                │
│     - 证据存储                                      │
│     - 状态存储                                      │
│     - 配置存储                                      │
│                                                     │
└─────────────────────────────────────────────────────┘
'''

DATA_FLOW = '''
核心数据流：

┌─────────────────────────────────────────────────────────────┐
│                         用户输入                             │
│                    (Spec 契约)                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    战略层                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ SpecLoader   │  │PolicyManager │  │CompletionGate│     │
│  │ (加载契约)    │  │ (策略检查)   │  │ (完成验证)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    管理层                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ SmartRouter  │  │StateManager  │  │ContextCompressor│  │
│  │ (智能路由)    │  │ (状态管理)   │  │ (上下文压缩)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
└───────────────────────────┼─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│        ┌────────────────────────────────────────┐          │
│        │          验证层                         │          │
│        │  ┌──────────────┐  ┌──────────────┐    │          │
│        │  │  Verifier    │  │EvidenceCollector │  │          │
│        │  │ (数据验证)    │  │  (证据收集)   │    │          │
│        │  └──────────────┘  └──────────────┘    │          │
│        └────────────────────────────────────────┘          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    执行层                                    │
│  ┌────────────────────────────────────────────┐             │
│  │              AgentPool                      │             │
│  │  ┌────────────────────────────────────┐    │             │
│  │  │Sense│Plan│Act│Verify│Judge│Explore│Reflect│  │             │
│  │  └────────────────────────────────────┘    │             │
│  └────────────────────────────────────────────┘             │
│                         │                                   │
│                         ▼                                   │
│              ┌──────────────────────┐                        │
│              │   Executor + Sandbox │                        │
│              │   (安全代码执行)      │                        │
│              └──────────────────────┘                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    工具层                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ BrowserTool  │  │  LLMClient   │  │ HTMLParser   │     │
│  │ (Playwright) │  │ (智谱 GLM)   │  │(BeautifulSoup)│    │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                 │               │
└─────────┴─────────────────┴─────────────────┴───────────────┘
          │                                  │
          └────────────────┬─────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    存储层                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │EvidenceStorage│  │ StateStorage │  │ ConfigStorage │    │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
'''

if __name__ == '__main__':
    print("="*60)
    print("Full Self-Crawling Agent - 项目结构说明")
    print("="*60)
    print()
    print(PROJECT_STRUCTURE)
    print()
    print("="*60)
    print("核心组件说明")
    print("="*60)
    print(CORE_COMPONENTS)
    print()
    print("="*60)
    print("数据流")
    print("="*60)
    print(DATA_FLOW)
