# Full Self-Crawling Agent

智能网页数据爬取代理系统 - 基于 AI 驱动的自主爬虫

## 架构设计

五层治理架构：

```
┌─────────────────────────────────────────┐
│         战略层 (Strategy Layer)          │
│  SpecLoader | PolicyManager | CompletionGate │
├─────────────────────────────────────────┤
│          管理层 (Management Layer)        │
│ SmartRouter | StateManager | ContextCompressor │
│            | RiskMonitor                  │
├─────────────────────────────────────────┤
│          验证层 (Verification Layer)      │
│ Verifier | EvidenceCollector | GateDecision │
├─────────────────────────────────────────┤
│          执行层 (Execution Layer)         │
│  AgentPool (7 capabilities) | Executor   │
├─────────────────────────────────────────┤
│           工具层 (Tool Layer)            │
│  Browser | LLMClient | Parser | Storage   │
└─────────────────────────────────────────┘
```

## 核心特性

- **契约驱动**：所有行为基于 Spec 契约定义
- **智能路由**：三层决策模式（程序→规则→LLM）
- **证据验证**：全程记录证据，客观验证完成
- **安全执行**：沙箱环境，策略约束
- **自我反思**：失败自动优化

## 7 种智能体能力

- **Sense** - 感知页面结构和特征
- **Plan** - 规划提取策略
- **Act** - 执行提取操作
- **Verify** - 验证数据质量
- **Judge** - 做出决策判断
- **Explore** - 探索页面链接
- **Reflect** - 反思和优化策略

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 API Key

设置智谱 GLM API Key：

```bash
export ZHIPU_API_KEY="your_api_key_here"
```

### 创建 Spec 契约

创建 `specs/example.yaml`：

```yaml
task_id: "task_001"
task_name: "Example E-commerce Product Crawl"
created_at: "2026-02-25T00:00:00"
version: "1.0"
extraction_type: "single_page"
targets:
  - name: "products"
    fields:
      - name: "title"
        type: "text"
        selector: ".product-title"
        required: true
        description: "Product title"
      - name: "price"
        type: "number"
        selector: ".price"
        required: true
        description: "Product price"
start_url: "https://example.com/products"
max_pages: 100
depth_limit: 3
validation_rules: {}
anti_bot:
  random_delay:
    min: 1
    max: 3
  user_agent_rotation: true
completion_criteria:
  min_items: 10
  quality_threshold: 0.9
```

### 运行任务

```bash
python src/main.py specs/example.yaml --api-key YOUR_API_KEY
```

## 项目结构

```
src/
├── config/          # 契约配置
│   ├── contracts.py    # 契约定义
│   └── loader.py       # 契约加载器
├── core/            # 核心组件
│   ├── policy_manager.py
│   ├── completion_gate.py
│   ├── smart_router.py
│   ├── state_manager.py
│   ├── risk_monitor.py
│   ├── context_compressor.py
│   └── verifier.py
├── agents/          # 智能体
│   └── base.py         # 7种智能体能力
├── executors/       # 执行器
│   └── executor.py     # 代码执行器
├── tools/           # 工具层
│   ├── browser.py      # 浏览器工具
│   ├── llm_client.py   # LLM客户端
│   ├── parser.py       # HTML解析器
│   └── storage.py      # 存储工具
└── main.py          # 主入口
```

## 技术栈

- **AI 模型**: 智谱 GLM-4 / GLM-4-Air / GLM-4-Flash
- **浏览器自动化**: Playwright
- **HTML 解析**: BeautifulSoup
- **编排框架**: LangGraph（规划中）
- **容器化**: Docker（沙箱执行）
- **编程语言**: Python 3.10+

## 配置文件

- `config/settings.json` - 系统配置
- `config/policies.json` - 策略配置
- `specs/` - Spec 契约定义

## 运行模式

### 无头模式（默认）

```bash
python src/main.py spec.yaml --api-key KEY
```

### 可视化模式

```bash
python src/main.py spec.yaml --api-key KEY --headless
```

### 调试模式

```bash
python src/main.py spec.yaml --api-key KEY --debug
```

## 开发指南

### 添加新的智能体能力

1. 在 `src/agents/base.py` 中添加新的 `AgentInterface` 子类
2. 实现 `execute()`、`get_description()`、`can_handle()` 方法
3. 在 `AgentPool` 中注册

### 添加新的工具

1. 在 `src/tools/` 中创建新工具
2. 实现异步方法
3. 在 `SelfCrawlingAgent` 中初始化

### 自定义策略

1. 修改 `src/core/policy_manager.py`
2. 或在 `config/policies.json` 中定义

## 证据存储

任务运行过程中会自动保存：

- **截图**: `evidence/{task_id}/screenshots/`
- **HTML**: `evidence/{task_id}/html/`
- **数据**: `evidence/{task_id}/data/`
- **日志**: `evidence/{task_id}/logs/`
- **指标**: `evidence/{task_id}/metrics/`

## 贡献指南

1. 遵循五层架构设计
2. 所有行为必须基于契约
3. 代码必须通过策略检查
4. 添加单元测试

## 许可证

MIT License
