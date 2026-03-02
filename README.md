# Full Self-Crawling Agent

智能网页数据爬取代理系统 - 基于 AI 驱动的自主爬虫

## 重构说明

此项目已经过结构规范化重构，采用现代 Python 项目标准布局。

### 目录结构

```
full-self-crawl-agent/
├── src/                          # 源代码
│   ├── __init__.py
│   ├── main.py                   # CLI 入口
│   ├── orchestrator.py           # 核心编排器
│   ├── pipeline.py               # 单页处理流水线
│   ├── run_mode.py               # 运行模式检测
│   ├── config/                   # 契约配置
│   ├── core/                     # 核心组件
│   ├── agents/                   # 智能体
│   ├── executors/                # 执行器
│   ├── tools/                    # 工具层
│   ├── api/                      # API 接口
│   └── monitoring/               # 监控组件
├── tests/                        # 测试目录
│   ├── unit/                     # 单元测试
│   ├── integration/              # 集成测试
│   ├── e2e/                      # 端到端测试
│   └── specs/                    # 测试用 Spec
├── docs/                         # 文档目录
│   ├── architecture/
│   ├── design/
│   ├── guides/
│   ├── implementation/
│   ├── release/
│   └── testing/
├── specs/                        # Spec 契约目录
│   ├── ecommerce/
│   ├── news/
│   └── templates/
├── examples/                     # 示例 Spec
├── scripts/                      # 工具脚本
├── config/                       # 运行时配置文件
├── .env.example                  # 环境变量模板
├── pyproject.toml                # 项目配置
├── pytest.ini
├── requirements.txt
├── run_task.py                   # 一键执行脚本
├── docker-compose.yml
├── Dockerfile / Dockerfile.dev / Dockerfile.sandbox
├── LICENSE
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 API Key

复制环境变量模板：

```bash
cp .env.example .env
# 然后编辑 .env 文件并添加您的 API 密钥
```

### 运行测试

```bash
pytest tests/
```

## 项目特性

- **契约驱动**：所有行为基于 Spec 契约定义
- **智能路由**：三层决策模式（程序→规则→LLM）
- **证据验证**：全程记录证据，客观验证完成
- **安全执行**：沙箱环境，策略约束
- **自我反思**：失败自动优化

有关更多详细信息，请参阅 `docs/` 目录中的文档。