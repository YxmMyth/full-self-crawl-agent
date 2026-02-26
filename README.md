# Full Self-Crawling Agent

智能网页数据爬取代理系统 - 基于 AI 驱动的自主爬虫

## 重构说明

此项目已经过结构规范化重构，采用现代 Python 项目标准布局。

### 目录结构

```
full-self-crawl-agent/
├── src/                          # 源代码
│   ├── __init__.py
│   ├── main.py
│   ├── config/                   # 契约配置
│   ├── core/                     # 核心组件
│   ├── agents/                   # 智能体
│   ├── executors/                # 执行器
│   ├── tools/                    # 工具层
│   └── utils/                    # 工具函数
├── tests/                        # 统一测试目录
│   ├── __init__.py
│   ├── unit/                     # 单元测试
│   └── integration/              # 集成测试
├── docs/                         # 统一文档目录
│   ├── README.md
│   ├── architecture/
│   ├── design/
│   ├── guides/
│   └── ...
├── specs/                        # 统一 Spec 契约目录
│   ├── ecommerce/
│   ├── news/
│   └── templates/
├── examples/                     # 示例代码
│   └── quick_start.py
├── scripts/                      # 脚本工具
├── config/                       # 运行时配置文件
├── .env.example                  # 环境变量模板
├── .gitignore
├── pyproject.toml                # 项目配置
├── pytest.ini
├── requirements.txt
├── LICENSE                       # 许可证文件
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