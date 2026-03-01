# Self-Crawling Agent - 容器化增强版

## 概述

Self-Crawling Agent 是一个自主的网页数据爬取系统，能够智能地分析页面结构、规划数据提取策略并自主完成爬取任务。此版本增强了容器化支持，支持在Docker容器中运行任务以获得更好的隔离和安全性。

## 核心特性

- **自主性**：系统能够在最少人工干预的情况下完成数据爬取任务
- **智能规划**：使用AI代理来感知页面、规划提取策略、执行提取并验证结果
- **容错能力**：具备错误恢复和策略调整能力
- **容器化支持**：支持在Docker容器中运行任务，提供更好的隔离性

## 架构设计

### 五层架构
1. **战略层**：SpecLoader, PolicyManager, CompletionGate
2. **管理层**：SmartRouter, StateManager, ContextCompressor, RiskMonitor
3. **验证层**：Verifier, EvidenceCollector
4. **执行层**：AgentPool, Executor
5. **工具层**：BrowserTool, LLMClient

### 七阶段执行流程
- Sense（感知）→ Plan（规划）→ Act（执行）→ Verify（验证）→ Gate（门禁）→ Judge（判断）→ Reflect（反思）

## 容器化增强

### 环境检测
系统现在能够自动检测是否在容器环境中运行，并相应调整安全策略：
- 容器环境中使用简化的沙箱（更注重性能）
- 非容器环境中使用严格的沙箱（更注重安全）

### 容器化执行
支持以下两种执行模式：
1. **本地执行**：在当前环境中运行
2. **容器化执行**：在Docker容器中运行（推荐用于生产环境）

## 安装和配置

### 环境要求
- Python 3.9+
- Docker（如果使用容器化执行）
- Node.js 18+（用于Playwright）

### 依赖安装
```bash
pip install -r requirements.txt
playwright install chromium
```

### 构建Docker镜像
```bash
# 使用构建脚本
./build.sh [镜像名] [push]

# 或直接使用docker命令
docker build -t self-crawling-agent:latest .
```

## 使用方法

### 1. 本地执行
```bash
python -m src.main specs/example_task.yaml
```

### 2. 本地执行（带调试）
```bash
python -m src.main specs/example_task.yaml --debug
```

### 3. 容器化执行
```bash
python -m src.main specs/example_task.yaml --docker --memory-limit 2g --timeout 3600
```

### 4. 使用专用运行脚本
```bash
# 容器化执行（推荐）
python run_task.py specs/example_task.yaml

# 本地执行
python run_task.py specs/example_task.yaml --local

# 指定容器参数
python run_task.py specs/example_task.yaml --image my-agent:v1 --memory 4g --cpu 2048
```

## 任务规格定义

任务通过YAML/JSON文件定义，包括：

```yaml
task_id: unique_task_id
task_name: 任务名称
target_url: https://example.com
targets:
  - name: title
    description: 页面标题
    selector: h1
    type: text
completion_criteria:
  min_items: 1
  quality_threshold: 0.8
```

### 目标驱动模式
创新的目标驱动模式允许Spec中仅包含`description`而不必指定`selector`，系统会根据描述智能推断最合适的CSS选择器。

## 沙箱系统

### 三层沙箱架构
1. **DefaultSandbox**：默认沙箱，使用subprocess执行并进行安全检查
2. **ContainerSandbox**：容器内沙箱，适用于已处于容器环境中的情况
3. **DockerSandbox**：Docker沙箱，在独立Docker容器中执行代码

### 动态沙箱选择
- 容器环境中：自动使用简化的沙箱（依赖容器隔离）
- 本地环境中：使用严格的沙箱（代码级安全检查）

## 多LLM提供商支持

系统支持多LLM提供商智能路由：
- **推理任务**：路由到 DeepSeek
- **编码任务**：路由到 GLM

## 状态管理和证据收集

- 完整的状态跟踪和管理
- 详细的执行过程证据收集
- 自动的结果验证和质量评估

## 执行器（Executor）

执行器负责在安全沙箱中执行动态生成的代码。根据运行环境自动选择适当的沙箱策略。

## 容器化优势

1. **完全隔离**：每个任务在独立的容器中运行
2. **资源控制**：精确控制CPU、内存、网络等资源使用
3. **安全性**：利用Docker的内置安全机制
4. **一致性**：开发、测试、生产环境保持一致
5. **可扩展性**：易于水平扩展，支持多任务并行

## 并发执行

容器化架构天然支持并发执行，每个任务运行在独立的容器中，无资源竞争和干扰。

## 开发和贡献

### 代码结构
```
src/
├── main.py          # 主入口
├── core/           # 核心组件
├── agents/         # AI代理
├── executors/      # 代码执行器
├── tools/          # 工具类
└── config/         # 配置管理
specs/              # 任务规格定义
tests/              # 测试代码
```

### 测试
```bash
# 运行单元测试
python -m pytest tests/unit/

# 运行集成测试
python -m pytest tests/integration/
```

## 部署

### 单任务执行
使用容器化执行单个任务，每个任务独立容器，资源隔离。

### 调度系统集成
系统设计支持与外部调度系统集成，每个任务作为独立容器启动。

## 故障排除

### 常见问题
1. **Playwright错误**：确保安装了正确版本的浏览器驱动
2. **内存不足**：增加容器内存限制
3. **网络访问**：检查防火墙和代理设置

### 调试技巧
- 使用 `--debug` 参数启用详细日志
- 检查容器日志：`docker logs <container_id>`
- 验证LLM API密钥配置

## 许可证

[在此处添加许可证信息]