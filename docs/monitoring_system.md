# 监控与进度追踪系统

此系统为 Self-Crawling Agent 提供了全面的监控和进度追踪功能，特别适用于 Orchestrator 编排场景。

## 核心组件

### 1. 指标收集器 (MetricsCollector)
- 收集 LLM 调用指标（延迟、成功率、token 使用量等）
- 记录页面加载指标
- 跟踪代码执行指标
- 提供指标摘要供 Orchestrator 查询

### 2. 进度追踪器 (ProgressTracker)
- 跟踪任务在各阶段的进度（感知、规划、执行、验证等）
- 提供任务状态转换管理
- 支持多任务并行进度管理
- 与 Orchestrator 集成的进度报告

### 3. 状态管理器集成
- 在原有的 StateManager 中集成了监控功能
- 通过 `update_progress()` 方法更新任务进度
- 通过 `record_*()` 方法记录各种指标
- 保持与原有功能的兼容性

## API 接口

监控系统提供以下查询接口供 Orchestrator 使用：

### 状态查询
- `get_task_status(task_id)` - 获取特定任务状态
- `get_all_task_progress()` - 获取所有任务进度
- `get_current_state()` - 获取当前状态快照

### 指标查询
- `get_metrics_summary()` - 获取指标摘要
- `get_statistics()` - 获取完整统计信息

## 与现有架构的集成

### LLM 客户端
- `MultiLLMClient` 现在接受 `metrics_collector` 参数
- 自动记录所有 LLM 调用的性能指标
- 支持多提供商的指标分离

### Agent 池
- `AgentPool` 现在接受 `metrics_collector` 和 `progress_tracker` 参数
- 自动在每个能力执行前后更新进度
- 与状态管理器协同工作

### 执行器
- `Executor` 现在接受 `metrics_collector` 参数
- 自动记录代码执行的性能指标

## Orchestrator 集成

Orchestrator 可以通过以下方式使用监控系统：

```python
# 获取任务进度
task_progress = agent.progress_tracker.get_task_progress(task_id)

# 获取性能指标
metrics = agent.metrics_collector.get_metrics_summary()

# 获取综合状态
state = agent.state_manager.get_state()
```

## 与主流程的集成

监控系统已经完全集成到主执行流程中：

1. 任务初始化时设置进度为 "initializing"
2. 每个阶段执行时更新相应进度
3. 所有 LLM 调用和代码执行都被记录
4. 错误和异常也被记录在指标中

## 数据结构

### 任务进度状态
- PENDING: 等待处理
- INITIALIZING: 初始化中
- RUNNING: 运行中
- VERIFYING: 验证中
- REFLECTING: 反思中
- COMPLETING: 完成中
- COMPLETED: 已完成
- FAILED: 失败
- CANCELLED: 已取消

### 任务阶段
- SENSE: 感知
- PLAN: 规划
- ACT: 执行
- VERIFY: 验证
- GATE: 门禁检查
- JUDGE: 决策
- REFLECT: 反思
- FINISH: 完成

此监控系统设计简洁，专注于为 Orchestrator 提供必要的运行时信息，而不包含可视化界面等复杂功能。