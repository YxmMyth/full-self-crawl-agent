# 需要重新实现的功能详情

## 1. 容器化支持

### Dockerfile 更改
- 添加了 dumb-init 以正确处理僵尸进程
- 创建了非root用户(appuser)以提高安全性
- 调整了包安装顺序
- 添加了权限设置和必要目录创建
- 修改了Playwright安装命令为 `playwright install chromium --with-deps`
- 添加了ENTRYPOINT使用dumb-init
- 将CMD改为ENTRYPOINT + CMD模式

### src/main.py 中的容器化检测
- 添加了 `_detect_container_environment()` 方法用于检测容器环境
- 添加了 `_check_cgroup_container()` 辅助方法
- 添加了 `_get_container_config()` 方法用于获取容器配置
- 添加了 `_create_executor()` 方法根据环境创建合适的执行器
- 在构造函数中添加了容器环境检测逻辑
- 在命令行参数中增加了 `--docker`, `--docker-image`, `--memory-limit`, `--cpu-shares`, `--timeout` 等选项

## 2. 监控与进度追踪系统

### src/agents/base.py 中的更改
- 在 AgentPool.__init__() 中添加了 metrics_collector 和 progress_tracker 参数
- 在 execute_capability() 方法中增加了进度更新逻辑，在开始和结束时都更新进度

### src/main.py 中的监控集成
- 导入了 MetricsCollector、ProgressTracker 等监控组件
- 在 SelfCrawlingAgent.__init__() 中初始化了 metrics_collector 和 progress_tracker
- 修改了 AgentPool 的初始化以传递监控组件
- 在各个阶段（Sense, Plan, Act, Verify, Gate, Judge, Reflect）添加了进度更新调用
- 在命令行主函数中分离了本地执行和容器执行的逻辑

### 进度追踪的详细实现
- 在每个主要步骤前后都添加了进度更新
- 使用了 TaskStage 枚举来标识不同阶段
- 使用了 ProgressStatus 来表示进度状态
- 为每个阶段分配了相应的进度百分比

## 后续实现计划

1. 首先拉取 GitHub 上的最新代码
2. 然后重新应用容器化支持功能
3. 再添加监控与进度追踪系统
4. 测试功能完整性