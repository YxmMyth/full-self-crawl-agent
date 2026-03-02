# 快速入门指南

## 一键启动容器化爬取任务

### 1. 准备环境
```bash
# 确保已安装依赖
pip install -r requirements.txt
playwright install chromium

# 构建Docker镜像
./build.sh
```

### 2. 创建任务规格
创建一个 `my_task.yaml` 文件：
```yaml
task_id: my_example_task
task_name: "示例爬取任务"
target_url: "https://quotes.toscrape.com/"
targets:
  - name: "quote"
    description: "名人名言"
    selector: ".quote .text"
    type: "text"
  - name: "author"
    description: "作者姓名"
    selector: ".quote .author"
    type: "text"
completion_criteria:
  min_items: 5
  quality_threshold: 0.8
```

### 3. 运行任务
```bash
# 方式1：容器化执行（推荐）
python run_task.py my_task.yaml

# 方式2：本地执行
python run_task.py my_task.yaml --local

# 方式3：使用主程序容器化选项
python -m src.main my_task.yaml --docker --memory-limit 2g
```

### 4. 查看结果
任务完成后，结果将保存在 `output/` 目录中。

## 核心概念

### 环境检测
- 系统自动检测是否在容器中运行
- 容器环境中使用简化的安全策略（性能优化）
- 本地环境中使用严格的安全策略（安全优先）

### 一键执行
`run_task.py` 脚本提供了最简单的执行接口：
- 自动检测并选择执行模式
- 自动构建镜像（如果不存在）
- 统一的参数接口

### 资源控制
- 内存限制：`--memory` 参数
- CPU限制：`--cpu` 参数
- 超时控制：`--timeout` 参数

## 最佳实践

### 生产环境
1. 使用容器化执行以确保隔离
2. 设置适当的资源限制
3. 使用专门的镜像标签
4. 监控资源使用情况

### 开发环境
1. 可以使用本地模式加快开发速度
2. 利用调试模式查看详细日志
3. 在部署前用容器模式验证

### 并发执行
每个任务在独立容器中运行，可安全地并行执行多个任务。

## 故障排除

### 镜像不存在
如果遇到镜像不存在的错误，运行：
```bash
./build.sh
```

### 权限问题
确保Docker daemon正在运行且有适当权限。

### 资源不足
增加容器的内存或CPU限制。

## 进阶使用

### 自定义镜像
```bash
python run_task.py my_task.yaml --image my-custom-agent:tag
```

### 环境变量
通过环境变量配置API密钥等：
```bash
export ZHIPU_API_KEY=your_key_here
export DEEPSEEK_API_KEY=your_key_here
```

### 挂载自定义路径
容器内的标准目录结构支持数据持久化。