# 使用指南

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入智谱 API Key：

```env
ZHIPU_API_KEY=your_actual_api_key_here
```

### 3. 运行示例

```bash
python src/main.py examples/example_ecommerce.yaml
```

## 创建任务契约

### 基本结构

创建 `specs/my_task.yaml`：

```yaml
task_id: "my_task_001"
task_name: "我的爬取任务"
created_at: "2026-02-25T00:00:00"
version: "1.0"
extraction_type: "single_page"  # single_page/multi_page/pagination/infinite_scroll

targets:
  - name: "data"
    fields:
      - name: "title"
        type: "text"
        selector: ".title"
        required: true
      - name: "price"
        type: "number"
        selector: ".price"
        required: false

start_url: "https://example.com"
max_pages: 100
depth_limit: 3

completion_criteria:
  min_items: 10
  quality_threshold: 0.9
```

### 字段类型

支持的字段类型：

- `text`: 纯文本
- `number`: 数字
- `date`: 日期
- `url`: URL
- `image`: 图片链接
- `html`: HTML 内容
- `raw`: 原始数据

### 提取类型

```yaml
# 单页提取
extraction_type: "single_page"

# 多页提取
extraction_type: "multi_page"

# 分页提取
extraction_type: "pagination"

# 无限滚动
extraction_type: "infinite_scroll"

# 表单提交
extraction_type: "form_submission"
```

## 运行任务

### 基本命令

```bash
python src/main.py specs/my_task.yaml
```

### 参数选项

```bash
# 使用 API Key
python src/main.py specs/my_task.yaml --api-key YOUR_KEY

# 调试模式
python src/main.py specs/my_task.yaml --debug

# 可视化模式（显示浏览器）
python src/main.py specs/my_task.yaml --headless false
```

## 查看结果

任务完成后，结果保存在：

```
evidence/{task_id}/
├── screenshots/    # 截图
├── html/          # HTML 快照
├── data/          # 提取的数据（JSON）
├── logs/          # 日志
└── metrics/       # 性能指标
```

### 数据格式

提取的数据保存为 JSON 格式：

```json
[
  {
    "title": "产品名称",
    "price": 99.99,
    "image": "https://example.com/image.jpg",
    ...
  },
  ...
]
```

## 配置说明

### 系统配置 (`config/settings.json`)

```json
{
  "llm": {
    "model": "glm-4",
    "temperature": 0.7,
    "max_tokens": 1024
  },
  "browser": {
    "headless": true,
    "timeout": 30000
  },
  "storage": {
    "evidence_dir": "./evidence"
  }
}
```

### 策略配置 (`config/policies.json`)

```json
{
  "policies": {
    "code_security": [
      {
        "name": "no_os_system",
        "level": "critical",
        "action": "reject"
      }
    ]
  }
}
```

## 故障排除

### 问题：Playwright 浏览器安装失败

```bash
python -m playwright install chromium
```

### 问题：API 调用失败

检查 `.env` 文件中的 API Key 是否正确。

### 问题：选择器找不到元素

1. 查看证据目录中的 HTML 快照
2. 使用浏览器开发者工具确认选择器
3. 调整选择器或使用更宽松的选择器

### 问题：数据质量低

1. 检查 `completion_criteria.quality_threshold` 阈值
2. 添加字段验证规则
3. 增加必填字段

## 高级用法

### 自定义策略

修改 `config/policies.json`：

```json
{
  "policies": {
    "custom_rules": [
      {
        "name": "custom_rule",
        "level": "warning",
        "condition": "custom_condition",
        "action": "warn"
      }
    ]
  }
}
```

### 使用 Docker

```bash
# 开发环境
docker-compose run dev

# 运行任务
docker-compose run sandbox python src/main.py specs/my_task.yaml
```

### 并行执行

```python
# 多任务并行
tasks = [
    'specs/task1.yaml',
    'specs/task2.yaml',
    'specs/task3.yaml'
]

import asyncio
results = await asyncio.gather(
    *[SelfCrawlingAgent(task).run() for task in tasks]
)
```

## API 参考

### SelfCrawlingAgent

```python
from src.main import SelfCrawlingAgent

agent = SelfCrawlingAgent(
    spec_path='specs/my_task.yaml',
    api_key='YOUR_API_KEY'
)

# 运行任务
result = await agent.run()

# 获取统计
stats = agent.get_stats()

# 停止
await agent.stop()
```

### 浏览器工具

```python
from src.tools.browser import BrowserTool

browser = BrowserTool(headless=True)
await browser.start()
await browser.navigate('https://example.com')
html = await browser.get_html()
screenshot = await browser.take_screenshot()
await browser.stop()
```

### LLM 客户端

```python
from src.tools.llm_client import CachedLLMClient

llm = CachedLLMClient(api_key, model='glm-4')
response = await llm.generate('你好')
stats = llm.get_stats()
```

## 最佳实践

1. **契约优先**: 先定义好 Spec 契约，明确提取目标
2. **逐步测试**: 从小规模开始，逐步扩大范围
3. **证据审查**: 查看证据目录，验证选择器是否正确
4. **质量优先**: 设置合适的质量阈值，确保数据准确
5. **合理限速**: 配置适当的延迟，避免被封禁
6. **错误处理**: 关注错误日志，及时调整策略

## 常见问题

**Q: 如何提取动态加载的内容？**

A: 使用浏览器工具，等待页面加载完成：

```yaml
anti_bot:
  wait_for_selector: ".dynamic-content"
```

**Q: 如何处理登录？**

A: 使用浏览器工具模拟登录：

```python
await browser.fill('#username', 'user')
await browser.fill('#password', 'pass')
await browser.click('#login')
await browser.wait_for_navigation()
```

**Q: 如何导出为 CSV/Excel？**

A: 使用 `DataExport` 工具：

```python
from src.tools.storage import DataExport
DataExport.to_csv(data, 'output.csv')
DataExport.to_excel(data, 'output.xlsx')
```

**Q: 如何处理反爬机制？**

A: 启用反爬策略：

```yaml
anti_bot:
  random_delay:
    min: 1
    max: 3
  user_agent_rotation: true
```

**Q: 能否并行爬取多个网站？**

A: 可以，创建多个任务契约并行执行。
