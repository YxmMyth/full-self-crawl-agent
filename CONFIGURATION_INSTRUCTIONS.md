# API Gateway 配置示例

## 设置环境变量

将以下内容添加到您的 `.env` 文件或设置为环境变量：

```bash
# API Gateway 配置
export API_GATEWAY_KEY="sk-j8qM9nlrE3cpu3mw1wjoGpVBxjDCzmuUGTgEMNhPlpo5Zesb"
export API_GATEWAY_MODEL="claude-opus-4-5-20251101"
export API_GATEWAY_API_BASE="http://45.78.224.156:3000/v1"
```

## 或者在代码中使用

如果您想在代码中直接设置（仅用于测试，请勿在生产环境硬编码密钥）：

```python
import os

# 设置API网关配置
os.environ['API_GATEWAY_KEY'] = 'sk-j8qM9nlrE3cpu3mw1wjoGpVBxjDCzmuUGTgEMNhPlpo5Zesb'
os.environ['API_GATEWAY_MODEL'] = 'claude-opus-4-5-20251101'
os.environ['API_GATEWAY_API_BASE'] = 'http://45.78.224.156:3000/v1'
```

## 通过程序设置

您也可以直接在使用 MultiLLMClient 之前设置：

```python
from src.tools.multi_llm_client import MultiLLMClient

# 确保环境变量已设置后再初始化客户端
client = MultiLLMClient.from_env()
```