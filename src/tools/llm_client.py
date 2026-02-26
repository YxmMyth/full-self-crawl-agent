"""
LLM 客户端 - 支持智谱 GLM 和阿里云百炼

用法示例：

# 阿里云百炼
llm = LLMClient(api_key='sk-xxx', model='qwen-max', provider='alibaba')
# 或自动检测
llm = LLMClient(api_key='sk-xxx', model='qwen-max')

# 智谱 GLM
llm = LLMClient(api_key='xxx', model='glm-4', provider='zhipu')

# 自定义 API 地址
llm = LLMClient(
    api_key='sk-xxx',
    model='qwen-max',
    api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
    provider='alibaba'
)
"""

from typing import Dict, Any, List, Optional
import httpx
import json
import os
from datetime import datetime


class LLMClient:
    """
    LLM 客户端

    支持：
    - 智谱 GLM 系列模型 (glm-4 / glm-4-air / glm-4-flash)
    - 阿里云百炼通义千问模型 (qwen-max / qwen-plus / qwen-turbo)
    - OpenAI 兼容 API
    """

    def __init__(self, api_key: str, model: str = 'glm-4',
                 api_base: Optional[str] = None):
        """
        初始化 LLM 客户端

        Args:
            api_key: API 密钥
            model: 模型名称
            api_base: API 基础地址（可选，默认智谱官方端点）
        """
        self.api_key = api_key
        self.model = model
        self.api_url = self._get_api_url(api_base)
        # 检测 provider
        if api_base and 'dashscope' in api_base:
            self.provider = 'alibaba'
        elif 'qwen' in model.lower():
            self.provider = 'alibaba'
        else:
            self.provider = 'zhipu'
        self.client = httpx.AsyncClient(timeout=60.0)
        self.call_count = 0
        self.total_tokens = 0

    def _get_api_url(self, api_base: Optional[str]) -> str:
        """获取 API 地址"""
        if api_base:
            return api_base.rstrip('/') + '/chat/completions'

        # 默认智谱 API
        return 'https://open.bigmodel.cn/api/paas/v4/chat/completions'

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                      max_tokens: int = 1024, temperature: float = 0.7,
                      top_p: float = 0.7) -> str:
        """
        生成文本

        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            max_tokens: 最大 token 数
            temperature: 温度参数
            top_p: Top-p 采样参数

        Returns:
            生成的文本
        """
        messages = []

        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })

        messages.append({
            'role': 'user',
            'content': prompt
        })

        response = await self._call_api(messages, max_tokens, temperature, top_p)
        self.call_count += 1

        if response and 'choices' in response:
            self.total_tokens += response.get('usage', {}).get('total_tokens', 0)
            return response['choices'][0]['message']['content']

        return ''

    async def chat(self, messages: List[Dict[str, str]],
                  max_tokens: int = 1024, temperature: float = 0.7,
                  top_p: float = 0.7) -> str:
        """对话接口"""
        response = await self._call_api(messages, max_tokens, temperature, top_p)
        self.call_count += 1

        if response and 'choices' in response:
            self.total_tokens += response.get('usage', {}).get('total_tokens', 0)
            return response['choices'][0]['message']['content']

        return ''

    async def _call_api(self, messages: List[Dict[str, str]],
                       max_tokens: int, temperature: float,
                       top_p: float) -> Optional[Dict[str, Any]]:
        """调用 API"""
        headers = {
            'Content-Type': 'application/json'
        }

        # 根据提供商设置认证头
        if self.provider == 'zhipu':
            headers['Authorization'] = f'Bearer {self.api_key}'
        elif self.provider == 'alibaba':
            headers['Authorization'] = f'Bearer {self.api_key}'
            headers['X-DashScope-Async'] = 'enable'
        else:
            headers['Authorization'] = f'Bearer {self.api_key}'

        data = {
            'model': self.model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p
        }

        try:
            response = await self.client.post(
                self.api_url,
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f'API Error: {response.status_code} - {response.text}')
                print(f'Provider: {self.provider}, Model: {self.model}')
                print(f'URL: {self.api_url}')
                return None

        except Exception as e:
            print(f'API Call Error: {str(e)}')
            return None

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'call_count': self.call_count,
            'total_tokens': self.total_tokens,
            'model': self.model,
            'provider': self.provider,
            'api_url': self.api_url
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class LLMCache:
    """
    LLM 结果缓存

    缓存相同的提示，避免重复调用
    """

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size

    def get_key(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """生成缓存键"""
        import hashlib
        content = f'{prompt}|{system_prompt or ""}'
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        """获取缓存"""
        if key in self.cache:
            entry = self.cache[key]
            entry['hits'] = entry.get('hits', 0) + 1
            return entry['response']
        return None

    def set(self, key: str, response: str) -> None:
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            # 移除最少使用的项
            least_used = min(self.cache.items(), key=lambda x: x[1].get('hits', 0))
            del self.cache[least_used[0]]

        self.cache[key] = {
            'response': response,
            'timestamp': datetime.now(),
            'hits': 1
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_hits = sum(entry.get('hits', 0) for entry in self.cache.values())
        return {
            'cache_size': len(self.cache),
            'total_hits': total_hits,
            'max_size': self.max_size
        }


class CachedLLMClient(LLMClient):
    """
    带缓存的 LLM 客户端
    """

    def __init__(self, api_key: str, model: str = 'glm-4',
                 api_base: Optional[str] = None):
        super().__init__(api_key, model, api_base)
        self.cache = LLMCache()

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                      max_tokens: int = 1024, temperature: float = 0.7,
                      top_p: float = 0.7) -> str:
        """带缓存的生成"""
        cache_key = self.cache.get_key(prompt, system_prompt)

        # 检查缓存
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # 调用 API
        result = await super().generate(prompt, system_prompt,
                                       max_tokens, temperature, top_p)

        # 缓存结果
        self.cache.set(cache_key, result)

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.cache.get_stats()
