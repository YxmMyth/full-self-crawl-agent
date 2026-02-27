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
import asyncio
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger('llm')


class ErrorType(str, Enum):
    """错误类型"""
    NETWORK = "network"  # 网络错误（可重试）
    RATE_LIMIT = "rate_limit"  # 速率限制（可重试）
    AUTH = "auth"  # 认证错误（不可重试）
    INVALID_REQUEST = "invalid_request"  # 请求错误（不可重试）
    SERVER = "server"  # 服务器错误（可重试）
    TIMEOUT = "timeout"  # 超时（可重试）
    UNKNOWN = "unknown"  # 未知错误


@dataclass
class LLMError:
    """LLM 错误信息"""
    error_type: ErrorType
    message: str
    status_code: Optional[int] = None
    is_recoverable: bool = True
    retry_after: Optional[int] = None  # 重试等待时间（秒）

    def __str__(self):
        return f"[{self.error_type.value}] {self.message}"


class LLMException(Exception):
    """LLM 异常基类"""

    def __init__(self, error: LLMError):
        self.error = error
        super().__init__(str(error))

    @property
    def is_recoverable(self) -> bool:
        return self.error.is_recoverable


class NetworkException(LLMException):
    """网络错误"""
    pass


class RateLimitException(LLMException):
    """速率限制"""
    pass


class AuthException(LLMException):
    """认证错误"""
    pass


class ServerException(LLMException):
    """服务器错误"""
    pass


class LLMClient:
    """
    LLM 客户端

    支持：
    - 智谱 GLM 系列模型 (glm-4 / glm-4-air / glm-4-flash)
    - 阿里云百炼通义千问模型 (qwen-max / qwen-plus / qwen-turbo)
    - OpenAI 兼容 API
    """

    # 默认重试配置
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0
    DEFAULT_RETRY_MULTIPLIER = 2.0

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
        if api_base:
            if 'dashscope' in api_base:
                self.provider = 'alibaba'
            elif 'deepseek' in api_base:
                self.provider = 'deepseek'
            elif 'bigmodel' in api_base:
                self.provider = 'zhipu'
            else:
                # 根据模型名称推断
                if 'deepseek' in model.lower():
                    self.provider = 'deepseek'
                elif 'qwen' in model.lower():
                    self.provider = 'alibaba'
                elif 'glm' in model.lower():
                    self.provider = 'zhipu'
                else:
                    self.provider = 'unknown'
        else:
            # 无 api_base，根据模型名称推断
            if 'deepseek' in model.lower():
                self.provider = 'deepseek'
            elif 'qwen' in model.lower():
                self.provider = 'alibaba'
            else:
                self.provider = 'zhipu'
        self.client = httpx.AsyncClient(timeout=60.0)
        self.call_count = 0
        self.total_tokens = 0
        self.error_count = 0
        self.retry_count = 0

    def _get_api_url(self, api_base: Optional[str]) -> str:
        """获取 API 地址"""
        if api_base:
            return api_base.rstrip('/') + '/chat/completions'

        # 默认智谱 API
        return 'https://open.bigmodel.cn/api/paas/v4/chat/completions'

    def _classify_error(self, status_code: Optional[int], error_message: str) -> LLMError:
        """分类错误"""
        if status_code == 401 or status_code == 403:
            return LLMError(
                error_type=ErrorType.AUTH,
                message=f"认证失败: {error_message}",
                status_code=status_code,
                is_recoverable=False
            )
        elif status_code == 429:
            return LLMError(
                error_type=ErrorType.RATE_LIMIT,
                message=f"速率限制: {error_message}",
                status_code=status_code,
                is_recoverable=True,
                retry_after=60  # 默认等待60秒
            )
        elif status_code and 500 <= status_code < 600:
            return LLMError(
                error_type=ErrorType.SERVER,
                message=f"服务器错误: {error_message}",
                status_code=status_code,
                is_recoverable=True
            )
        elif status_code and 400 <= status_code < 500:
            return LLMError(
                error_type=ErrorType.INVALID_REQUEST,
                message=f"请求错误: {error_message}",
                status_code=status_code,
                is_recoverable=False
            )
        elif 'timeout' in error_message.lower():
            return LLMError(
                error_type=ErrorType.TIMEOUT,
                message=f"请求超时: {error_message}",
                is_recoverable=True
            )
        elif 'network' in error_message.lower() or 'connection' in error_message.lower():
            return LLMError(
                error_type=ErrorType.NETWORK,
                message=f"网络错误: {error_message}",
                is_recoverable=True
            )
        else:
            return LLMError(
                error_type=ErrorType.UNKNOWN,
                message=f"未知错误: {error_message}",
                status_code=status_code,
                is_recoverable=False
            )

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                      max_tokens: int = 1024, temperature: float = 0.7,
                      top_p: float = 0.7, max_retries: int = None) -> str:
        """
        生成文本

        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            max_tokens: 最大 token 数
            temperature: 温度参数
            top_p: Top-p 采样参数
            max_retries: 最大重试次数

        Returns:
            生成的文本

        Raises:
            LLMException: 当发生不可恢复的错误时
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

        response = await self._call_api_with_retry(
            messages, max_tokens, temperature, top_p, max_retries
        )
        self.call_count += 1

        if response and 'choices' in response:
            self.total_tokens += response.get('usage', {}).get('total_tokens', 0)
            return response['choices'][0]['message']['content']

        return ''

    async def chat(self, messages: List[Dict[str, str]],
                  max_tokens: int = 1024, temperature: float = 0.7,
                  top_p: float = 0.7, max_retries: int = None) -> str:
        """对话接口"""
        response = await self._call_api_with_retry(
            messages, max_tokens, temperature, top_p, max_retries
        )
        self.call_count += 1

        if response and 'choices' in response:
            self.total_tokens += response.get('usage', {}).get('total_tokens', 0)
            return response['choices'][0]['message']['content']

        return ''

    async def _call_api_with_retry(self, messages: List[Dict[str, str]],
                                    max_tokens: int, temperature: float,
                                    top_p: float, max_retries: int = None) -> Optional[Dict[str, Any]]:
        """带重试的 API 调用"""
        max_retries = max_retries or self.DEFAULT_MAX_RETRIES
        last_error = None

        for attempt in range(max_retries):
            try:
                result = await self._call_api(messages, max_tokens, temperature, top_p)
                if result is not None:
                    return result

                # API 返回 None，可能是临时错误
                last_error = LLMError(
                    error_type=ErrorType.UNKNOWN,
                    message="API 返回空响应",
                    is_recoverable=True
                )

            except LLMException as e:
                last_error = e.error
                if not e.is_recoverable:
                    # 不可恢复的错误，直接抛出
                    raise
            except Exception as e:
                last_error = LLMError(
                    error_type=ErrorType.UNKNOWN,
                    message=str(e),
                    is_recoverable=True
                )

            # 如果不是最后一次尝试，等待后重试
            if attempt < max_retries - 1 and last_error.is_recoverable:
                delay = self.DEFAULT_RETRY_DELAY * (self.DEFAULT_RETRY_MULTIPLIER ** attempt)
                if last_error.retry_after:
                    delay = max(delay, last_error.retry_after)
                logger.warning(f"LLM 调用失败，{delay:.1f}秒后重试 (尝试 {attempt + 2}/{max_retries}): {last_error}")
                await asyncio.sleep(delay)
                self.retry_count += 1

        # 所有重试都失败
        self.error_count += 1
        raise LLMException(last_error)

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
                error_text = response.text
                error = self._classify_error(response.status_code, error_text)
                logger.error(f'API Error: {response.status_code} - {error_text}')
                logger.debug(f'Provider: {self.provider}, Model: {self.model}')
                logger.debug(f'URL: {self.api_url}')

                if not error.is_recoverable:
                    raise LLMException(error)

                return None

        except httpx.TimeoutException:
            error = self._classify_error(None, "Request timeout")
            raise LLMException(error)

        except httpx.ConnectError as e:
            error = self._classify_error(None, f"Connection error: {str(e)}")
            raise LLMException(error)

        except LLMException:
            raise

        except Exception as e:
            error = self._classify_error(None, str(e))
            logger.error(f'API Call Error: {str(e)}')
            if not error.is_recoverable:
                raise LLMException(error)
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
            'api_url': self.api_url,
            'error_count': self.error_count,
            'retry_count': self.retry_count
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
