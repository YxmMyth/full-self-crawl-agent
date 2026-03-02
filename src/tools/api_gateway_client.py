"""
统一API网关客户端 - 支持http://45.78.224.156:3000平台
该平台支持超过30个提供商的模型，包括MoonshotAI、OpenAI、Grok、Zhipu、Volcengine、
Cohere、Claude、Gemini、Suno、Minimax、Wenxin、Spark、Qingyan、DeepSeek、Qwen等。
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

from .llm_client import LLMException, ErrorType, LLMError

logger = logging.getLogger('llm')


class APIGatewayProvider(str, Enum):
    """API网关支持的提供商"""
    OPENAI = "openai"
    QWEN = "qwen"
    GPT = "gpt"
    GEMINI = "gemini"
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    ZHIPU = "zhipu"
    MOONSHOT = "moonshot"
    GROK = "grok"
    CUSTOM = "custom"


@dataclass
class APIGatewayConfig:
    """API网关配置"""
    api_key: str
    model: str
    api_base: str = "http://45.78.224.156:3000/v1"
    provider: Optional[APIGatewayProvider] = None
    timeout: int = 60

    @classmethod
    def from_env(cls) -> Optional['APIGatewayConfig']:
        """从环境变量加载配置"""
        api_key = os.getenv('API_GATEWAY_KEY')
        if not api_key:
            api_key = os.getenv('API_GATEWAY_API_KEY')

        if not api_key:
            return None

        model = os.getenv('API_GATEWAY_MODEL', 'gpt-3.5-turbo')
        api_base = os.getenv('API_GATEWAY_API_BASE', 'http://45.78.224.156:3000/v1')

        # 自动识别提供商
        provider = None
        if 'gpt' in model.lower():
            provider = APIGatewayProvider.OPENAI
        elif 'qwen' in model.lower():
            provider = APIGatewayProvider.QWEN
        elif 'claude' in model.lower():
            provider = APIGatewayProvider.CLAUDE
        elif 'gemini' in model.lower():
            provider = APIGatewayProvider.GEMINI
        elif 'deepseek' in model.lower():
            provider = APIGatewayProvider.DEEPSEEK
        elif 'glm' in model.lower():
            provider = APIGatewayProvider.ZHIPU
        elif 'moonshot' in model.lower():
            provider = APIGatewayProvider.MOONSHOT
        elif 'grok' in model.lower():
            provider = APIGatewayProvider.GROK
        elif 'claude-opus' in model.lower() or 'claude-sonnet' in model.lower() or 'claude-haiku' in model.lower():
            provider = APIGatewayProvider.CLAUDE

        return cls(
            api_key=api_key,
            model=model,
            api_base=api_base.rstrip('/'),
            provider=provider
        )


class APIGatewayClient:
    """
    统一API网关客户端

    支持通过单一入口访问多个LLM提供商的模型：
    - http://45.78.224.156:3000/v1/chat/completions
    - 兼容OpenAI格式的API调用
    """

    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0
    DEFAULT_RETRY_MULTIPLIER = 2.0

    def __init__(self, config: APIGatewayConfig):
        """
        初始化API网关客户端

        Args:
            config: API网关配置
        """
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout)
        self.call_count = 0
        self.total_tokens = 0
        self.error_count = 0
        self.retry_count = 0
        logger.info(f"API Gateway Client initialized with model: {config.model}, provider: {config.provider}")

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

    async def chat(self, messages,
                   max_tokens: int = 1024, temperature: float = 0.7,
                   top_p: float = 0.7, max_retries: int = None, **kwargs) -> str:
        """对话接口（自动兼容 string / List[Dict] / 多模态 content）"""
        messages = self._sanitize_messages(messages)
        response = await self._call_api_with_retry(
            messages, max_tokens, temperature, top_p, max_retries
        )
        self.call_count += 1

        if response and 'choices' in response:
            self.total_tokens += response.get('usage', {}).get('total_tokens', 0)
            return response['choices'][0]['message']['content']

        return ''

    @staticmethod
    def _sanitize_messages(messages) -> List[Dict[str, str]]:
        """规范化 messages 格式，确保兼容 OpenAI API 标准。"""
        if isinstance(messages, str):
            return [{'role': 'user', 'content': messages}]

        if not isinstance(messages, list):
            return [{'role': 'user', 'content': str(messages)}]

        sanitized = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get('content', '')
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                content = '\n'.join(text_parts) if text_parts else str(content)
            sanitized.append({
                'role': msg.get('role', 'user'),
                'content': str(content),
            })
        return sanitized or [{'role': 'user', 'content': ''}]

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
                    message="API Gateway returned empty response",
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
                logger.warning(f"API Gateway call failed, retrying in {delay:.1f}s (attempt {attempt + 2}/{max_retries}): {last_error}")
                await asyncio.sleep(delay)
                self.retry_count += 1

        # 所有重试都失败
        self.error_count += 1
        raise LLMException(last_error)

    async def _call_api(self, messages: List[Dict[str, str]],
                        max_tokens: int, temperature: float,
                        top_p: float) -> Optional[Dict[str, Any]]:
        """调用API网关"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.config.api_key}'
        }

        # 为claude模型特别处理参数，因为它不支持同时设置temperature和top_p
        data = {
            'model': self.config.model,
            'messages': messages,
            'max_tokens': max_tokens
        }

        # 检查是否为claude模型
        if (self.config.provider == APIGatewayProvider.CLAUDE or
            'claude' in self.config.model.lower() or
            'claude-opus' in self.config.model.lower() or
            'claude-sonnet' in self.config.model.lower() or
            'claude-haiku' in self.config.model.lower()):
            # 对于Claude模型，仅使用temperature
            data['temperature'] = temperature
        else:
            # 对于其他模型，使用全部参数
            data.update({
                'temperature': temperature,
                'top_p': top_p
            })

        try:
            api_url = f"{self.config.api_base}/chat/completions"
            logger.debug(f"Calling API Gateway: {api_url} with model {self.config.model}")

            response = await self.client.post(
                api_url,
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                result = response.json()
                logger.debug(f"API Gateway response received, tokens: {result.get('usage', {}).get('total_tokens', 'unknown')}")
                return result
            else:
                error_text = response.text
                error = self._classify_error(response.status_code, error_text)
                logger.error(f'API Gateway Error: {response.status_code} - {error_text}')

                if not error.is_recoverable:
                    raise LLMException(error)

                return None

        except httpx.TimeoutException:
            error = self._classify_error(None, "API Gateway request timeout")
            raise LLMException(error)

        except httpx.ConnectError as e:
            error = self._classify_error(None, f"API Gateway connection error: {str(e)}")
            raise LLMException(error)

        except LLMException:
            raise

        except Exception as e:
            error = self._classify_error(None, str(e))
            logger.error(f'API Gateway Call Error: {str(e)}')
            if not error.is_recoverable:
                raise LLMException(error)
            return None

    def _classify_error(self, status_code: Optional[int], error_message: str) -> LLMError:
        """分类API网关错误"""
        if status_code == 401 or status_code == 403:
            return LLMError(
                error_type=ErrorType.AUTH,
                message=f"API Gateway authentication failed: {error_message}",
                status_code=status_code,
                is_recoverable=False
            )
        elif status_code == 429:
            return LLMError(
                error_type=ErrorType.RATE_LIMIT,
                message=f"API Gateway rate limit exceeded: {error_message}",
                status_code=status_code,
                is_recoverable=True,
                retry_after=60  # 默认等待60秒
            )
        elif status_code and 500 <= status_code < 600:
            return LLMError(
                error_type=ErrorType.SERVER,
                message=f"API Gateway server error: {error_message}",
                status_code=status_code,
                is_recoverable=True
            )
        elif status_code and 400 <= status_code < 500:
            return LLMError(
                error_type=ErrorType.INVALID_REQUEST,
                message=f"API Gateway invalid request: {error_message}",
                status_code=status_code,
                is_recoverable=False
            )
        elif 'timeout' in error_message.lower():
            return LLMError(
                error_type=ErrorType.TIMEOUT,
                message=f"API Gateway request timeout: {error_message}",
                is_recoverable=True
            )
        elif 'network' in error_message.lower() or 'connection' in error_message.lower():
            return LLMError(
                error_type=ErrorType.NETWORK,
                message=f"API Gateway network error: {error_message}",
                is_recoverable=True
            )
        else:
            return LLMError(
                error_type=ErrorType.UNKNOWN,
                message=f"API Gateway unknown error: {error_message}",
                status_code=status_code,
                is_recoverable=False
            )

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'call_count': self.call_count,
            'total_tokens': self.total_tokens,
            'model': self.config.model,
            'provider': self.config.provider.value if self.config.provider else 'unknown',
            'api_url': self.config.api_base,
            'error_count': self.error_count,
            'retry_count': self.retry_count
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class CachedAPIGatewayClient(APIGatewayClient):
    """
    带缓存的API网关客户端
    """
    def __init__(self, config: APIGatewayConfig):
        super().__init__(config)
        # 动态导入LLMCache以避免循环依赖
        from .llm_client import LLMCache
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