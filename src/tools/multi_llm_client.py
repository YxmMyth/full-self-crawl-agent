"""
多 LLM 提供商管理器

支持将不同类型的任务路由到不同的 LLM 提供商：
- 推理任务 (reasoning): 使用 DeepSeek Reasoner
- 编码任务 (coding): 使用智谱 GLM

使用方法:
    client = MultiLLMClient.from_env()

    # 推理任务 - 自动使用 DeepSeek
    result = await client.reason("分析页面结构...")

    # 编码任务 - 自动使用 GLM
    code = await client.code("生成 Python 提取代码...")

    # 通用调用 - 保持向后兼容
    result = await client.generate("通用提示...")
"""

from typing import Dict, Any, List, Optional, Literal
import os
import logging
from enum import Enum
from dataclasses import dataclass

from .llm_client import LLMClient, CachedLLMClient, LLMException
from .llm_circuit_breaker import LLMCircuitBreakerManager
from .api_gateway_client import APIGatewayClient, CachedAPIGatewayClient, APIGatewayConfig

logger = logging.getLogger('llm')


class TaskType(str, Enum):
    """任务类型"""
    REASONING = "reasoning"  # 推理任务：分析、规划、判断、反思
    CODING = "coding"        # 编码任务：生成代码
    GENERAL = "general"      # 通用任务：使用默认客户端


@dataclass
class ProviderConfig:
    """提供商配置"""
    api_key: str
    model: str
    api_base: Optional[str] = None
    provider_name: str = "unknown"

    @classmethod
    def from_env_prefix(cls, prefix: str) -> Optional['ProviderConfig']:
        """从环境变量加载配置"""
        api_key = os.getenv(f'{prefix}_API_KEY')
        if not api_key:
            return None

        model = os.getenv(f'{prefix}_MODEL', '')
        api_base = os.getenv(f'{prefix}_API_BASE')

        return cls(
            api_key=api_key,
            model=model,
            api_base=api_base,
            provider_name=prefix.lower()
        )


class MultiLLMClient:
    """
    多 LLM 提供商管理器

    将不同类型的任务路由到不同的 LLM：
    - 推理任务 -> DeepSeek (reasoning_client) 或 API网关 (reasoning_client_v2)
    - 编码任务 -> GLM (coding_client) 或 API网关 (coding_client_v2)

    三层回退机制：
    1. 优先使用指定类型的客户端
    2. 如果失败，尝试使用另一个客户端
    3. 如果都失败，抛出异常
    """

    def __init__(
        self,
        reasoning_client: Optional[LLMClient] = None,
        coding_client: Optional[LLMClient] = None,  # 保留以保持向后兼容
        default_client: Optional[LLMClient] = None,
        reasoning_client_v2: Optional[APIGatewayClient] = None,
        coding_client_v2: Optional[APIGatewayClient] = None,
        default_client_v2: Optional[APIGatewayClient] = None
    ):
        """
        初始化多提供商客户端

        Args:
            reasoning_client: 推理任务客户端 (旧版，如DeepSeek)
            coding_client: 编码任务客户端 (已废弃，仅为向后兼容)
            default_client: 默认客户端 (旧版，用于通用任务和回退)
            reasoning_client_v2: 新版推理任务客户端 (API网关)
            coding_client_v2: 新版编码任务客户端 (API网关)
            default_client_v2: 新版默认客户端 (API网关，用于通用任务和回退)
        """
        # 创建断路器管理器
        self.circuit_breaker_manager = LLMCircuitBreakerManager()

        # 包装旧版客户端以使用断路器
        self.reasoning_client = None
        if reasoning_client:
            self.reasoning_client = self.circuit_breaker_manager.wrap_client(
                reasoning_client, "reasoning"
            )

        # GLM已废弃，编码任务由其他客户端处理
        self.coding_client = None
        if coding_client:
            logger.warning("警告: GLM客户端已废弃，编码任务将使用其他客户端处理")
            self.coding_client = self.circuit_breaker_manager.wrap_client(
                coding_client, "coding"
            )

        self.default_client = None
        if default_client:
            self.default_client = self.circuit_breaker_manager.wrap_client(
                default_client, "default"
            )
        elif reasoning_client:
            self.default_client = self.circuit_breaker_manager.wrap_client(
                reasoning_client, "default"
            )
        elif coding_client:
            self.default_client = self.circuit_breaker_manager.wrap_client(
                coding_client, "default"
            )

        # 包装新版API网关客户端以使用断路器
        self.reasoning_client_v2 = None
        if reasoning_client_v2:
            self.reasoning_client_v2 = self.circuit_breaker_manager.wrap_client(
                reasoning_client_v2, "reasoning_v2"
            )

        self.coding_client_v2 = None
        if coding_client_v2:
            self.coding_client_v2 = self.circuit_breaker_manager.wrap_client(
                coding_client_v2, "coding_v2"
            )

        self.default_client_v2 = None
        if default_client_v2:
            self.default_client_v2 = self.circuit_breaker_manager.wrap_client(
                default_client_v2, "default_v2"
            )
        elif reasoning_client_v2:
            self.default_client_v2 = self.circuit_breaker_manager.wrap_client(
                reasoning_client_v2, "default_v2"
            )
        elif coding_client_v2:
            self.default_client_v2 = self.circuit_breaker_manager.wrap_client(
                coding_client_v2, "default_v2"
            )

        # 统计信息
        self.reasoning_calls = 0
        self.coding_calls = 0
        self.fallback_calls = 0
        self.error_count = 0

    @classmethod
    def from_env(cls) -> 'MultiLLMClient':
        """
        从环境变量创建多提供商客户端

        环境变量优先级:
            API_GATEWAY > OPENAI > ANTHROPIC > OLLAMA > DEEPSEEK
        """
        # 加载API网关配置 - 优先使用
        api_gateway_config = APIGatewayConfig.from_env()
        openai_config = ProviderConfig.from_env_prefix('OPENAI')
        if openai_config and not openai_config.model:
            openai_config.model = 'gpt-4o'

        anthropic_config = ProviderConfig.from_env_prefix('ANTHROPIC')
        if anthropic_config and not anthropic_config.model:
            anthropic_config.model = 'claude-sonnet-4-20250514'

        ollama_base = os.getenv('OLLAMA_API_BASE')
        ollama_model = os.getenv('OLLAMA_MODEL', 'llama3')
        ollama_api_key = os.getenv('OLLAMA_API_KEY', 'ollama')
        ollama_config = None
        if ollama_base or os.getenv('OLLAMA_MODEL'):
            ollama_config = ProviderConfig(
                api_key=ollama_api_key,
                model=ollama_model,
                api_base=ollama_base or 'http://localhost:11434',
                provider_name='ollama'
            )

        deepseek_config = ProviderConfig.from_env_prefix('DEEPSEEK')
        if deepseek_config and not deepseek_config.model:
            deepseek_config.model = 'deepseek-reasoner'

        reasoning_client = None
        coding_client = None
        default_client = None
        reasoning_client_v2 = None
        coding_client_v2 = None
        default_client_v2 = None

        if api_gateway_config:
            logger.info(f"使用API网关配置: {api_gateway_config.model}")
            reasoning_client_v2 = CachedAPIGatewayClient(api_gateway_config)
            coding_client_v2 = CachedAPIGatewayClient(api_gateway_config)
            default_client_v2 = coding_client_v2 or reasoning_client_v2
            return cls(
                reasoning_client=None,
                coding_client=None,
                default_client=None,
                reasoning_client_v2=reasoning_client_v2,
                coding_client_v2=coding_client_v2,
                default_client_v2=default_client_v2
            )

        provider_chain = [openai_config, anthropic_config, ollama_config, deepseek_config]
        primary = next((cfg for cfg in provider_chain if cfg and cfg.api_key), None)
        fallback = deepseek_config if primary is not deepseek_config else None

        if primary:
            reasoning_client = CachedLLMClient(
                api_key=primary.api_key,
                model=primary.model,
                api_base=primary.api_base
            )
            default_client = reasoning_client
            logger.info(f"初始化主 LLM 提供商: {primary.provider_name} ({primary.model})")

        if fallback and fallback.api_key:
            coding_client = CachedLLMClient(
                api_key=fallback.api_key,
                model=fallback.model or 'deepseek-reasoner',
                api_base=fallback.api_base
            )
            logger.info("DeepSeek 已作为回退客户端初始化")

        if not reasoning_client and not coding_client:
            raise ValueError(
                "至少需要配置一个 LLM 提供商 (API_GATEWAY_KEY / OPENAI_API_KEY / "
                "ANTHROPIC_API_KEY / OLLAMA_API_BASE / DEEPSEEK_API_KEY)"
            )

        return cls(
            reasoning_client=reasoning_client,
            coding_client=coding_client,
            default_client=default_client or reasoning_client or coding_client,
            reasoning_client_v2=None,
            coding_client_v2=None,
            default_client_v2=None
        )

    async def reason(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        推理任务 - 优先使用 API网关，然后是 DeepSeek

        适用于：分析、规划、判断、反思等需要深度推理的任务

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数传递给 LLMClient.generate

        Returns:
            生成的文本
        """
        self.reasoning_calls += 1

        # 优先使用新版API网关推理客户端
        if self.reasoning_client_v2:
            try:
                result = await self.reasoning_client_v2.generate(
                    prompt, system_prompt=system_prompt, **kwargs
                )
                logger.debug(f"[API-Gateway] 推理任务完成")
                return result
            except Exception as e:
                logger.warning(f"API网关推理调用失败: {e}，尝试使用备选客户端")
                # 注意：这里不增加error_count，因为这是正常的回退逻辑
                pass

        # 其次使用原版推理客户端
        if self.reasoning_client:
            try:
                result = await self.reasoning_client.generate(
                    prompt, system_prompt=system_prompt, **kwargs
                )
                logger.debug(f"[DeepSeek] 推理任务完成")
                return result
            except LLMException as e:
                logger.warning(f"DeepSeek 调用失败: {e}，尝试使用备选客户端")
                self.error_count += 1
            except Exception as e:
                # 检查是否是因为断路器开启
                if "Circuit Breaker is OPEN" in str(e):
                    logger.warning(f"DeepSeek 断路器开启: {e}")
                else:
                    logger.warning(f"DeepSeek 调用失败: {e}")
                self.error_count += 1

        # 回退到新版编码客户端
        if self.coding_client_v2:
            self.fallback_calls += 1
            logger.info("[回退] 使用API网关编码客户端处理推理任务")
            return await self.coding_client_v2.generate(
                prompt, system_prompt=system_prompt, **kwargs
            )

        # 回退到原版编码客户端
        if self.coding_client:
            self.fallback_calls += 1
            logger.info("[回退] 使用智谱客户端处理推理任务")
            return await self.coding_client.generate(
                prompt, system_prompt=system_prompt, **kwargs
            )

        # 最后回退到新版默认客户端
        if self.default_client_v2:
            self.fallback_calls += 1
            return await self.default_client_v2.generate(
                prompt, system_prompt=system_prompt, **kwargs
            )

        # 最后的最后回退到原版默认客户端
        if self.default_client:
            self.fallback_calls += 1
            return await self.default_client.generate(
                prompt, system_prompt=system_prompt, **kwargs
            )

        raise RuntimeError("无可用的 LLM 客户端")

    async def code(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        编码任务 - 优先使用 API网关

        适用于：生成代码、代码补全等编码相关任务

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数传递给 LLMClient.generate

        Returns:
            生成的代码
        """
        self.coding_calls += 1

        # 优先使用新版API网关编码客户端
        if self.coding_client_v2:
            try:
                result = await self.coding_client_v2.generate(
                    prompt, system_prompt=system_prompt, **kwargs
                )
                logger.debug(f"[API-Gateway] 编码任务完成")
                return result
            except Exception as e:
                logger.warning(f"API网关编码调用失败: {e}，尝试使用备选客户端")
                # 注意：这里不增加error_count，因为这是正常的回退逻辑
                pass

        # 回退到新版推理客户端（可能更适合编码任务）
        if self.reasoning_client_v2:
            self.fallback_calls += 1
            logger.info("[回退] 使用API网关推理客户端处理编码任务")
            return await self.reasoning_client_v2.generate(
                prompt, system_prompt=system_prompt, **kwargs
            )

        # 回退到原版推理客户端
        if self.reasoning_client:
            self.fallback_calls += 1
            logger.info("[回退] 使用 DeepSeek 客户端处理编码任务")
            return await self.reasoning_client.generate(
                prompt, system_prompt=system_prompt, **kwargs
            )

        # 最后回退到新版默认客户端
        if self.default_client_v2:
            self.fallback_calls += 1
            return await self.default_client_v2.generate(
                prompt, system_prompt=system_prompt, **kwargs
            )

        # 最后的最后回退到原版默认客户端
        if self.default_client:
            self.fallback_calls += 1
            return await self.default_client.generate(
                prompt, system_prompt=system_prompt, **kwargs
            )

        raise RuntimeError("无可用的 LLM 客户端")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_type: TaskType = TaskType.GENERAL,
        **kwargs
    ) -> str:
        """
        通用生成接口 - 支持任务类型路由

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            task_type: 任务类型 (reasoning/coding/general)
            **kwargs: 其他参数

        Returns:
            生成的文本
        """
        if task_type == TaskType.REASONING:
            return await self.reason(prompt, system_prompt=system_prompt, **kwargs)
        elif task_type == TaskType.CODING:
            return await self.code(prompt, system_prompt=system_prompt, **kwargs)
        else:
            # 通用任务优先使用新版API网关默认客户端
            if self.default_client_v2:
                try:
                    result = await self.default_client_v2.generate(
                        prompt, system_prompt=system_prompt, **kwargs
                    )
                    logger.debug(f"[API-Gateway] 通用任务完成")
                    return result
                except Exception as e:
                    logger.warning(f"API网关通用调用失败: {e}，尝试使用备选客户端")
                    # 注意：这里不增加error_count，因为这是正常的回退逻辑
                    pass

            # 回退到原版默认客户端
            if self.default_client:
                return await self.default_client.generate(
                    prompt, system_prompt=system_prompt, **kwargs
                )
            raise RuntimeError("无可用的 LLM 客户端")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        task_type: TaskType = TaskType.GENERAL,
        **kwargs
    ) -> str:
        """
        对话接口 - 支持任务类型路由

        Args:
            messages: 消息列表
            task_type: 任务类型
            **kwargs: 其他参数

        Returns:
            生成的文本
        """
        # 根据任务类型选择客户端，优先使用新版API网关客户端
        if task_type == TaskType.REASONING:
            client = self.reasoning_client_v2 or self.reasoning_client or self.default_client_v2 or self.default_client
        elif task_type == TaskType.CODING:
            client = self.coding_client_v2 or self.reasoning_client_v2 or self.reasoning_client or self.default_client_v2 or self.default_client
        else:
            client = self.default_client_v2 or self.default_client

        if client:
            return await client.chat(messages, **kwargs)

        raise RuntimeError("无可用的 LLM 客户端")

    async def close(self):
        """关闭所有客户端"""
        # 关闭新版API网关客户端
        if self.reasoning_client_v2 and hasattr(self.reasoning_client_v2.client, 'close'):
            await self.reasoning_client_v2.client.close()
        if self.coding_client_v2 and hasattr(self.coding_client_v2.client, 'close'):
            await self.coding_client_v2.client.close()
        if (self.default_client_v2 and
            hasattr(self.default_client_v2.client, 'close') and
            self.default_client_v2.client not in [
                getattr(self.reasoning_client_v2, 'client', None),
                getattr(self.coding_client_v2, 'client', None)
            ]):
            await self.default_client_v2.client.close()

        # 关闭原版客户端
        if self.reasoning_client and hasattr(self.reasoning_client.client, 'close'):
            await self.reasoning_client.client.close()
        # GLM已废弃，不需要关闭
        if (self.default_client and
            hasattr(self.default_client.client, 'close') and
            self.default_client.client not in [
                getattr(self.reasoning_client, 'client', None),
                getattr(self.coding_client, 'client', None)  # 实际上不会用到，因为coding_client已废弃
            ]):
            await self.default_client.client.close()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'reasoning_calls': self.reasoning_calls,
            'coding_calls': self.coding_calls,
            'fallback_calls': self.fallback_calls,
            'error_count': self.error_count,
            'providers': {}
        }

        # 添加断路器统计信息
        stats['circuit_breakers'] = self.circuit_breaker_manager.get_stats()

        if self.reasoning_client_v2:
            client_model = getattr(self.reasoning_client_v2.client, 'config', None)
            model_name = getattr(client_model, 'model', 'unknown') if client_model else 'unknown'
            provider_name = getattr(client_model, 'provider', 'unknown') if client_model else 'unknown'

            stats['providers']['reasoning_v2'] = {
                'model': model_name,
                'provider': provider_name,
                **self.reasoning_client_v2.get_stats()
            }

        if self.coding_client_v2:
            client_model = getattr(self.coding_client_v2.client, 'config', None)
            model_name = getattr(client_model, 'model', 'unknown') if client_model else 'unknown'
            provider_name = getattr(client_model, 'provider', 'unknown') if client_model else 'unknown'

            stats['providers']['coding_v2'] = {
                'model': model_name,
                'provider': provider_name,
                **self.coding_client_v2.get_stats()
            }

        if self.reasoning_client:
            stats['providers']['reasoning'] = {
                'model': getattr(self.reasoning_client.client, 'model', 'unknown'),
                'provider': getattr(self.reasoning_client.client, 'provider', 'unknown'),
                **self.reasoning_client.get_stats()
            }

        # GLM已废弃，不再显示其统计信息

        return stats

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        cache_stats = {}

        if self.reasoning_client_v2 and hasattr(self.reasoning_client_v2, 'get_cache_stats'):
            cache_stats['reasoning_v2'] = self.reasoning_client_v2.get_cache_stats()

        if self.coding_client_v2 and hasattr(self.coding_client_v2, 'get_cache_stats'):
            cache_stats['coding_v2'] = self.coding_client_v2.get_cache_stats()

        if self.reasoning_client and hasattr(self.reasoning_client, 'get_cache_stats'):
            cache_stats['reasoning'] = self.reasoning_client.get_cache_stats()

        # GLM已废弃，不再包含其缓存统计信息

        return cache_stats

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
