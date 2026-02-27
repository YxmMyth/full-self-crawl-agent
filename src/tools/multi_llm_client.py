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
    - 推理任务 -> DeepSeek (reasoning_client)
    - 编码任务 -> GLM (coding_client)

    三层回退机制：
    1. 优先使用指定类型的客户端
    2. 如果失败，尝试使用另一个客户端
    3. 如果都失败，抛出异常
    """

    def __init__(
        self,
        reasoning_client: Optional[LLMClient] = None,
        coding_client: Optional[LLMClient] = None,
        default_client: Optional[LLMClient] = None
    ):
        """
        初始化多提供商客户端

        Args:
            reasoning_client: 推理任务客户端 (DeepSeek)
            coding_client: 编码任务客户端 (GLM)
            default_client: 默认客户端 (用于通用任务和回退)
        """
        self.reasoning_client = reasoning_client
        self.coding_client = coding_client
        self.default_client = default_client or reasoning_client or coding_client

        # 统计信息
        self.reasoning_calls = 0
        self.coding_calls = 0
        self.fallback_calls = 0
        self.error_count = 0

    @classmethod
    def from_env(cls) -> 'MultiLLMClient':
        """
        从环境变量创建多提供商客户端

        环境变量格式:
            DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_API_BASE
            ZHIPU_API_KEY, ZHIPU_MODEL (或 LLM_MODEL), ZHIPU_API_BASE (或 LLM_API_BASE)
        """
        # 加载 DeepSeek 配置 (推理任务)
        deepseek_config = ProviderConfig.from_env_prefix('DEEPSEEK')

        # 加载智谱配置 (编码任务)
        zhipu_api_key = os.getenv('ZHIPU_API_KEY')
        zhipu_model = os.getenv('ZHIPU_MODEL') or os.getenv('LLM_MODEL', 'glm-4')
        zhipu_api_base = os.getenv('ZHIPU_API_BASE') or os.getenv('LLM_API_BASE')

        reasoning_client = None
        coding_client = None

        # 创建 DeepSeek 客户端 (推理)
        if deepseek_config:
            reasoning_client = CachedLLMClient(
                api_key=deepseek_config.api_key,
                model=deepseek_config.model or 'deepseek-reasoner',
                api_base=deepseek_config.api_base
            )
            logger.info(f"DeepSeek 推理客户端已初始化: {deepseek_config.model or 'deepseek-reasoner'}")
        else:
            logger.warning("未配置 DEEPSEEK_API_KEY，推理任务将使用智谱客户端")

        # 创建智谱客户端 (编码)
        if zhipu_api_key:
            coding_client = CachedLLMClient(
                api_key=zhipu_api_key,
                model=zhipu_model,
                api_base=zhipu_api_base
            )
            logger.info(f"智谱编码客户端已初始化: {zhipu_model}")
        else:
            logger.warning("未配置 ZHIPU_API_KEY，编码任务将使用 DeepSeek 客户端")

        # 确保至少有一个客户端可用
        if not reasoning_client and not coding_client:
            raise ValueError("至少需要配置一个 LLM 提供商 (DEEPSEEK_API_KEY 或 ZHIPU_API_KEY)")

        return cls(
            reasoning_client=reasoning_client,
            coding_client=coding_client,
            default_client=coding_client or reasoning_client  # 默认使用编码客户端
        )

    async def reason(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        推理任务 - 使用 DeepSeek

        适用于：分析、规划、判断、反思等需要深度推理的任务

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数传递给 LLMClient.generate

        Returns:
            生成的文本
        """
        self.reasoning_calls += 1

        # 优先使用推理客户端
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

        # 回退到编码客户端
        if self.coding_client:
            self.fallback_calls += 1
            logger.info("[回退] 使用智谱客户端处理推理任务")
            return await self.coding_client.generate(
                prompt, system_prompt=system_prompt, **kwargs
            )

        # 最后回退到默认客户端
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
        编码任务 - 使用 GLM

        适用于：生成代码、代码补全等编码相关任务

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数传递给 LLMClient.generate

        Returns:
            生成的代码
        """
        self.coding_calls += 1

        # 优先使用编码客户端
        if self.coding_client:
            try:
                result = await self.coding_client.generate(
                    prompt, system_prompt=system_prompt, **kwargs
                )
                logger.debug(f"[GLM] 编码任务完成")
                return result
            except LLMException as e:
                logger.warning(f"GLM 调用失败: {e}，尝试使用备选客户端")
                self.error_count += 1

        # 回退到推理客户端
        if self.reasoning_client:
            self.fallback_calls += 1
            logger.info("[回退] 使用 DeepSeek 客户端处理编码任务")
            return await self.reasoning_client.generate(
                prompt, system_prompt=system_prompt, **kwargs
            )

        # 最后回退到默认客户端
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
            # 通用任务使用默认客户端
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
        # 根据任务类型选择客户端
        if task_type == TaskType.REASONING:
            client = self.reasoning_client or self.coding_client or self.default_client
        elif task_type == TaskType.CODING:
            client = self.coding_client or self.reasoning_client or self.default_client
        else:
            client = self.default_client

        if client:
            return await client.chat(messages, **kwargs)

        raise RuntimeError("无可用的 LLM 客户端")

    async def close(self):
        """关闭所有客户端"""
        if self.reasoning_client:
            await self.reasoning_client.close()
        if self.coding_client:
            await self.coding_client.close()
        if self.default_client and self.default_client not in [self.reasoning_client, self.coding_client]:
            await self.default_client.close()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'reasoning_calls': self.reasoning_calls,
            'coding_calls': self.coding_calls,
            'fallback_calls': self.fallback_calls,
            'error_count': self.error_count,
            'providers': {}
        }

        if self.reasoning_client:
            stats['providers']['reasoning'] = {
                'model': self.reasoning_client.model,
                'provider': getattr(self.reasoning_client, 'provider', 'unknown'),
                **self.reasoning_client.get_stats()
            }

        if self.coding_client:
            stats['providers']['coding'] = {
                'model': self.coding_client.model,
                'provider': getattr(self.coding_client, 'provider', 'unknown'),
                **self.coding_client.get_stats()
            }

        return stats

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        cache_stats = {}

        if self.reasoning_client and hasattr(self.reasoning_client, 'get_cache_stats'):
            cache_stats['reasoning'] = self.reasoning_client.get_cache_stats()

        if self.coding_client and hasattr(self.coding_client, 'get_cache_stats'):
            cache_stats['coding'] = self.coding_client.get_cache_stats()

        return cache_stats

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()