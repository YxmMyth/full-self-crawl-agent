"""
LLM Circuit Breaker - 断路器模式
用于在 LLM 客户端连续失败时熔断，避免雪崩效应
"""
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any, Dict, Callable, Awaitable
import logging
import time

logger = logging.getLogger('llm.circuit_breaker')


class CircuitState(Enum):
    """断路器状态"""
    CLOSED = "closed"  # 正常状态，请求正常发送
    OPEN = "open"      # 熔断状态，直接拒绝请求
    HALF_OPEN = "half_open"  # 半开状态，试探性放行请求


class CircuitBreaker:
    """
    LLM 客户端断路器
    当连续失败次数达到阈值时，自动切换到 OPEN 状态
    在 OPEN 状态一段时间后，切换到 HALF_OPEN 状态进行试探
    如果试探成功，则回到 CLOSED 状态；如果失败，则回到 OPEN 状态
    """

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 half_open_max_requests: int = 3):
        """
        初始化断路器

        Args:
            failure_threshold: 失败次数阈值，超过此值则开启断路器
            recovery_timeout: 恢复超时时间（秒），OPEN状态下等待多久后进入HALF_OPEN
            half_open_max_requests: 半开状态下最大试探请求数
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_requests = 0
        self.half_open_success = False

        # 统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rejected_requests = 0

    def call_allowed(self) -> bool:
        """检查是否允许发起调用"""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # 检查是否到达恢复时间
            if (self.last_failure_time and
                datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)):
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 0
                self.half_open_success = True
                logger.info(f"LLM Circuit Breaker 进入半开状态，等待试探")
                return True
            else:
                self.rejected_requests += 1
                return False
        elif self.state == CircuitState.HALF_OPEN:
            # 半开状态下只允许有限次数的请求
            if self.half_open_requests < self.half_open_max_requests:
                self.half_open_requests += 1
                return True
            else:
                # 已超出最大试探次数，仍然保持 OPEN
                return False
        return False

    def record_success(self):
        """记录成功调用"""
        self.total_requests += 1
        self.successful_requests += 1

        # 成功调用后重置状态
        self.failure_count = 0
        self.last_failure_time = None

        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"LLM Circuit Breaker 试探成功，进入关闭状态")

        self.state = CircuitState.CLOSED
        self.half_open_requests = 0
        self.half_open_success = True

    def record_failure(self, error: Optional[Exception] = None):
        """记录失败调用"""
        self.total_requests += 1
        self.failed_requests += 1

        self.failure_count += 1
        self.last_failure_time = datetime.now()

        # 检查是否达到失败阈值
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"LLM Circuit Breaker 开启！连续失败次数: {self.failure_count}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'rejected_requests': self.rejected_requests,
            'failure_rate': self.failed_requests / max(1, self.total_requests) if self.total_requests > 0 else 0,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
        }


class CircuitBreakerWrapper:
    """
    LLM 客户端断路器包装器
    为任何 LLM 客户端添加断路器功能
    """

    def __init__(self, client, circuit_breaker: CircuitBreaker):
        self.client = client
        self.circuit_breaker = circuit_breaker
        self.logger = logging.getLogger('llm.circuit_wrapper')

    async def generate(self, *args, **kwargs) -> str:
        """包装 generate 方法"""
        if not self.circuit_breaker.call_allowed():
            self.logger.warning("LLM Circuit Breaker 开启，拒绝调用请求")
            raise Exception(f"LLM Circuit Breaker is OPEN. Current state: {self.circuit_breaker.state.value}")

        try:
            result = await self.client.generate(*args, **kwargs)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            self.circuit_breaker.record_failure(e)
            raise

    async def chat(self, *args, **kwargs) -> str:
        """包装 chat 方法"""
        if not self.circuit_breaker.call_allowed():
            self.logger.warning("LLM Circuit Breaker 开启，拒绝调用请求")
            raise Exception(f"LLM Circuit Breaker is OPEN. Current state: {self.circuit_breaker.state.value}")

        try:
            result = await self.client.chat(*args, **kwargs)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            self.circuit_breaker.record_failure(e)
            raise

    async def close(self):
        """关闭客户端"""
        if hasattr(self.client, 'close'):
            await self.client.close()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        client_stats = {}
        if hasattr(self.client, 'get_stats'):
            client_stats = self.client.get_stats()

        return {
            'client': client_stats,
            'circuit_breaker': self.circuit_breaker.get_stats()
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        if hasattr(self.client, 'get_cache_stats'):
            return self.client.get_cache_stats()
        return {}


class LLMCircuitBreakerManager:
    """
    LLM 断路器管理器
    为 MultiLLMClient 提供断路器管理功能
    """

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 half_open_max_requests: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests

        # 为每种类型的客户端维护独立的断路器
        self.reasoning_circuit_breaker = CircuitBreaker(
            failure_threshold, recovery_timeout, half_open_max_requests
        )
        self.coding_circuit_breaker = CircuitBreaker(
            failure_threshold, recovery_timeout, half_open_max_requests
        )
        self.default_circuit_breaker = CircuitBreaker(
            failure_threshold, recovery_timeout, half_open_max_requests
        )

    def wrap_client(self, client, client_type: str = "default"):
        """包装客户端"""
        if client_type == "reasoning":
            circuit_breaker = self.reasoning_circuit_breaker
        elif client_type == "coding":
            circuit_breaker = self.coding_circuit_breaker
        else:
            circuit_breaker = self.default_circuit_breaker

        return CircuitBreakerWrapper(client, circuit_breaker)

    def get_stats(self) -> Dict[str, Any]:
        """获取所有断路器统计"""
        return {
            'reasoning_breaker': self.reasoning_circuit_breaker.get_stats(),
            'coding_breaker': self.coding_circuit_breaker.get_stats(),
            'default_breaker': self.default_circuit_breaker.get_stats()
        }