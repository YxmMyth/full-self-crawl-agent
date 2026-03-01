"""
指标收集器 - 集中收集系统性能指标供Orchestrator使用
"""
import time
import asyncio
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import threading


@dataclass
class LLMCallMetrics:
    """LLM调用指标"""
    provider: str
    duration: float  # 耗时（毫秒）
    success: bool
    tokens_used: Optional[int] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PageLoadMetrics:
    """页面加载指标"""
    url: str
    duration: float  # 耗时（毫秒）
    success: bool
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class CodeExecutionMetrics:
    """代码执行指标"""
    duration: float  # 耗时（毫秒）
    success: bool
    error: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MetricsCollector:
    """指标收集器 - 收集系统运行时指标供Orchestrator查询"""

    def __init__(self):
        self._lock = threading.RLock()
        self.llm_metrics: list[LLMCallMetrics] = []
        self.page_load_metrics: list[PageLoadMetrics] = []
        self.code_execution_metrics: list[CodeExecutionMetrics] = []
        self.custom_metrics: Dict[str, Any] = {}

    def record_llm_call(self, provider: str, duration: float, success: bool, tokens: Optional[int] = None):
        """记录LLM调用指标"""
        with self._lock:
            metric = LLMCallMetrics(
                provider=provider,
                duration=duration,
                success=success,
                tokens_used=tokens
            )
            self.llm_metrics.append(metric)

    def record_page_load(self, url: str, duration: float, success: bool):
        """记录页面加载指标"""
        with self._lock:
            metric = PageLoadMetrics(
                url=url,
                duration=duration,
                success=success
            )
            self.page_load_metrics.append(metric)

    def record_code_execution(self, duration: float, success: bool, error: Optional[str] = None):
        """记录代码执行指标"""
        with self._lock:
            metric = CodeExecutionMetrics(
                duration=duration,
                success=success,
                error=error
            )
            self.code_execution_metrics.append(metric)

    def set_custom_metric(self, key: str, value: Any):
        """设置自定义指标"""
        with self._lock:
            self.custom_metrics[key] = value

    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要，供Orchestrator查询"""
        with self._lock:
            # 计算LLM调用统计数据
            llm_calls = len(self.llm_metrics)
            llm_successes = len([m for m in self.llm_metrics if m.success])
            llm_avg_duration = sum(m.duration for m in self.llm_metrics) / llm_calls if llm_calls > 0 else 0
            llm_error_rate = (llm_calls - llm_successes) / llm_calls if llm_calls > 0 else 0

            # 计算页面加载统计数据
            page_loads = len(self.page_load_metrics)
            page_load_successes = len([m for m in self.page_load_metrics if m.success])
            page_avg_duration = sum(m.duration for m in self.page_load_metrics) / page_loads if page_loads > 0 else 0
            page_error_rate = (page_loads - page_load_successes) / page_loads if page_loads > 0 else 0

            # 计算代码执行统计数据
            code_executions = len(self.code_execution_metrics)
            code_successes = len([m for m in self.code_execution_metrics if m.success])
            code_avg_duration = sum(m.duration for m in self.code_execution_metrics) / code_executions if code_executions > 0 else 0
            code_error_rate = (code_executions - code_successes) / code_executions if code_executions > 0 else 0

            return {
                'llm_calls': {
                    'total': llm_calls,
                    'successes': llm_successes,
                    'failures': llm_calls - llm_successes,
                    'average_duration': llm_avg_duration,
                    'error_rate': llm_error_rate,
                    'recent_metrics': [
                        {
                            'provider': m.provider,
                            'duration': m.duration,
                            'success': m.success,
                            'tokens': m.tokens_used
                        }
                        for m in self.llm_metrics[-10:]  # 最近10条记录
                    ]
                },
                'page_loads': {
                    'total': page_loads,
                    'successes': page_load_successes,
                    'failures': page_loads - page_load_successes,
                    'average_duration': page_avg_duration,
                    'error_rate': page_error_rate,
                    'recent_metrics': [
                        {
                            'url': m.url,
                            'duration': m.duration,
                            'success': m.success
                        }
                        for m in self.page_load_metrics[-10:]  # 最近10条记录
                    ]
                },
                'code_executions': {
                    'total': code_executions,
                    'successes': code_successes,
                    'failures': code_executions - code_successes,
                    'average_duration': code_avg_duration,
                    'error_rate': code_error_rate,
                    'recent_metrics': [
                        {
                            'duration': m.duration,
                            'success': m.success,
                            'error': m.error
                        }
                        for m in self.code_execution_metrics[-10:]  # 最近10条记录
                    ]
                },
                'custom_metrics': dict(self.custom_metrics),
                'timestamp': datetime.now().isoformat()
            }

    def clear_metrics(self):
        """清除指标数据"""
        with self._lock:
            self.llm_metrics.clear()
            self.page_load_metrics.clear()
            self.code_execution_metrics.clear()
            self.custom_metrics.clear()

    async def record_llm_call_async(self, provider: str, duration: float, success: bool, tokens: Optional[int] = None):
        """异步记录LLM调用指标"""
        self.record_llm_call(provider, duration, success, tokens)

    async def record_page_load_async(self, url: str, duration: float, success: bool):
        """异步记录页面加载指标"""
        self.record_page_load(url, duration, success)

    async def record_code_execution_async(self, duration: float, success: bool, error: Optional[str] = None):
        """异步记录代码执行指标"""
        self.record_code_execution(duration, success, error)