"""
上下文压缩器 - 管理层
压缩上下文以适应 LLM 的 token 限制
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re


class ContextCompressor:
    """
    上下文压缩器

    职责：
    - 压缩历史记录
    - 提取关键信息
    - 汇总统计信息
    - 保留核心证据

    压缩策略：
    - 删除冗余信息
    - 合并相似条目
    - 提取摘要
    - 保留最近和关键信息
    """

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.compression_stats = {
            'original_size': 0,
            'compressed_size': 0,
            'compression_ratio': 0.0
        }

    def compress(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        压缩上下文

        Args:
            context: 原始上下文

        Returns:
            压缩后的上下文
        """
        self.compression_stats['original_size'] = self._estimate_tokens(context)

        compressed = {}

        # 1. 保留核心元数据
        compressed['task_id'] = context.get('task_id')
        compressed['current_state'] = context.get('current_state')
        compressed['timestamp'] = datetime.now().isoformat()

        # 2. 压缩提取的数据
        extracted_data = context.get('extracted_data', [])
        compressed['extracted_data_summary'] = self._compress_extracted_data(extracted_data)

        # 3. 压缩历史记录
        history = context.get('history', [])
        compressed['history_summary'] = self._compress_history(history)

        # 4. 压缩错误和警告
        errors = context.get('errors', [])
        warnings = context.get('warnings', [])
        compressed['errors_summary'] = self._compress_errors(errors)
        compressed['warnings_summary'] = self._compress_warnings(warnings)

        # 5. 压缩资源使用信息
        compressed['resource_usage'] = context.get('resource_usage', {})

        # 6. 保留最近的关键事件
        recent_events = context.get('recent_events', [])
        compressed['recent_events'] = self._compress_recent_events(recent_events)

        # 7. 保留关键统计信息
        compressed['stats'] = self._extract_stats(context)

        self.compression_stats['compressed_size'] = self._estimate_tokens(compressed)
        if self.compression_stats['original_size'] > 0:
            self.compression_stats['compression_ratio'] = (
                self.compression_stats['compressed_size'] / self.compression_stats['original_size']
            )

        return compressed

    def _compress_extracted_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """压缩提取的数据"""
        if not data:
            return {
                'count': 0,
                'sample': [],
                'fields': []
            }

        # 统计信息
        count = len(data)
        fields = list(data[0].keys()) if data else []

        # 保留样本（前3条和后3条）
        sample_size = min(3, count)
        sample = []
        if count <= 6:
            sample = data
        else:
            sample = data[:3] + data[-3:]

        return {
            'count': count,
            'fields': fields,
            'sample': sample,
            'summary': f"已提取 {count} 条数据，包含字段: {', '.join(fields)}"
        }

    def _compress_history(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """压缩历史记录"""
        if not history:
            return {
                'count': 0,
                'recent': []
            }

        count = len(history)

        # 保留最近的10条记录
        recent_count = min(10, count)
        recent = history[-recent_count:]

        return {
            'count': count,
            'recent': recent,
            'summary': f"共有 {count} 条历史记录，显示最近 {recent_count} 条"
        }

    def _compress_errors(self, errors: List[str]) -> Dict[str, Any]:
        """压缩错误信息"""
        if not errors:
            return {
                'count': 0,
                'recent': [],
                'types': {}
            }

        # 统计错误类型
        error_types = {}
        for error in errors:
            error_type = self._categorize_error(error)
            error_types[error_type] = error_types.get(error_type, 0) + 1

        # 保留最近的5个错误
        recent = errors[-5:]

        return {
            'count': len(errors),
            'recent': recent,
            'types': error_types,
            'summary': f"共 {len(errors)} 个错误，类型分布: {error_types}"
        }

    def _compress_warnings(self, warnings: List[str]) -> Dict[str, Any]:
        """压缩警告信息"""
        if not warnings:
            return {
                'count': 0,
                'recent': []
            }

        # 保留最近的5个警告
        recent = warnings[-5:]

        return {
            'count': len(warnings),
            'recent': recent,
            'summary': f"共 {len(warnings)} 个警告"
        }

    def _compress_recent_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """压缩最近事件"""
        if not events:
            return []

        # 保留最近的20个事件
        return events[-20:]

    def _extract_stats(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """提取关键统计信息"""
        stats = {}

        # 提取数据统计
        extracted_data = context.get('extracted_data', [])
        stats['extracted_count'] = len(extracted_data)

        # 错误统计
        errors = context.get('errors', [])
        stats['error_count'] = len(errors)

        # 警告统计
        warnings = context.get('warnings', [])
        stats['warning_count'] = len(warnings)

        # 资源使用
        resource_usage = context.get('resource_usage', {})
        stats['memory_mb'] = resource_usage.get('memory_mb', 0)
        stats['cpu_percent'] = resource_usage.get('cpu_percent', 0)

        # 执行时间
        stats['execution_time_seconds'] = context.get('execution_time_seconds', 0)

        return stats

    def _categorize_error(self, error: str) -> str:
        """分类错误"""
        error_lower = error.lower()

        if 'timeout' in error_lower or 'time out' in error_lower:
            return 'timeout'
        elif 'network' in error_lower or 'connection' in error_lower:
            return 'network'
        elif 'selector' in error_lower or 'element' in error_lower:
            return 'selector'
        elif 'memory' in error_lower or 'oom' in error_lower:
            return 'memory'
        elif 'cpu' in error_lower:
            return 'cpu'
        else:
            return 'other'

    def _estimate_tokens(self, obj: Any) -> int:
        """估算 token 数量（粗略估计）"""
        import json
        text = json.dumps(obj, ensure_ascii=False)
        # 粗略估计：每4个字符约1个token
        return len(text) // 4

    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        return self.compression_stats

    def set_max_tokens(self, max_tokens: int) -> None:
        """设置最大 token 数"""
        self.max_tokens = max_tokens
