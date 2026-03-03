"""
全局状态管理器 - 管理层
管理任务的全局状态
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
import asyncio
import copy

@dataclass
class StateSnapshot:
    """状态快照"""
    timestamp: datetime
    state: Dict[str, Any]
    metadata: Dict[str, Any]


class StateManager:
    """
    全局状态管理器

    职责：
    - 维护任务的全局状态
    - 提供状态快照和回滚
    - 管理状态变更历史
    - 支持状态持久化

    状态变更规则：
    - 状态只能通过规范的机制变更
    - 每次状态变更都会创建快照
    - 支持状态回滚
    - 使用锁机制保证原子性

    统一的异步优先 API，所有公共方法都是异步的
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self.current_state: Dict[str, Any] = initial_state or {}
        self.history: List[StateSnapshot] = []
        self.max_history: int = 100  # 最大历史记录数
        self._lock = asyncio.Lock()  # 用于保护状态更新的锁
        self._update_count = 0  # 更新计数器

    async def create_initial_state(self, task_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """创建初始状态（异步版本）"""
        async with self._lock:
            initial_state = {
                'task_id': task_id,
                'spec': spec,
                'stage': 'initialized',
                'iteration': 0,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'performance_data': {},
                'failure_history': [],
                'evidence_collected': {},
                'update_count': 0,
                # 爬取追踪字段
                'visited_urls': [],
                'queue_size': 0,
                'pages_crawled': 0,
                'per_url_results': {},
            }

            self.current_state = initial_state
            return initial_state

    def create_initial_state_sync(self, task_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """创建初始状态（同步版本）"""
        initial_state = {
            'task_id': task_id,
            'spec': spec,
            'stage': 'initialized',
            'iteration': 0,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'performance_data': {},
            'failure_history': [],
            'evidence_collected': {},
            'update_count': 0,
            # 爬取追踪字段
            'visited_urls': [],
            'queue_size': 0,
            'pages_crawled': 0,
            'per_url_results': {},
        }
        self.current_state = initial_state
        return initial_state

    def update_state_sync(self, updates: Dict[str, Any], reason: str = "") -> None:
        """
        同步版本的更新状态方法（用于向后兼容）

        Args:
            updates: 状态更新字典
            reason: 更新原因
        """
        # 创建快照
        self._create_snapshot_sync(reason)

        # 深拷贝更新以避免外部修改
        updates_copy = copy.deepcopy(updates)

        # 更新状态
        self.current_state.update(updates_copy)

        # 更新时间戳和计数器
        self.current_state['timestamp'] = datetime.now().isoformat()
        self.current_state['updated_at'] = datetime.now().isoformat()
        self._update_count += 1
        self.current_state['update_count'] = self._update_count

    def _create_snapshot_sync(self, reason: str = "") -> None:
        """同步版本创建状态快照"""
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            state=copy.deepcopy(self._get_important_state()),  # 深拷贝避免引用问题，优化内存使用
            metadata={'reason': reason, 'update_count': self._update_count}
        )
        self.history.append(snapshot)

        # 限制历史记录数量
        if len(self.history) > self.max_history:
            self.history.pop(0)

    async def update_state(self, updates: Dict[str, Any], reason: str = "") -> None:
        """
        更新状态（异步版本，线程安全）

        Args:
            updates: 状态更新字典
            reason: 更新原因
        """
        async with self._lock:
            # 创建快照
            await self._create_snapshot(reason)

            # 深拷贝更新以避免外部修改
            updates_copy = copy.deepcopy(updates)

            # 更新状态
            self.current_state.update(updates_copy)

            # 更新时间戳和计数器
            self.current_state['timestamp'] = datetime.now().isoformat()
            self.current_state['updated_at'] = datetime.now().isoformat()
            self._update_count += 1
            self.current_state['update_count'] = self._update_count

    async def _create_snapshot(self, reason: str = "") -> None:
        """创建状态快照"""
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            state=copy.deepcopy(self._get_important_state()),  # 深拷贝避免引用问题，优化内存使用
            metadata={'reason': reason, 'update_count': self._update_count}
        )
        self.history.append(snapshot)

        # 限制历史记录数量
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _get_important_state(self) -> Dict[str, Any]:
        """
        获取需要快照的重要状态字段，避免保存大体积数据
        """
        # 返回精简版本的状态，排除体积大的字段
        state_copy = {}
        for key, value in self.current_state.items():
            # 避免保存过大的字段，如HTML快照等
            # 添加对各种大体积数据类型的检查
            if key in ['html_snapshot', 'previous_html', 'large_data_blob']:
                # 对于已知的大体积字段，跳过或进行压缩
                continue
            elif isinstance(value, str) and len(value) > 10000:
                # 跳过超过10KB的字符串
                continue
            elif isinstance(value, list) and len(value) > 1000:
                # 跳过超过1000项的列表
                continue
            elif isinstance(value, dict) and len(str(value)) > 10000:
                # 跳过序列化后超过10KB的字典
                continue
            else:
                state_copy[key] = value
        return state_copy

    async def get_state(self) -> Dict[str, Any]:
        """获取当前状态（返回深拷贝以避免外部修改）- 现在是异步方法以确保一致性"""
        async with self._lock:
            return copy.deepcopy(self.current_state)

    def get_state_reference(self) -> Dict[str, Any]:
        """获取当前状态引用（用于只读场景，性能更好）- 同步方法保持不变"""
        return self.current_state.copy()

    async def get_history(self, limit: int = 10) -> List[StateSnapshot]:
        """获取状态历史 - 现在是异步方法以确保一致性"""
        async with self._lock:
            return self.history[-limit:]

    async def add_error(self, error: str) -> None:
        """添加错误记录（异步版本）"""
        async with self._lock:
            if 'errors' not in self.current_state:
                self.current_state['errors'] = []
            self.current_state['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'message': error
            })
            self.current_state['last_error'] = error

    def add_error_sync(self, error: str) -> None:
        """添加错误记录（同步版本，用于向后兼容）"""
        if 'errors' not in self.current_state:
            self.current_state['errors'] = []
        self.current_state['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'message': error
        })
        self.current_state['last_error'] = error

    async def add_extracted_data(self, data: List[Any]) -> None:
        """添加提取的数据（异步版本）"""
        async with self._lock:
            if 'extracted_data' not in self.current_state:
                self.current_state['extracted_data'] = []
            self.current_state['extracted_data'].extend(data)
            self.current_state['total_extracted'] = len(self.current_state['extracted_data'])

    def rollback(self, steps: int = 1) -> bool:
        """
        回滚到之前的状态 - 同步方法保持不变

        Args:
            steps: 回滚步数

        Returns:
            是否成功回滚
        """
        if len(self.history) < steps:
            return False

        snapshot = self.history[-steps]
        self.current_state = copy.deepcopy(snapshot.state)
        self._update_count = snapshot.metadata.get('update_count', 0)
        return True

    def get_memory_usage(self) -> Dict[str, int]:
        """获取内存使用情况的估算"""
        import sys

        def get_size(obj, seen=None):
            """Recursively find size of objects"""
            size = sys.getsizeof(obj)
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            # Important mark as seen *before* entering recursion to handle
            # self-referential objects
            seen.add(obj_id)
            if isinstance(obj, dict):
                size += sum([get_size(v, seen) for v in obj.values()])
                size += sum([get_size(k, seen) for k in obj.keys()])
            elif hasattr(obj, '__dict__'):
                size += get_size(obj.__dict__, seen)
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
                size += sum([get_size(i, seen) for i in obj])
            return size

        return {
            'current_state_size': get_size(self.current_state),
            'history_size': sum(get_size(snapshot) for snapshot in self.history),
            'total_estimated_size': get_size(self.current_state) + sum(get_size(snapshot) for snapshot in self.history)
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（兼容方法）- 同步方法保持不变"""
        return self.get_state()

    def get_update_count(self) -> int:
        """获取更新计数 - 同步方法保持不变"""
        return self._update_count

    async def sync_frontier(self, frontier: Any) -> None:
        """
        从 CrawlFrontier 同步爬取追踪数据到状态（异步版本，统一接口）。

        Args:
            frontier: CrawlFrontier 实例
        """
        async with self._lock:
            self.current_state['visited_urls'] = frontier.get_visited_urls()
            self.current_state['queue_size'] = frontier.queue_size()
            self.current_state['pages_crawled'] = frontier.pages_crawled()

    def sync_frontier_sync(self, frontier: Any) -> None:
        """从 CrawlFrontier 同步爬取追踪数据到状态（同步版本）。"""
        self.current_state['visited_urls'] = frontier.get_visited_urls()
        self.current_state['queue_size'] = frontier.queue_size()
        self.current_state['pages_crawled'] = frontier.pages_crawled()

    async def add_url_result(self, url: str, result: Dict[str, Any]) -> None:
        """
        记录单个 URL 的提取结果摘要。

        Args:
            url: 已处理的 URL
            result: 该 URL 的结果摘要（如 {'records_count': 5, 'success': True}）
        """
        async with self._lock:
            if 'per_url_results' not in self.current_state:
                self.current_state['per_url_results'] = {}
            self.current_state['per_url_results'][url] = result

    def add_url_result_sync(self, url: str, result: Dict[str, Any]) -> None:
        """记录单个 URL 的提取结果摘要（同步版本）。"""
        if 'per_url_results' not in self.current_state:
            self.current_state['per_url_results'] = {}
        self.current_state['per_url_results'][url] = result

    # ---- 监控系统委托方法 ----
    # StateManager 作为统一入口，委托给 MetricsCollector 和 ProgressTracker

    def _ensure_monitoring(self):
        """延迟初始化监控组件"""
        if not hasattr(self, '_metrics_collector'):
            from src.monitoring.metrics_collector import MetricsCollector
            from src.monitoring.progress_tracker import ProgressTracker
            self._metrics_collector = MetricsCollector()
            self._progress_tracker = ProgressTracker()

    async def update_progress(self, task_id, stage, progress, details=None, status=None):
        """委托给 ProgressTracker"""
        self._ensure_monitoring()
        await self._progress_tracker.update_progress(task_id, stage, progress, details, status)

    def get_task_progress(self, task_id):
        """委托给 ProgressTracker"""
        self._ensure_monitoring()
        return self._progress_tracker.get_task_progress(task_id)

    async def record_llm_call(self, provider, duration, success, tokens=None):
        """委托给 MetricsCollector"""
        self._ensure_monitoring()
        await self._metrics_collector.record_llm_call_async(provider, duration, success, tokens)

    async def record_page_load(self, url, duration, success):
        """委托给 MetricsCollector"""
        self._ensure_monitoring()
        await self._metrics_collector.record_page_load_async(url, duration, success)

    async def record_code_execution(self, duration, success, error=None):
        """委托给 MetricsCollector"""
        self._ensure_monitoring()
        await self._metrics_collector.record_code_execution_async(duration, success, error)

    def get_metrics_summary(self):
        """委托给 MetricsCollector"""
        self._ensure_monitoring()
        return self._metrics_collector.get_metrics_summary()