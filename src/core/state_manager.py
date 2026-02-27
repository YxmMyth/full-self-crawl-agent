"""
全局状态管理器 - 管理层
管理任务的全局状态
"""

from typing import Dict, Any, List, Optional, Tuple
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
                'update_count': 0
            }

            self.current_state = initial_state
            return initial_state

    def create_initial_state_sync(self, task_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """创建初始状态（同步版本，用于向后兼容）"""
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
            'update_count': 0
        }

        self.current_state = initial_state
        return initial_state

    async def update_state(self, updates: Dict[str, Any], reason: str = "") -> None:
        """
        更新状态（异步版本，线程安全）

        Args:
            updates: 状态更新字典
            reason: 更新原因
        """
        async with self._lock:
            # 创建快照
            self._create_snapshot(reason)

            # 深拷贝更新以避免外部修改
            updates_copy = copy.deepcopy(updates)

            # 更新状态
            self.current_state.update(updates_copy)

            # 更新时间戳和计数器
            self.current_state['timestamp'] = datetime.now().isoformat()
            self.current_state['updated_at'] = datetime.now().isoformat()
            self._update_count += 1
            self.current_state['update_count'] = self._update_count

    def update_state_sync(self, updates: Dict[str, Any], reason: str = "") -> None:
        """
        更新状态（同步版本，用于向后兼容）

        注意：此方法不提供线程安全保证，在异步环境中请使用 update_state

        Args:
            updates: 状态更新字典
            reason: 更新原因
        """
        # 创建快照
        self._create_snapshot(reason)

        # 更新状态
        self.current_state.update(updates)

        # 更新时间戳
        self.current_state['timestamp'] = datetime.now().isoformat()
        self.current_state['updated_at'] = datetime.now().isoformat()
        self._update_count += 1

    def _create_snapshot(self, reason: str = "") -> None:
        """创建状态快照"""
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            state=copy.deepcopy(self.current_state),  # 深拷贝避免引用问题
            metadata={'reason': reason, 'update_count': self._update_count}
        )
        self.history.append(snapshot)

        # 限制历史记录数量
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_state(self) -> Dict[str, Any]:
        """获取当前状态（返回深拷贝以避免外部修改）"""
        return copy.deepcopy(self.current_state)

    def get_state_reference(self) -> Dict[str, Any]:
        """获取当前状态引用（用于只读场景，性能更好）"""
        return self.current_state.copy()

    def get_history(self, limit: int = 10) -> List[StateSnapshot]:
        """获取状态历史"""
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
        """添加错误记录（同步版本）"""
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

    def add_extracted_data_sync(self, data: List[Any]) -> None:
        """添加提取的数据（同步版本）"""
        if 'extracted_data' not in self.current_state:
            self.current_state['extracted_data'] = []
        self.current_state['extracted_data'].extend(data)
        self.current_state['total_extracted'] = len(self.current_state['extracted_data'])

    def rollback(self, steps: int = 1) -> bool:
        """
        回滚到之前的状态

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

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（兼容方法）"""
        return self.get_state()

    def get_update_count(self) -> int:
        """获取更新计数"""
        return self._update_count

