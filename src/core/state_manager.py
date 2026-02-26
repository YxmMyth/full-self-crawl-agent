"""
全局状态管理器 - 管理层
管理任务的全局状态
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


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
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self.current_state: Dict[str, Any] = initial_state or {}
        self.history: List[StateSnapshot] = []
        self.max_history: int = 100  # 最大历史记录数

    def create_initial_state(self, task_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """创建初始状态"""
        initial_state = {
            'task_id': task_id,
            'spec': spec,
            'stage': 'initialized',
            'iteration': 0,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'performance_data': {},
            'failure_history': [],
            'evidence_collected': {}
        }

        self.current_state = initial_state
        return initial_state

    def update_state(self, updates: Dict[str, Any], reason: str = "") -> None:
        """
        更新状态

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

    def _create_snapshot(self, reason: str = "") -> None:
        """创建状态快照"""
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            state=self.current_state.copy(),
            metadata={'reason': reason}
        )
        self.history.append(snapshot)

        # 限制历史记录数量
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return self.current_state.copy()

    def get_history(self, limit: int = 10) -> List[StateSnapshot]:
        """获取状态历史"""
        return self.history[-limit:]

    def add_error(self, error: str) -> None:
        """添加错误记录"""
        if 'errors' not in self.current_state:
            self.current_state['errors'] = []
        self.current_state['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'message': error
        })
        self.current_state['last_error'] = error

    def add_extracted_data(self, data: List[Any]) -> None:
        """添加提取的数据"""
        if 'extracted_data' not in self.current_state:
            self.current_state['extracted_data'] = []
        self.current_state['extracted_data'].extend(data)
        self.current_state['total_extracted'] = len(self.current_state['extracted_data'])

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（兼容方法）"""
        return self.get_state()

