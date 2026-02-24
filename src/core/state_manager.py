"""
全局状态管理器 - 管理层
管理任务的全局状态
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from .contracts import StateContract, TaskStatus


@dataclass
class StateSnapshot:
    """状态快照"""
    timestamp: datetime
    state: StateContract
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

    def __init__(self, initial_state: Optional[StateContract] = None):
        self.current_state: StateContract = initial_state or self._create_empty_state()
        self.history: List[StateSnapshot] = []
        self.max_history: int = 100  # 最大历史记录数

    def _create_empty_state(self) -> StateContract:
        """创建空状态"""
        from .contracts import SpecContract
        empty_spec = SpecContract(
            task_id="empty",
            task_name="Empty Task",
            created_at=datetime.now()
        )
        return StateContract.create_initial("empty", empty_spec)

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
        for key, value in updates.items():
            if hasattr(self.current_state, key):
                setattr(self.current_state, key, value)
            else:
                # 如果是元数据字段
                if key in self.current_state.metadata:
                    self.current_state.metadata[key] = value
                else:
                    # 添加到元数据
                    self.current_state.metadata[key] = value

        # 更新时间戳
        self.current_state.timestamp = datetime.now()

    def _create_snapshot(self, reason: str = "") -> None:
        """创建状态快照"""
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            state=self._clone_state(self.current_state),
            metadata={'reason': reason}
        )
        self.history.append(snapshot)

        # 限制历史记录数量
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _clone_state(self, state: StateContract) -> StateContract:
        """克隆状态"""
        import copy
        return copy.deepcopy(state)

    def rollback_to(self, timestamp: datetime) -> bool:
        """
        回滚到指定时间点

        Args:
            timestamp: 目标时间点

        Returns:
            是否成功
        """
        # 查找最近的快照
        for snapshot in reversed(self.history):
            if snapshot.timestamp <= timestamp:
                self.current_state = self._clone_state(snapshot.state)
                return True

        return False

    def get_state(self) -> StateContract:
        """获取当前状态"""
        return self.current_state

    def get_history(self, limit: int = 10) -> List[StateSnapshot]:
        """获取状态历史"""
        return self.history[-limit:]

    def get_status(self) -> TaskStatus:
        """获取当前状态"""
        return self.current_state.status

    def set_status(self, status: TaskStatus, reason: str = "") -> None:
        """设置任务状态"""
        self.update_state({'status': status}, reason=reason)

    def add_error(self, error: str) -> None:
        """添加错误"""
        self.current_state.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """添加警告"""
        self.current_state.warnings.append(warning)

    def update_progress(self, **kwargs) -> None:
        """更新进度"""
        for key, value in kwargs.items():
            if hasattr(self.current_state.progress, key):
                setattr(self.current_state.progress, key, value)

    def add_extracted_data(self, data: Dict[str, Any]) -> None:
        """添加提取的数据"""
        self.current_state.extracted_data.append(data)
        self.current_state.progress.successful_items += 1
        self.current_state.progress.total_items += 1

    def update_resource_usage(self, **kwargs) -> None:
        """更新资源使用"""
        for key, value in kwargs.items():
            if hasattr(self.current_state.resource_usage, key):
                setattr(self.current_state.resource_usage, key, value)

    def add_evidence_ref(self, ref: str) -> None:
        """添加证据引用"""
        self.current_state.evidence_refs.append(ref)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'current_state': self.current_state.to_dict(),
            'history_count': len(self.history)
        }

    def persist(self, filepath: str) -> None:
        """持久化状态"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'StateManager':
        """从文件加载状态"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # TODO: 完整实现加载逻辑
            return cls()
