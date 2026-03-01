"""
进度追踪器 - 跟踪任务进度和阶段完成情况供Orchestrator使用
"""
import asyncio
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
import threading
from dataclasses import dataclass


class TaskStage(Enum):
    """任务阶段定义"""
    SENSE = "sense"          # 感知
    PLAN = "plan"            # 规划
    ACT = "act"              # 执行
    VERIFY = "verify"        # 验证
    GATE = "gate"            # 门禁检查
    JUDGE = "judge"          # 决策
    REFLECT = "reflect"      # 反思
    FINISH = "finish"        # 完成


class ProgressStatus(Enum):
    """进度状态定义"""
    PENDING = "pending"              # 等待处理
    INITIALIZING = "initializing"    # 初始化中
    RUNNING = "running"              # 运行中
    PAUSED = "paused"               # 暂停
    VERIFYING = "verifying"         # 验证中
    REFLECTING = "reflecting"       # 反思中
    COMPLETING = "completing"       # 完成中
    COMPLETED = "completed"         # 已完成
    FAILED = "failed"               # 失败
    CANCELLED = "cancelled"         # 已取消


@dataclass
class StageProgress:
    """阶段进度"""
    stage: TaskStage
    progress: float  # 0.0 to 1.0
    status: ProgressStatus
    details: Dict[str, Any]
    updated_at: datetime
    start_time: Optional[datetime] = None


@dataclass
class TaskProgress:
    """任务进度"""
    task_id: str
    overall_progress: float  # 0.0 to 1.0
    current_stage: TaskStage
    status: ProgressStatus
    stages: Dict[str, StageProgress]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


class ProgressTracker:
    """进度追踪器 - 跟踪任务进度和阶段完成情况"""

    def __init__(self):
        self._lock = threading.RLock()
        self.task_progresses: Dict[str, TaskProgress] = {}

    async def update_progress(self, task_id: str, stage: TaskStage, progress: float,
                            details: Dict[str, Any] = None, status: ProgressStatus = None):
        """更新特定任务的进度"""
        if details is None:
            details = {}

        async with asyncio.Lock():  # 使用async lock确保线程安全
            with self._lock:
                if task_id not in self.task_progresses:
                    # 创建新的任务进度记录
                    self.task_progresses[task_id] = TaskProgress(
                        task_id=task_id,
                        overall_progress=0.0,
                        current_stage=stage,
                        status=status or ProgressStatus.RUNNING,
                        stages={},
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        metadata={}
                    )

                task_progress = self.task_progresses[task_id]

                # 更新阶段进度
                stage_key = stage.value
                if stage_key not in task_progress.stages:
                    task_progress.stages[stage_key] = StageProgress(
                        stage=stage,
                        progress=progress,
                        status=status or ProgressStatus.RUNNING,
                        details=details,
                        updated_at=datetime.now(),
                        start_time=datetime.now()
                    )
                else:
                    task_progress.stages[stage_key].progress = progress
                    task_progress.stages[stage_key].status = status or ProgressStatus.RUNNING
                    task_progress.stages[stage_key].details = details
                    task_progress.stages[stage_key].updated_at = datetime.now()

                # 更新任务状态
                task_progress.current_stage = stage
                task_progress.status = status or ProgressStatus.RUNNING
                task_progress.updated_at = datetime.now()

                # 计算总体进度
                task_progress.overall_progress = self._calculate_overall_progress(task_progress.stages)

    def _calculate_overall_progress(self, stages: Dict[str, StageProgress]) -> float:
        """计算总体进度"""
        if not stages:
            return 0.0

        # 根据各阶段权重计算总进度
        stage_weights = {
            'sense': 0.15,
            'plan': 0.20,
            'act': 0.25,
            'verify': 0.20,
            'gate': 0.05,
            'judge': 0.05,
            'reflect': 0.10
        }

        total_weighted_progress = 0.0
        total_weight = 0.0

        for stage_key, stage_progress in stages.items():
            weight = stage_weights.get(stage_key, 0.0)
            total_weighted_progress += stage_progress.progress * weight
            total_weight += weight

        return total_weighted_progress / total_weight if total_weight > 0 else 0.0

    def get_task_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务进度详情，供Orchestrator查询"""
        with self._lock:
            if task_id not in self.task_progresses:
                return None

            task_progress = self.task_progresses[task_id]

            # 转换为字典格式
            stages_dict = {}
            for stage_key, stage_progress in task_progress.stages.items():
                stages_dict[stage_key] = {
                    'stage': stage_progress.stage.value,
                    'progress': stage_progress.progress,
                    'status': stage_progress.status.value,
                    'details': stage_progress.details,
                    'updated_at': stage_progress.updated_at.isoformat(),
                    'start_time': stage_progress.start_time.isoformat() if stage_progress.start_time else None
                }

            return {
                'task_id': task_progress.task_id,
                'overall_progress': task_progress.overall_progress,
                'current_stage': task_progress.current_stage.value,
                'status': task_progress.status.value,
                'stages': stages_dict,
                'created_at': task_progress.created_at.isoformat(),
                'updated_at': task_progress.updated_at.isoformat(),
                'metadata': task_progress.metadata
            }

    def get_all_progress(self) -> Dict[str, Dict[str, Any]]:
        """获取所有任务进度，供Orchestrator批量查询"""
        with self._lock:
            all_progress = {}
            for task_id in self.task_progresses:
                all_progress[task_id] = self.get_task_progress(task_id)
            return all_progress

    async def update_task_status(self, task_id: str, status: ProgressStatus):
        """更新任务状态"""
        async with asyncio.Lock():
            with self._lock:
                if task_id in self.task_progresses:
                    self.task_progresses[task_id].status = status
                    self.task_progresses[task_id].updated_at = datetime.now()

    async def add_metadata(self, task_id: str, key: str, value: Any):
        """添加元数据到任务"""
        async with asyncio.Lock():
            with self._lock:
                if task_id in self.task_progresses:
                    self.task_progresses[task_id].metadata[key] = value
                    self.task_progresses[task_id].updated_at = datetime.now()

    def task_exists(self, task_id: str) -> bool:
        """检查任务是否存在"""
        with self._lock:
            return task_id in self.task_progresses