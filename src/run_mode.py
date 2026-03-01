"""
运行模式检测

LOCAL:     python -m src.main specs/xxx.json
CONTAINER: Orchestrator 启动的 Docker 容器（环境变量 TASK_SPEC 存在）
"""

import os
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger('run_mode')


class RunMode(Enum):
    LOCAL = "local"
    CONTAINER = "container"


@dataclass
class RunContext:
    """运行上下文 — 所有环境差异收敛在这里"""
    mode: RunMode
    task_id: str = ""
    spec: Dict[str, Any] = field(default_factory=dict)
    spec_path: Optional[str] = None
    redis_url: Optional[str] = None

    @property
    def is_container(self) -> bool:
        return self.mode == RunMode.CONTAINER

    @property
    def is_local(self) -> bool:
        return self.mode == RunMode.LOCAL

    @property
    def sandbox_strict(self) -> bool:
        """本地开发严格校验，容器内不限"""
        return self.is_local


def detect_run_mode() -> RunMode:
    """有 TASK_SPEC 环境变量 → 容器模式，否则本地"""
    if os.environ.get("TASK_SPEC"):
        return RunMode.CONTAINER
    return RunMode.LOCAL


def build_run_context(spec_path: Optional[str] = None) -> RunContext:
    """构建运行上下文"""
    mode = detect_run_mode()

    if mode == RunMode.CONTAINER:
        spec = json.loads(os.environ["TASK_SPEC"])
        return RunContext(
            mode=mode,
            task_id=os.environ.get("TASK_ID", spec.get("task_id", "unknown")),
            spec=spec,
            redis_url=os.environ.get("RESULT_REDIS_URL"),
        )
    else:
        return RunContext(
            mode=mode,
            spec_path=spec_path,
        )
