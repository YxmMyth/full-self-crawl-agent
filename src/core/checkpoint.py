"""
Checkpoint — 爬取状态持久化与恢复

解决长时间爬取中的断点续爬问题。

功能：
- 定期保存爬取状态（frontier、已访问 URL、已提取数据）到 JSON
- 从 checkpoint 恢复爬取状态，继续未完成的爬取
- 自动清理过期 checkpoint
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = Path('states/checkpoints')
CHECKPOINT_INTERVAL = 10  # 每处理 N 页保存一次


class CrawlCheckpoint:
    """爬取 checkpoint 管理器"""

    def __init__(self, task_id: str, checkpoint_dir: Optional[Path] = None,
                 interval: int = CHECKPOINT_INTERVAL):
        self.task_id = task_id
        self.checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self._pages_since_save = 0

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"{self.task_id}_checkpoint.json"

    def should_save(self) -> bool:
        """是否应该保存 checkpoint（每 N 页触发一次）"""
        self._pages_since_save += 1
        return self._pages_since_save >= self.interval

    def save(self, state: Dict[str, Any]) -> None:
        """保存 checkpoint 到磁盘"""
        checkpoint = {
            'task_id': self.task_id,
            'timestamp': time.time(),
            'visited_urls': list(state.get('visited_urls', [])),
            'all_results': state.get('all_results', []),
            'frontier_queued': list(state.get('frontier_queued', [])),
            'frontier_visited': list(state.get('frontier_visited', [])),
            'reflect_hints': state.get('reflect_hints', {}),
            'pages_completed': state.get('pages_completed', 0),
            'total_records': state.get('total_records', 0),
        }

        # 原子写入：先写临时文件再重命名
        tmp_path = self.checkpoint_path.with_suffix('.tmp')
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2, default=str)
            tmp_path.replace(self.checkpoint_path)
            self._pages_since_save = 0
            logger.info(f"[Checkpoint] 已保存: {len(checkpoint['visited_urls'])} URLs, "
                       f"{checkpoint['total_records']} 条记录")
        except Exception as e:
            logger.error(f"[Checkpoint] 保存失败: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

    def load(self) -> Optional[Dict[str, Any]]:
        """从磁盘加载 checkpoint"""
        if not self.checkpoint_path.exists():
            return None

        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"[Checkpoint] 已加载: {len(data.get('visited_urls', []))} URLs, "
                       f"{data.get('total_records', 0)} 条记录")
            return data
        except Exception as e:
            logger.error(f"[Checkpoint] 加载失败: {e}")
            return None

    def exists(self) -> bool:
        """是否存在可恢复的 checkpoint"""
        return self.checkpoint_path.exists()

    def cleanup(self) -> None:
        """清理 checkpoint 文件（爬取完成后调用）"""
        try:
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
                logger.info(f"[Checkpoint] 已清理: {self.checkpoint_path}")
        except Exception as e:
            logger.warning(f"[Checkpoint] 清理失败: {e}")
