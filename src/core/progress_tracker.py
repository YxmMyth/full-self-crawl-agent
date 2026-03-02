"""
CrawlProgressTracker — 结构化爬取进度跟踪

提供实时、结构化的爬取进度报告，解决"不知道进行到哪一步了"的问题。

功能：
- 记录每页处理事件（开始、完成、失败）
- 定时输出进度摘要（页数、记录数、成功率、预估剩余时间）
- JSON 格式事件日志，便于自动化解析
- 支持回调函数（可扩展到 WebSocket / 文件写入）
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PageEvent:
    """单页处理事件"""
    url: str
    event: str  # page_started | page_completed | page_failed
    timestamp: float = field(default_factory=time.time)
    records_extracted: int = 0
    quality_score: float = 0.0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'url': self.url,
            'event': self.event,
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'records_extracted': self.records_extracted,
            'quality_score': self.quality_score,
            'duration_seconds': round(self.duration_seconds, 2),
            'depth': self.depth,
        }
        if self.error:
            d['error'] = self.error
        return d


class CrawlProgressTracker:
    """
    爬取进度跟踪器

    使用方式:
        tracker = CrawlProgressTracker(total_target=200)
        tracker.page_started(url, depth=1)
        tracker.page_completed(url, records=5, quality=0.85)
        tracker.print_progress()
    """

    def __init__(
        self,
        total_target: int = 100,
        progress_interval: int = 5,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Args:
            total_target: 目标总页面数
            progress_interval: 每处理 N 页输出一次进度摘要
            on_event: 可选回调，每个事件触发时调用
        """
        self.total_target = total_target
        self.progress_interval = progress_interval
        self.on_event = on_event

        self._events: List[PageEvent] = []
        self._start_time = time.time()
        self._pages_completed = 0
        self._pages_failed = 0
        self._total_records = 0
        self._current_url: Optional[str] = None
        self._page_start_times: Dict[str, float] = {}

    def page_started(self, url: str, depth: int = 0):
        """记录页面开始处理"""
        self._current_url = url
        self._page_start_times[url] = time.time()
        event = PageEvent(url=url, event='page_started', depth=depth)
        self._emit(event)

    def page_completed(self, url: str, records: int = 0, quality: float = 0.0):
        """记录页面处理完成"""
        duration = time.time() - self._page_start_times.pop(url, time.time())
        self._pages_completed += 1
        self._total_records += records
        event = PageEvent(
            url=url, event='page_completed',
            records_extracted=records, quality_score=quality,
            duration_seconds=duration,
        )
        self._emit(event)

        if self._pages_completed % self.progress_interval == 0:
            self.print_progress()

    def page_failed(self, url: str, error: str = ''):
        """记录页面处理失败"""
        duration = time.time() - self._page_start_times.pop(url, time.time())
        self._pages_failed += 1
        event = PageEvent(
            url=url, event='page_failed',
            duration_seconds=duration, error=error,
        )
        self._emit(event)

    def _emit(self, event: PageEvent):
        """发送事件"""
        self._events.append(event)
        event_dict = event.to_dict()

        # 结构化日志输出
        logger.info(f"[Progress] {json.dumps(event_dict, ensure_ascii=False)}")

        # 回调
        if self.on_event:
            try:
                self.on_event(event_dict)
            except Exception:
                pass

    def get_progress(self) -> Dict[str, Any]:
        """获取当前进度摘要"""
        elapsed = time.time() - self._start_time
        total_processed = self._pages_completed + self._pages_failed
        success_rate = (self._pages_completed / total_processed * 100) if total_processed > 0 else 0
        avg_time = elapsed / total_processed if total_processed > 0 else 0
        remaining = self.total_target - total_processed
        eta_seconds = avg_time * remaining if avg_time > 0 else 0

        return {
            'pages_completed': self._pages_completed,
            'pages_failed': self._pages_failed,
            'total_processed': total_processed,
            'total_target': self.total_target,
            'total_records': self._total_records,
            'success_rate': round(success_rate, 1),
            'elapsed_seconds': round(elapsed, 1),
            'avg_seconds_per_page': round(avg_time, 1),
            'eta_seconds': round(eta_seconds, 0),
            'current_url': self._current_url,
        }

    def print_progress(self):
        """输出人类可读的进度摘要"""
        p = self.get_progress()
        eta_min = p['eta_seconds'] / 60
        elapsed_min = p['elapsed_seconds'] / 60

        summary = (
            f"📊 进度: {p['total_processed']}/{p['total_target']} 页 "
            f"({p['success_rate']}% 成功) | "
            f"记录: {p['total_records']} | "
            f"耗时: {elapsed_min:.1f}min | "
            f"预计剩余: {eta_min:.1f}min | "
            f"均速: {p['avg_seconds_per_page']:.1f}s/页"
        )
        print(summary)
        logger.info(f"[Progress] {json.dumps(p, ensure_ascii=False)}")

    def get_summary(self) -> Dict[str, Any]:
        """获取最终摘要（爬取结束后调用）"""
        progress = self.get_progress()
        progress['events_count'] = len(self._events)
        progress['status'] = 'completed'
        return progress
