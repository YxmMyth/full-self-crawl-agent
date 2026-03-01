"""
CrawlFrontier - 爬取边界管理
实现优先队列 + 已访问集合 + 过滤器的完整爬取边界管理。

功能：
- 优先级队列（heapq）管理待抓取 URL
- 已访问集合（visited set）去重
- 静态资源过滤
- URL 规范化（移除 fragment、规范化末尾斜杠）
- 域名/路径过滤（url_patterns）
- 深度限制
"""

from __future__ import annotations

import heapq
import re
from typing import Any, Dict, Iterator, List, Optional, Set
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode


# 静态资源扩展名过滤列表
_STATIC_EXTENSIONS = {
    '.css', '.js', '.ico', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp',
    '.bmp', '.tiff', '.woff', '.woff2', '.ttf', '.eot', '.otf',
    '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.zip', '.tar', '.gz', '.rar', '.7z',
}


def canonicalize_url(url: str) -> str:
    """
    规范化 URL：
    - 移除 fragment（#...）
    - 规范化路径（末尾斜杠统一保留，避免重复）
    - 保留查询字符串（按键名排序以去重等价 URL）
    """
    parsed = urlparse(url)
    # 移除 fragment
    # 对查询字符串排序，避免参数顺序不同导致重复
    query = parsed.query
    if query:
        params = parse_qs(query, keep_blank_values=True)
        sorted_query = urlencode(sorted(params.items()), doseq=True)
    else:
        sorted_query = ''

    normalized = urlunparse((
        parsed.scheme,
        parsed.netloc.lower(),
        parsed.path,
        parsed.params,
        sorted_query,
        '',  # 移除 fragment
    ))
    return normalized


def _is_static_resource(url: str) -> bool:
    """判断 URL 是否指向静态资源（不需要爬取的文件）"""
    parsed = urlparse(url)
    path = parsed.path.lower()
    # 获取文件扩展名
    dot_pos = path.rfind('.')
    if dot_pos != -1:
        ext = path[dot_pos:]
        if ext in _STATIC_EXTENSIONS:
            return True
    return False


class CrawlItem:
    """
    爬取队列中的一个条目

    支持 heapq 比较（按优先级升序，数值越小优先级越高）。
    """
    __slots__ = ('priority', 'url', 'depth', 'metadata')

    def __init__(
        self,
        url: str,
        depth: int = 0,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.priority = priority
        self.url = url
        self.depth = depth
        self.metadata = metadata or {}

    def __lt__(self, other: 'CrawlItem') -> bool:
        return self.priority < other.priority

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CrawlItem):
            return False
        return self.url == other.url

    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'depth': self.depth,
            'priority': self.priority,
            'metadata': self.metadata,
        }


class CrawlFrontier:
    """
    爬取边界管理器

    职责：
    - 管理待抓取 URL 的优先队列
    - 维护已访问 URL 集合（去重）
    - 过滤静态资源和非目标域 URL
    - 支持 url_patterns 白名单过滤
    - 记录爬取统计数据

    用法::

        frontier = CrawlFrontier(base_url='https://example.com', max_depth=3)
        frontier.push('https://example.com/page/1')

        while not frontier.is_empty():
            item = frontier.pop()
            # 处理 item.url ...
            frontier.mark_visited(item.url)
    """

    def __init__(
        self,
        base_url: str = '',
        max_depth: int = 3,
        max_pages: int = 100,
        url_patterns: Optional[List[str]] = None,
        same_domain_only: bool = True,
    ):
        """
        Args:
            base_url: 起始 URL，用于提取基准域名
            max_depth: 最大爬取深度（0 表示只爬 base_url）
            max_pages: 最大爬取页面数量
            url_patterns: URL 路径白名单正则列表（可选）；非空时 URL 须匹配至少一个
            same_domain_only: 是否只爬取同一域名的 URL（默认 True）
        """
        self._heap: List[CrawlItem] = []
        self._visited: Set[str] = set()
        self._queued: Set[str] = set()  # 已入队（避免重复入队）
        self._pages_crawled: int = 0

        self.base_url = base_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.url_patterns = url_patterns or []
        self.same_domain_only = same_domain_only

        # 提取基准域名
        if base_url:
            parsed = urlparse(base_url)
            self._base_domain = parsed.netloc.lower()
            self._base_scheme = parsed.scheme
        else:
            self._base_domain = ''
            self._base_scheme = 'https'

        # 统计
        self._total_pushed: int = 0
        self._total_filtered: int = 0

    # ------------------------------------------------------------------
    # 入队
    # ------------------------------------------------------------------

    def push(
        self,
        url: str,
        depth: int = 0,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        将 URL 加入队列。

        Args:
            url: 要爬取的 URL
            depth: 当前深度
            priority: 优先级（数值越小越优先）
            metadata: 附加元数据（如来源页面、链接文本等）

        Returns:
            True 表示成功入队，False 表示被过滤或已存在
        """
        # 规范化
        url = canonicalize_url(url)

        # 已访问或已入队
        if url in self._visited or url in self._queued:
            return False

        # 深度限制
        if depth > self.max_depth:
            self._total_filtered += 1
            return False

        # 页面总数限制
        if self._total_pushed >= self.max_pages:
            self._total_filtered += 1
            return False

        # 过滤静态资源
        if _is_static_resource(url):
            self._total_filtered += 1
            return False

        # 域名过滤
        if self.same_domain_only and self._base_domain:
            parsed = urlparse(url)
            if parsed.netloc.lower() != self._base_domain:
                self._total_filtered += 1
                return False

        # URL 模式过滤（白名单）
        if self.url_patterns:
            parsed_path = urlparse(url).path
            if not any(re.search(p, parsed_path) for p in self.url_patterns):
                self._total_filtered += 1
                return False

        item = CrawlItem(url=url, depth=depth, priority=priority, metadata=metadata)
        heapq.heappush(self._heap, item)
        self._queued.add(url)
        self._total_pushed += 1
        return True

    def push_many(
        self,
        urls: List[str],
        depth: int = 0,
        priority: int = 0,
        base_url: str = '',
    ) -> int:
        """
        批量加入 URL（相对 URL 会被解析为绝对 URL）。

        Args:
            urls: URL 列表（可包含相对路径）
            depth: 当前深度
            priority: 优先级
            base_url: 用于解析相对 URL 的基准 URL

        Returns:
            实际入队的 URL 数量
        """
        resolve_base = base_url or self.base_url
        pushed = 0
        for url in urls:
            if resolve_base:
                url = urljoin(resolve_base, url)
            if self.push(url, depth=depth, priority=priority):
                pushed += 1
        return pushed

    # ------------------------------------------------------------------
    # 出队
    # ------------------------------------------------------------------

    def pop(self) -> Optional[CrawlItem]:
        """
        取出优先级最高的 URL。

        Returns:
            CrawlItem，若队列为空则返回 None
        """
        while self._heap:
            item = heapq.heappop(self._heap)
            self._queued.discard(item.url)
            # 若已访问（可能在出队前被 mark_visited），跳过
            if item.url not in self._visited:
                return item
        return None

    # ------------------------------------------------------------------
    # 已访问管理
    # ------------------------------------------------------------------

    def mark_visited(self, url: str) -> None:
        """标记 URL 为已访问"""
        url = canonicalize_url(url)
        self._visited.add(url)
        self._queued.discard(url)
        self._pages_crawled += 1

    def is_visited(self, url: str) -> bool:
        """判断 URL 是否已访问"""
        return canonicalize_url(url) in self._visited

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        """队列是否为空"""
        return len(self._heap) == 0

    def queue_size(self) -> int:
        """当前队列大小"""
        return len(self._heap)

    def visited_count(self) -> int:
        """已访问 URL 数量"""
        return len(self._visited)

    def pages_crawled(self) -> int:
        """已爬取页面数量"""
        return self._pages_crawled

    def get_visited_urls(self) -> List[str]:
        """返回已访问 URL 列表（排序）"""
        return sorted(self._visited)

    def get_stats(self) -> Dict[str, Any]:
        """返回爬取统计信息"""
        return {
            'queue_size': self.queue_size(),
            'visited_count': self.visited_count(),
            'pages_crawled': self._pages_crawled,
            'total_pushed': self._total_pushed,
            'total_filtered': self._total_filtered,
            'max_depth': self.max_depth,
            'max_pages': self.max_pages,
        }

    def reset(self) -> None:
        """重置边界（清空队列和已访问集合）"""
        self._heap.clear()
        self._visited.clear()
        self._queued.clear()
        self._pages_crawled = 0
        self._total_pushed = 0
        self._total_filtered = 0

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[CrawlItem]:
        """迭代器：依次 pop 直到队列为空"""
        while not self.is_empty():
            item = self.pop()
            if item:
                yield item
