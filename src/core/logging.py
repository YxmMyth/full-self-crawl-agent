"""
日志配置 - 统一日志管理
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """
    设置日志配置

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        log_file: 日志文件路径（可选）
        log_format: 自定义日志格式
        include_timestamp: 是否包含时间戳

    Returns:
        配置好的根日志器
    """
    # 获取日志级别
    log_level = getattr(logging, level.upper(), logging.INFO)

    # 默认格式
    if log_format is None:
        if include_timestamp:
            log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        else:
            log_format = "[%(levelname)s] %(name)s: %(message)s"

    # 创建格式器
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除现有处理器
    root_logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    获取日志器

    Args:
        name: 日志器名称

    Returns:
        Logger 实例
    """
    return logging.getLogger(name)


class LogContext:
    """
    日志上下文管理器

    用于临时修改日志级别或添加上下文信息
    """

    def __init__(self, logger: logging.Logger, level: Optional[str] = None,
                 extra: Optional[dict] = None):
        self.logger = logger
        self.new_level = getattr(logging, level.upper(), None) if level else None
        self.old_level = None
        self.extra = extra or {}

    def __enter__(self):
        if self.new_level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.new_level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_level is not None:
            self.logger.setLevel(self.old_level)


class TaskLogger:
    """
    任务日志器

    为特定任务提供带前缀的日志记录
    """

    def __init__(self, task_id: str, base_logger: Optional[logging.Logger] = None):
        self.task_id = task_id
        self.logger = base_logger or get_logger('task')

    def _format(self, msg: str) -> str:
        return f"[{self.task_id}] {msg}"

    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(self._format(msg), *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(self._format(msg), *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(self._format(msg), *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(self._format(msg), *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        self.logger.exception(self._format(msg), *args, **kwargs)


# 常用日志器
agent_logger = get_logger('agent')
browser_logger = get_logger('browser')
llm_logger = get_logger('llm')
executor_logger = get_logger('executor')
state_logger = get_logger('state')