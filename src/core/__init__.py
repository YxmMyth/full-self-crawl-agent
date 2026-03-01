"""Core module - 核心组件"""
from .completion_gate import CompletionGate
from .smart_router import SmartRouter
from .state_manager import StateManager
from .risk_monitor import RiskMonitor
from .context_compressor import ContextCompressor
from .crawl_frontier import CrawlFrontier, CrawlItem, canonicalize_url

__all__ = [
    'CompletionGate',
    'SmartRouter',
    'StateManager',
    'RiskMonitor',
    'ContextCompressor',
    'CrawlFrontier',
    'CrawlItem',
    'canonicalize_url',
]
