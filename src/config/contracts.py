"""
契约定义系统 - 战略层契约规范
定义所有行为必须遵守的契约规范
"""

from typing import Dict, List, Optional, Any, Set, Literal
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json


# ==================== 任务目标契约 ====================

class ExtractionType(str, Enum):
    """数据提取类型"""
    SINGLE_PAGE = "single_page"  # 单页提取
    MULTI_PAGE = "multi_page"  # 多页提取
    PAGINATION = "pagination"  # 分页提取
    INFINITE_SCROLL = "infinite_scroll"  # 无限滚动
    FORM_SUBMISSION = "form_submission"  # 表单提交


class FieldType(str, Enum):
    """字段类型定义"""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    URL = "url"
    IMAGE = "image"
    HTML = "html"
    RAW = "raw"


@dataclass
class FieldSpec:
    """字段规范"""
    name: str  # 字段名
    type: FieldType  # 字段类型
    selector: str  # CSS/XPath 选择器
    required: bool = False  # 是否必填
    multiple: bool = False  # 是否多值
    description: Optional[str] = None  # 字段描述
    validation_rules: Optional[List[str]] = None  # 验证规则


@dataclass
class ExtractionTarget:
    """提取目标"""
    name: str  # 目标名称
    fields: List[FieldSpec]  # 字段列表
    url_pattern: Optional[str] = None  # URL 模式（用于多页）
    pagination: Optional[Dict[str, Any]] = None  # 分页配置


@dataclass
class SpecContract:
    """
    Spec 契约 - 定义任务的完整规范

    契约一旦加载即冻结，不可修改
    所有行为必须严格遵守契约定义
    """
    # 基本信息
    task_id: str  # 任务ID
    task_name: str  # 任务名称
    created_at: datetime  # 创建时间
    version: str = "1.0"  # 契约版本

    # 提取目标
    extraction_type: ExtractionType = ExtractionType.SINGLE_PAGE
    targets: List[ExtractionTarget] = field(default_factory=list)

    # 数据源
    start_url: str = ""  # 起始URL
    max_pages: int = 100  # 最大页数
    depth_limit: int = 3  # 爬取深度限制

    # 验证规则
    validation_rules: Dict[str, Any] = field(default_factory=dict)

    # 反爬策略
    anti_bot: Dict[str, Any] = field(default_factory=dict)

    # 完成标准
    completion_criteria: Dict[str, Any] = field(default_factory=dict)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 冻结状态
    _frozen: bool = False

    def __post_init__(self):
        """初始化后冻结契约"""
        self._frozen = True

    def __setattr__(self, key, value):
        """防止修改已冻结的契约"""
        if getattr(self, '_frozen', False) and key != '_frozen':
            raise PermissionError(f"SpecContract is frozen and cannot be modified")
        super().__setattr__(key, value)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpecContract':
        """从字典加载契约"""
        # 解析提取类型
        extraction_type = ExtractionType(data.get('extraction_type', 'single_page'))

        # 解析字段规范
        targets = []
        for target_data in data.get('targets', []):
            fields = []
            for field_data in target_data.get('fields', []):
                field = FieldSpec(
                    name=field_data['name'],
                    type=FieldType(field_data.get('type', 'text')),
                    selector=field_data['selector'],
                    required=field_data.get('required', False),
                    multiple=field_data.get('multiple', False),
                    description=field_data.get('description'),
                    validation_rules=field_data.get('validation_rules')
                )
                fields.append(field)

            target = ExtractionTarget(
                name=target_data['name'],
                fields=fields,
                url_pattern=target_data.get('url_pattern'),
                pagination=target_data.get('pagination')
            )
            targets.append(target)

        # 创建契约
        contract = cls(
            task_id=data['task_id'],
            task_name=data['task_name'],
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            version=data.get('version', '1.0'),
            extraction_type=extraction_type,
            targets=targets,
            start_url=data.get('start_url', ''),
            max_pages=data.get('max_pages', 100),
            depth_limit=data.get('depth_limit', 3),
            validation_rules=data.get('validation_rules', {}),
            anti_bot=data.get('anti_bot', {}),
            completion_criteria=data.get('completion_criteria', {}),
            metadata=data.get('metadata', {})
        )

        return contract

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'created_at': self.created_at.isoformat(),
            'version': self.version,
            'extraction_type': self.extraction_type.value,
            'targets': [
                {
                    'name': target.name,
                    'fields': [
                        {
                            'name': field.name,
                            'type': field.type.value,
                            'selector': field.selector,
                            'required': field.required,
                            'multiple': field.multiple,
                            'description': field.description,
                            'validation_rules': field.validation_rules
                        }
                        for field in target.fields
                    ],
                    'url_pattern': target.url_pattern,
                    'pagination': target.pagination
                }
                for target in self.targets
            ],
            'start_url': self.start_url,
            'max_pages': self.max_pages,
            'depth_limit': self.depth_limit,
            'validation_rules': self.validation_rules,
            'anti_bot': self.anti_bot,
            'completion_criteria': self.completion_criteria,
            'metadata': self.metadata
        }


# ==================== 状态契约 ====================

class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"  # 待处理
    RUNNING = "running"  # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    PAUSED = "paused"  # 已暂停
    CANCELLED = "cancelled"  # 已取消


@dataclass
class ExtractionProgress:
    """提取进度"""
    total_items: int = 0  # 总条目数
    successful_items: int = 0  # 成功条目数
    failed_items: int = 0  # 失败条目数
    current_page: int = 0  # 当前页码
    current_url: Optional[str] = None  # 当前URL


@dataclass
class ResourceUsage:
    """资源使用情况"""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    network_bytes: int = 0
    execution_time_seconds: float = 0.0


@dataclass
class StateContract:
    """
    State 契约 - 定义任务的运行时状态

    状态由系统管理，代理可以读取但不能直接修改
    状态变更必须通过规范的机制
    """
    # 基本信息
    task_id: str
    timestamp: datetime

    # 任务状态
    status: TaskStatus
    progress: ExtractionProgress = field(default_factory=ExtractionProgress)

    # 已提取数据
    extracted_data: List[Dict[str, Any]] = field(default_factory=list)

    # 错误和警告
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # 资源使用
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)

    # 证据引用
    evidence_refs: List[str] = field(default_factory=list)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_initial(cls, task_id: str, spec: SpecContract) -> 'StateContract':
        """创建初始状态"""
        return cls(
            task_id=task_id,
            timestamp=datetime.now(),
            status=TaskStatus.PENDING,
            progress=ExtractionProgress(),
            extracted_data=[],
            errors=[],
            warnings=[],
            resource_usage=ResourceUsage(),
            evidence_refs=[],
            metadata={'spec_version': spec.version}
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'progress': {
                'total_items': self.progress.total_items,
                'successful_items': self.progress.successful_items,
                'failed_items': self.progress.failed_items,
                'current_page': self.progress.current_page,
                'current_url': self.progress.current_url
            },
            'extracted_data_count': len(self.extracted_data),
            'errors_count': len(self.errors),
            'warnings_count': len(self.warnings),
            'resource_usage': {
                'cpu_percent': self.resource_usage.cpu_percent,
                'memory_mb': self.resource_usage.memory_mb,
                'network_bytes': self.resource_usage.network_bytes,
                'execution_time_seconds': self.resource_usage.execution_time_seconds
            },
            'evidence_refs_count': len(self.evidence_refs),
            'metadata': self.metadata
        }
