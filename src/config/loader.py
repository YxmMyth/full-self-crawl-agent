"""
契约加载器 - 战略层
负责加载并验证 Spec 契约
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union
from .contracts import SpecContract, StateContract


class SpecLoader:
    """
    Spec 契约加载器

    职责：
    - 加载 Spec 契约文件（JSON/YAML）
    - 验证契约完整性
    - 返回冻结的 SpecContract 对象

    特性：
    - 契约一旦加载即冻结，不可修改
    - 支持版本控制
    - 严格的验证机制
    """

    def __init__(self, spec_dir: Union[str, Path]):
        self.spec_dir = Path(spec_dir)

    def load_spec(self, spec_path: Union[str, Path]) -> SpecContract:
        """
        加载 Spec 契约

        Args:
            spec_path: 契约文件路径

        Returns:
            冻结的 SpecContract 对象

        Raises:
            FileNotFoundError: 契约文件不存在
            ValueError: 契约验证失败
        """
        path = Path(spec_path)

        # 加载契约内容
        spec_data = self._load_file(path)

        # 验证契约
        self._validate_spec(spec_data)

        # 创建并返回冻结的契约对象
        return SpecContract.from_dict(spec_data)

    def load_state(self, task_id: str) -> StateContract:
        """
        加载任务状态

        Args:
            task_id: 任务ID

        Returns:
            StateContract 对象
        """
        state_path = self.spec_dir / task_id / 'state.json'

        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")

        state_data = self._load_json(state_path)
        # TODO: 从字典创建 StateContract
        return StateContract.create_initial(task_id, SpecContract())  # 临时

    def _load_file(self, path: Path) -> Dict[str, Any]:
        """加载契约文件（支持 JSON/YAML）"""
        if not path.exists():
            raise FileNotFoundError(f"Spec file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif path.suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """加载 JSON 文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _validate_spec(self, spec_data: Dict[str, Any]) -> None:
        """
        验证契约完整性

        验证项：
        - 必需字段存在
        - 任务名称非空
        - 提取目标有效
        - 字段定义完整
        """
        # 验证必需字段
        required_fields = ['task_id', 'task_name', 'targets']
        for field in required_fields:
            if field not in spec_data:
                raise ValueError(f"Missing required field: {field}")

        # 验证任务名称
        if not spec_data['task_name'].strip():
            raise ValueError("Task name cannot be empty")

        # 验证提取目标
        targets = spec_data.get('targets', [])
        if not targets:
            raise ValueError("At least one extraction target is required")

        # 验证每个目标
        for i, target in enumerate(targets):
            if 'name' not in target:
                raise ValueError(f"Target {i} missing 'name' field")
            if not target['name'].strip():
                raise ValueError(f"Target {i} name cannot be empty")

            # 验证字段
            fields = target.get('fields', [])
            if not fields:
                raise ValueError(f"Target '{target['name']}' must have at least one field")

            for j, field in enumerate(fields):
                # 必需字段
                if 'name' not in field:
                    raise ValueError(f"Target '{target['name']}' field {j} missing 'name'")
                if 'selector' not in field:
                    raise ValueError(f"Target '{target['name']}' field '{field.get('name', j)}' missing 'selector'")

    def save_spec(self, spec: SpecContract, output_path: Union[str, Path]) -> None:
        """保存契约到文件"""
        path = Path(output_path)
        spec_dict = spec.to_dict()

        with open(path, 'w', encoding='utf-8') as f:
            if path.suffix == '.json':
                json.dump(spec_dict, f, indent=2, ensure_ascii=False)
            elif path.suffix in ['.yaml', '.yml']:
                yaml.dump(spec_dict, f, allow_unicode=True, default_flow_style=False)

    def create_spec_template(self) -> Dict[str, Any]:
        """创建契约模板"""
        return {
            'task_id': 'task_001',
            'task_name': 'Example Task',
            'created_at': '2026-02-24T00:00:00',
            'version': '1.0',
            'extraction_type': 'single_page',
            'targets': [
                {
                    'name': 'products',
                    'fields': [
                        {
                            'name': 'title',
                            'type': 'text',
                            'selector': '.product-title',
                            'required': True,
                            'description': 'Product title'
                        },
                        {
                            'name': 'price',
                            'type': 'number',
                            'selector': '.price',
                            'required': True,
                            'description': 'Product price'
                        }
                    ]
                }
            ],
            'start_url': 'https://example.com',
            'max_pages': 100,
            'depth_limit': 3,
            'validation_rules': {},
            'anti_bot': {
                'random_delay': {'min': 1, 'max': 3},
                'user_agent_rotation': True
            },
            'completion_criteria': {
                'min_items': 10,
                'quality_threshold': 0.9
            }
        }
