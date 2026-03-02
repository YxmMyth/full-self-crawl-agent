"""
验证智能体 - VerifyAgent
验证提取数据的质量
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import json
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from .base import AgentInterface


class VerifyAgent(AgentInterface):
    """验证智能体 - 验证提取数据质量"""

    def __init__(self, verifier=None):
        super().__init__("VerifyAgent", "verify")
        self.verifier = verifier

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行验证任务
        """
        extracted_data = context.get('extracted_data', [])
        spec = context.get('spec', {})
        extraction_metrics = context.get('extraction_metrics', {})

        try:
            # 使用传入的验证器或使用内置验证逻辑
            if self.verifier:
                quality_score = self.verifier.verify_quality(extracted_data, spec)
            else:
                quality_score = self._calculate_quality_score(extracted_data, spec)

            # 检查验证规则
            verification_result = self._check_validation_rules(extracted_data, spec)

            result = {
                'success': True,
                'quality_score': quality_score,
                'valid_items': verification_result['valid_items'],
                'total_items': len(extracted_data),
                'verification_result': verification_result,
                'extraction_metrics': extraction_metrics
            }

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'quality_score': 0.0,
                'valid_items': 0,
                'total_items': 0,
                'verification_result': {'issues': [str(e)]}
            }

    def _calculate_quality_score(self, data: List[Dict], spec: Dict) -> float:
        """计算质量分数"""
        if not data:
            return 0.0

        total_weight = 0
        weighted_score = 0

        # 基于数据完整性计算分数
        for item in data:
            item_score = 0
            field_count = 0

            targets = spec.get('targets', [])
            for target in targets:
                for field in target.get('fields', []):
                    field_name = field['name']
                    required = field.get('required', False)

                    if field_name in item and item[field_name]:
                        item_score += 1
                    elif required:
                        # 如果字段是必需的但是空的，则扣分
                        pass

                    field_count += 1

            if field_count > 0:
                item_score /= field_count
                # 对必需字段缺失进行惩罚
                for target in targets:
                    for field in target.get('fields', []):
                        field_name = field['name']
                        required = field.get('required', False)
                        if required and field_name in item and not item[field_name]:
                            item_score *= 0.7  # 降低分数

                weighted_score += item_score
                total_weight += 1

        # 考虑数据量因素
        min_expected = spec.get('completion_criteria', {}).get('min_items', 1)
        volume_factor = min(len(data) / min_expected, 1.0) if min_expected > 0 else 1.0

        final_score = (weighted_score / total_weight if total_weight > 0 else 0) * volume_factor

        # 确保分数在合理范围内
        return max(0.0, min(1.0, final_score))

    def _check_validation_rules(self, data: List[Dict], spec: Dict) -> Dict[str, Any]:
        """检查验证规则"""
        issues = []
        valid_items = 0

        targets = spec.get('targets', [])
        for i, item in enumerate(data):
            is_valid = True

            for target in targets:
                for field in target.get('fields', []):
                    field_name = field['name']
                    field_type = field.get('type', 'text')
                    required = field.get('required', False)

                    value = item.get(field_name)

                    if required and (value is None or value == ''):
                        issues.append(f"第{i}项: 必需字段 '{field_name}' 缺失")
                        is_valid = False
                        continue

                    if value and not self._validate_field_type(value, field_type):
                        issues.append(f"第{i}项: 字段 '{field_name}' 类型不匹配，期望 {field_type}")
                        is_valid = False

            if is_valid:
                valid_items += 1

        return {
            'issues': issues,
            'valid_items': valid_items,
            'total_items': len(data),
            'valid_percentage': (valid_items / len(data)) * 100 if data else 0
        }

    def _validate_field_type(self, value: Any, field_type: str) -> bool:
        """验证字段类型"""
        if field_type == 'text':
            return isinstance(value, str)
        elif field_type == 'number':
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        elif field_type == 'url':
            import re
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            )
            return bool(url_pattern.match(str(value)))
        elif field_type == 'email':
            import re
            email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
            return bool(email_pattern.match(str(value)))
        else:
            # 默认为文本类型
            return isinstance(value, str)

    def get_description(self) -> str:
        return "验证提取数据的质量和完整性"

    def can_handle(self, context: Dict[str, Any]) -> bool:
        return 'extracted_data' in context