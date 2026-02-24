"""
完成门禁检查器 - 战略层
基于证据验证任务完成情况
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from .contracts import SpecContract, StateContract


class GateStatus(str, Enum):
    """门禁状态"""
    PASSED = "passed"  # 通过
    FAILED = "failed"  # 失败
    PENDING = "pending"  # 待定


@dataclass
class GateDecision:
    """门禁决策"""
    status: GateStatus
    evidence_quality: float  # 证据质量评分 (0-1)
    reasons: List[str]  # 通过/失败原因
    recommendations: List[str]  # 建议
    timestamp: datetime


class CompletionGate:
    """
    完成门禁检查器

    职责：
    - 基于证据验证任务是否完成
    - 评估证据质量
    - 独立判断，不依赖代理的主观陈述

    核心特性：
    - 契约驱动：基于 Spec 中的完成标准
    - 证据验证：客观证据而非主观判断
    - 质量评估：评估提取数据的质量
    """

    def __init__(self, spec: SpecContract):
        self.spec = spec
        self.completion_criteria = spec.completion_criteria

    def check_completion(self, state: StateContract, evidence_data: Dict[str, Any]) -> GateDecision:
        """
        检查任务完成情况

        Args:
            state: 当前状态
            evidence_data: 证据数据（包含提取的数据、截图等）

        Returns:
            GateDecision: 门禁决策
        """
        reasons = []
        recommendations = []
        passed = True

        # 1. 检查提取的数据量
        if not self._check_data_quantity(state, reasons, recommendations):
            passed = False

        # 2. 检查数据质量
        quality_score = self._assess_data_quality(evidence_data, reasons, recommendations)
        if quality_score < self._get_quality_threshold():
            passed = False

        # 3. 检查字段完整性
        if not self._check_field_completeness(evidence_data, reasons, recommendations):
            passed = False

        # 4. 检查错误率
        if not self._check_error_rate(state, reasons, recommendations):
            passed = False

        status = GateStatus.PASSED if passed else GateStatus.FAILED

        return GateDecision(
            status=status,
            evidence_quality=quality_score,
            reasons=reasons,
            recommendations=recommendations,
            timestamp=datetime.now()
        )

    def _check_data_quantity(self, state: StateContract, reasons: List[str],
                            recommendations: List[str]) -> bool:
        """检查数据量是否达标"""
        min_items = self.completion_criteria.get('min_items', 10)
        actual_items = len(state.extracted_data)

        if actual_items < min_items:
            reasons.append(f"数据量不足：期望 {min_items} 条，实际 {actual_items} 条")
            recommendations.append(f"继续提取数据，至少需要 {min_items - actual_items} 条")
            return False

        reasons.append(f"数据量达标：{actual_items} / {min_items} 条")
        return True

    def _check_field_completeness(self, evidence_data: Dict[str, Any],
                                   reasons: List[str],
                                   recommendations: List[str]) -> bool:
        """检查字段完整性"""
        extracted_data = evidence_data.get('extracted_data', [])
        if not extracted_data:
            reasons.append("没有提取到任何数据")
            return False

        # 检查每个必填字段
        missing_fields = set()
        for item in extracted_data[:10]:  # 检查前10条
            for target in self.spec.targets:
                for field in target.fields:
                    if field.required and field.name not in item:
                        missing_fields.add(field.name)

        if missing_fields:
            reasons.append(f"缺少必填字段：{', '.join(missing_fields)}")
            recommendations.append(f"检查选择器是否正确：{', '.join(missing_fields)}")
            return False

        reasons.append("所有必填字段完整")
        return True

    def _check_error_rate(self, state: StateContract, reasons: List[str],
                          recommendations: List[str]) -> bool:
        """检查错误率"""
        total = state.progress.total_items
        failed = state.progress.failed_items

        if total == 0:
            return True  # 还没有开始提取

        error_rate = failed / total
        max_error_rate = self.completion_criteria.get('max_error_rate', 0.1)

        if error_rate > max_error_rate:
            reasons.append(f"错误率过高：{error_rate:.1%} (阈值 {max_error_rate:.1%})")
            recommendations.append("检查页面结构是否发生变化")
            return False

        reasons.append(f"错误率正常：{error_rate:.1%}")
        return True

    def _assess_data_quality(self, evidence_data: Dict[str, Any],
                             reasons: List[str],
                             recommendations: List[str]) -> float:
        """评估数据质量"""
        extracted_data = evidence_data.get('extracted_data', [])

        if not extracted_data:
            return 0.0

        # 质量评估维度：
        # 1. 数据完整性
        completeness_score = self._assess_completeness(extracted_data)

        # 2. 数据一致性
        consistency_score = self._assess_consistency(extracted_data)

        # 3. 数据准确性（基于字段类型验证）
        accuracy_score = self._assess_accuracy(extracted_data)

        # 综合评分
        quality_score = (
            completeness_score * 0.4 +
            consistency_score * 0.3 +
            accuracy_score * 0.3
        )

        if quality_score >= self._get_quality_threshold():
            reasons.append(f"数据质量良好：{quality_score:.2f}")
        else:
            reasons.append(f"数据质量不足：{quality_score:.2f}")
            recommendations.append("优化选择器或调整提取策略")

        return quality_score

    def _assess_completeness(self, data: List[Dict[str, Any]]) -> float:
        """评估数据完整性"""
        if not data:
            return 0.0

        total_fields = sum(len(target.fields) for target in self.spec.targets)
        required_fields = sum(
            len([f for f in target.fields if f.required])
            for target in self.spec.targets
        )

        if required_fields == 0:
            return 1.0

        filled_fields = 0
        for item in data[:50]:  # 检查前50条
            for target in self.spec.targets:
                for field in target.fields:
                    if field.required and item.get(field.name):
                        filled_fields += 1

        return min(1.0, filled_fields / (required_fields * min(50, len(data))))

    def _assess_consistency(self, data: List[Dict[str, Any]]) -> float:
        """评估数据一致性"""
        if len(data) < 2:
            return 1.0  # 单条数据视为一致

        # 检查每条数据的字段数量是否一致
        field_counts = [len(item) for item in data[:50]]
        unique_counts = len(set(field_counts))

        # 如果所有数据的字段数量相同，则一致性为1
        if unique_counts == 1:
            return 1.0

        # 否则根据字段数量的变化程度评分
        max_count = max(field_counts)
        min_count = min(field_counts)
        variation = (max_count - min_count) / max_count if max_count > 0 else 0

        return 1.0 - variation

    def _assess_accuracy(self, data: List[Dict[str, Any]]) -> float:
        """评估数据准确性"""
        if not data:
            return 0.0

        valid_count = 0
        total_count = 0

        for item in data[:50]:  # 检查前50条
            for target in self.spec.targets:
                for field in target.fields:
                    value = item.get(field.name)
                    if value is None:
                        continue

                    total_count += 1
                    if self._validate_field_value(field, value):
                        valid_count += 1

        return valid_count / total_count if total_count > 0 else 0.0

    def _validate_field_value(self, field: Any, value: Any) -> bool:
        """验证字段值是否符合类型要求"""
        # 简单的类型验证
        if field.type == 'number':
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        elif field.type == 'date':
            # 简单的日期格式检查
            if isinstance(value, str):
                return len(value) >= 8  # 至少8个字符
            return isinstance(value, (str, datetime))
        elif field.type == 'url':
            return isinstance(value, str) and value.startswith(('http://', 'https://'))
        else:
            return value is not None

    def _get_quality_threshold(self) -> float:
        """获取质量阈值"""
        return self.completion_criteria.get('quality_threshold', 0.9)

    def get_completion_criteria(self) -> Dict[str, Any]:
        """获取完成标准"""
        return {
            'min_items': self.completion_criteria.get('min_items', 10),
            'quality_threshold': self.completion_criteria.get('quality_threshold', 0.9),
            'max_error_rate': self.completion_criteria.get('max_error_rate', 0.1)
        }
