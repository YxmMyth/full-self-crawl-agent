"""
验证层组件
包括 Verifier、EvidenceCollector、GateDecision
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib


class VerificationStatus(str, Enum):
    """验证状态"""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class EvidenceType(str, Enum):
    """证据类型"""
    SCREENSHOT = "screenshot"  # 截图
    HTML_SNAPSHOT = "html_snapshot"  # HTML 快照
    EXTRACTED_DATA = "extracted_data"  # 提取的数据
    SELECTOR_TEST = "selector_test"  # 选择器测试结果
    PERFORMANCE_METRICS = "performance_metrics"  # 性能指标
    ERROR_LOG = "error_log"  # 错误日志
    NETWORK_REQUEST = "network_request"  # 网络请求记录


@dataclass
class Evidence:
    """证据数据"""
    type: EvidenceType
    data: Any
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'type': self.type.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'metadata': self.metadata,
            'data': self._serialize_data()
        }

    def _serialize_data(self) -> Any:
        """序列化数据"""
        if isinstance(self.data, (str, int, float, bool, list, dict)):
            return self.data
        elif hasattr(self.data, 'to_dict'):
            return self.data.to_dict()
        else:
            return str(self.data)


# ==================== 证据收集器 ====================

class EvidenceCollector:
    """
    证据收集器

    职责：
    - 收集各种类型的证据
    - 存储证据
    - 检索证据
    - 导出证据报告

    证据类型：
    - 页面截图
    - HTML 快照
    - 提取的数据
    - 选择器测试结果
    - 性能指标
    - 错误日志
    - 网络请求记录
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir
        self.evidences: List[Evidence] = []
        self.evidence_index: Dict[str, List[int]] = {}  # 按类型索引

    def add_screenshot(self, screenshot_data: bytes, url: str,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加截图证据"""
        evidence = Evidence(
            type=EvidenceType.SCREENSHOT,
            data=screenshot_data,
            timestamp=datetime.now(),
            source=url,
            metadata=metadata or {}
        )
        return self._add_evidence(evidence)

    def add_html_snapshot(self, html: str, url: str,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加 HTML 快照"""
        evidence = Evidence(
            type=EvidenceType.HTML_SNAPSHOT,
            data=html,
            timestamp=datetime.now(),
            source=url,
            metadata=metadata or {}
        )
        return self._add_evidence(evidence)

    def add_extracted_data(self, data: List[Dict[str, Any]], url: str,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加提取的数据"""
        evidence = Evidence(
            type=EvidenceType.EXTRACTED_DATA,
            data=data,
            timestamp=datetime.now(),
            source=url,
            metadata=metadata or {}
        )
        return self._add_evidence(evidence)

    def add_selector_test(self, selector: str, result: List[str],
                          url: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加选择器测试结果"""
        evidence = Evidence(
            type=EvidenceType.SELECTOR_TEST,
            data={
                'selector': selector,
                'result': result,
                'count': len(result)
            },
            timestamp=datetime.now(),
            source=url,
            metadata=metadata or {}
        )
        return self._add_evidence(evidence)

    def add_performance_metrics(self, metrics: Dict[str, Any],
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加性能指标"""
        evidence = Evidence(
            type=EvidenceType.PERFORMANCE_METRICS,
            data=metrics,
            timestamp=datetime.now(),
            source='system',
            metadata=metadata or {}
        )
        return self._add_evidence(evidence)

    def add_error_log(self, error: str, context: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加错误日志"""
        evidence = Evidence(
            type=EvidenceType.ERROR_LOG,
            data={
                'error': error,
                'context': context or {}
            },
            timestamp=datetime.now(),
            source='system',
            metadata=metadata or {}
        )
        return self._add_evidence(evidence)

    def add_network_request(self, url: str, method: str, status: int,
                            response_time: float,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加网络请求记录"""
        evidence = Evidence(
            type=EvidenceType.NETWORK_REQUEST,
            data={
                'url': url,
                'method': method,
                'status': status,
                'response_time': response_time
            },
            timestamp=datetime.now(),
            source='network',
            metadata=metadata or {}
        )
        return self._add_evidence(evidence)

    def _add_evidence(self, evidence: Evidence) -> str:
        """添加证据到存储"""
        self.evidences.append(evidence)

        # 更新索引
        evidence_type = evidence.type.value
        if evidence_type not in self.evidence_index:
            self.evidence_index[evidence_type] = []
        self.evidence_index[evidence_type].append(len(self.evidences) - 1)

        # 生成证据 ID
        evidence_id = self._generate_evidence_id(evidence)

        # 持久化（如果配置了存储目录）
        if self.storage_dir:
            self._persist_evidence(evidence, evidence_id)

        return evidence_id

    def _generate_evidence_id(self, evidence: Evidence) -> str:
        """生成证据 ID"""
        content = f"{evidence.type.value}_{evidence.timestamp.isoformat()}_{evidence.source}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _persist_evidence(self, evidence: Evidence, evidence_id: str) -> None:
        """持久化证据"""
        import os
        os.makedirs(self.storage_dir, exist_ok=True)

        filepath = os.path.join(self.storage_dir, f"{evidence_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(evidence.to_dict(), f, indent=2, ensure_ascii=False)

    def get_evidences_by_type(self, evidence_type: EvidenceType) -> List[Evidence]:
        """按类型获取证据"""
        indices = self.evidence_index.get(evidence_type.value, [])
        return [self.evidences[i] for i in indices]

    def get_recent_evidences(self, count: int = 10) -> List[Evidence]:
        """获取最近的证据"""
        return self.evidences[-count:]

    def get_evidence_report(self) -> Dict[str, Any]:
        """生成证据报告"""
        report = {
            'total_evidences': len(self.evidences),
            'by_type': {},
            'timeline': []
        }

        # 按类型统计
        for evidence_type in EvidenceType:
            evidences = self.get_evidences_by_type(evidence_type)
            report['by_type'][evidence_type.value] = len(evidences)

        # 时间线
        for evidence in self.evidences[-20:]:
            report['timeline'].append({
                'time': evidence.timestamp.isoformat(),
                'type': evidence.type.value,
                'source': evidence.source
            })

        return report

    def collect_plan(self, generated_code: str, strategy: str) -> str:
        """收集规划阶段的证据"""
        evidence = Evidence(
            type=EvidenceType.EXTRACTED_DATA,
            data={
                'generated_code': generated_code,
                'strategy': strategy
            },
            timestamp=datetime.now(),
            source='planning',
            metadata={'phase': 'planning'}
        )
        return self._add_evidence(evidence)

    def save_index(self) -> None:
        """保存证据索引"""
        if not self.storage_dir:
            return

        import os
        os.makedirs(self.storage_dir, exist_ok=True)

        index_data = {
            'total_evidences': len(self.evidences),
            'by_type': self.evidence_index,
            'evidences': [e.to_dict() for e in self.evidences]
        }

        filepath = os.path.join(self.storage_dir, 'index.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)


# ==================== 结果验证器 ====================

class Verifier:
    """
    结果验证器

    职责：
    - 验证提取的数据质量
    - 检查数据完整性
    - 验证数据格式
    - 评估数据准确性

    验证维度：
    - 完整性：必填字段是否完整
    - 格式：数据格式是否符合预期
    - 一致性：多条数据间是否一致
    - 准确性：数据值是否合理
    """

    def __init__(self, spec):
        self.spec = spec

    def verify(self, data: List[Dict[str, Any]],
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        验证提取的数据

        Returns:
            验证结果字典
        """
        context = context or {}

        results = {
            'status': VerificationStatus.PASSED,
            'total_items': len(data),
            'valid_items': 0,
            'invalid_items': 0,
            'issues': [],
            'scores': {}
        }

        if not data:
            results['status'] = VerificationStatus.FAILED
            results['issues'].append({
                'level': 'error',
                'message': '没有提取到任何数据'
            })
            return results

        # 逐条验证
        for i, item in enumerate(data):
            item_result = self._verify_item(item, context)
            if item_result['valid']:
                results['valid_items'] += 1
            else:
                results['invalid_items'] += 1
                results['issues'].extend([
                    {
                        'item_index': i,
                        **issue
                    }
                    for issue in item_result['issues']
                ])

        # 计算各项评分
        results['scores'] = {
            'completeness': self._score_completeness(data),
            'format': self._score_format(data),
            'consistency': self._score_consistency(data),
            'accuracy': self._score_accuracy(data, context)
        }

        # 确定整体状态
        valid_ratio = results['valid_items'] / results['total_items'] if results['total_items'] > 0 else 0
        if valid_ratio == 1:
            results['status'] = VerificationStatus.PASSED
        elif valid_ratio >= 0.5:
            results['status'] = VerificationStatus.PARTIAL
        else:
            results['status'] = VerificationStatus.FAILED

        return results

    def _verify_item(self, item: Dict[str, Any],
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """验证单条数据"""
        issues = []
        valid = True

        # 检查必填字段 (支持字典格式的 spec)
        targets = self._get_targets()
        for target in targets:
            for field in target.get('fields', []):
                if field.get('required', False):
                    field_name = field.get('name')
                    if field_name not in item or not item[field_name]:
                        issues.append({
                            'level': 'error',
                            'type': 'missing_required_field',
                            'message': f'缺少必填字段: {field_name}',
                            'field': field_name
                        })
                        valid = False

                    # 检查格式
                    elif field.get('type'):
                        field_type = field.get('type')
                        if isinstance(field_type, str):
                            format_valid = self._validate_format(
                                item[field_name],
                                field_type,
                                field.get('validation_rules', [])
                            )
                        else:
                            format_valid = True  # type 不是字符串时跳过
                        if not format_valid:
                            issues.append({
                                'level': 'warning',
                                'type': 'invalid_format',
                                'message': f'字段格式不正确: {field_name}',
                                'field': field_name,
                                'expected_type': field_type
                            })

        return {
            'valid': valid,
            'issues': issues
        }

    def _get_targets(self) -> List[Dict]:
        """获取目标列表（兼容字典和对象格式）"""
        if isinstance(self.spec, dict):
            return self.spec.get('targets', [])
        elif hasattr(self.spec, 'targets'):
            targets = self.spec.targets
            # 如果是对象列表，转换为字典
            if targets and hasattr(targets[0], 'fields'):
                return [
                    {
                        'name': t.name if hasattr(t, 'name') else '',
                        'fields': [
                            {
                                'name': f.name if hasattr(f, 'name') else f.get('name'),
                                'type': f.type if hasattr(f, 'type') else f.get('type'),
                                'required': f.required if hasattr(f, 'required') else f.get('required', False),
                                'validation_rules': f.validation_rules if hasattr(f, 'validation_rules') else f.get('validation_rules', [])
                            }
                            for f in (t.fields if hasattr(t, 'fields') else t.get('fields', []))
                        ]
                    }
                    for t in targets
                ]
            return targets
        return []

    def _validate_format(self, value: Any, expected_type: str,
                         rules: Optional[List[str]] = None) -> bool:
        """验证字段格式"""
        if value is None:
            return True  # None 由 required 检查

        try:
            if expected_type == 'number':
                float(value)
            elif expected_type == 'date':
                # 简单的日期验证
                str(value)
            elif expected_type == 'url':
                if not str(value).startswith(('http://', 'https://')):
                    return False

            # 自定义验证规则
            if rules:
                for rule in rules:
                    if not self._apply_validation_rule(value, rule):
                        return False

            return True
        except (ValueError, TypeError):
            return False

    def _apply_validation_rule(self, value: Any, rule: str) -> bool:
        """应用验证规则"""
        # 简单的规则解析
        # 实际应用中应该使用更复杂的规则引擎
        if '>' in rule:
            threshold = float(rule.split('>')[1].strip())
            return float(value) > threshold
        elif '<' in rule:
            threshold = float(rule.split('<')[1].strip())
            return float(value) < threshold
        elif 'len' in rule:
            min_len = int(rule.split('(')[1].split(')')[0].split(',')[0].strip())
            return len(str(value)) >= min_len

        return True

    def _score_completeness(self, data: List[Dict[str, Any]]) -> float:
        """计算完整性评分"""
        if not data:
            return 0.0

        required_fields = set()
        targets = self._get_targets()
        for target in targets:
            for field in target.get('fields', []):
                if field.get('required', False):
                    required_fields.add(field.get('name'))

        if not required_fields:
            return 1.0

        total_required = len(required_fields) * len(data)
        filled = sum(
            1 for item in data
            for field in required_fields
            if item.get(field)
        )

        return filled / total_required if total_required > 0 else 1.0

    def _score_format(self, data: List[Dict[str, Any]]) -> float:
        """计算格式评分"""
        if not data:
            return 0.0

        total_fields = 0
        valid_fields = 0

        targets = self._get_targets()
        for item in data:
            for target in targets:
                for field in target.get('fields', []):
                    field_name = field.get('name')
                    if field_name in item:
                        total_fields += 1
                        field_type = field.get('type')
                        if isinstance(field_type, str):
                            if self._validate_format(
                                item[field_name],
                                field_type,
                                field.get('validation_rules', [])
                            ):
                                valid_fields += 1
                        else:
                            valid_fields += 1

        return valid_fields / total_fields if total_fields > 0 else 1.0

    def _score_consistency(self, data: List[Dict[str, Any]]) -> float:
        """计算一致性评分"""
        if len(data) < 2:
            return 1.0

        # 检查字段数量一致性
        field_counts = [len(item) for item in data]
        unique_counts = len(set(field_counts))

        return 1.0 - (unique_counts - 1) / max(field_counts) if field_counts else 1.0

    def _score_accuracy(self, data: List[Dict[str, Any]],
                        context: Dict[str, Any]) -> float:
        """计算准确性评分"""
        # 这里可以添加更复杂的准确性检查
        # 例如：数值范围检查、日期合理性检查等

        score = 1.0

        targets = self._get_targets()
        for item in data:
            for target in targets:
                for field in target.get('fields', []):
                    field_name = field.get('name')
                    if field_name in item and field.get('validation_rules'):
                        for rule in field.get('validation_rules', []):
                            if not self._apply_validation_rule(item[field_name], rule):
                                score -= 0.1

        return max(0.0, score)


# ==================== 门禁决策器 ====================

class GateDecision:
    """
    门禁决策器

    职责：
    - 基于验证结果做出决策
    - 决定是否继续、重试或终止
    - 提供决策依据
    """

    def __init__(self, spec):
        self.spec = spec
        self.completion_criteria = spec.completion_criteria

    def decide(self, verification_result: Dict[str, Any],
               state: Any) -> Dict[str, Any]:
        """
        做出门禁决策

        Returns:
            决策字典
        """
        decision = {
            'action': 'continue',  # continue, retry, terminate
            'reason': '',
            'recommendations': []
        }

        # 检查验证状态
        verification_status = verification_result.get('status')

        if verification_status == VerificationStatus.FAILED:
            decision['action'] = 'retry'
            decision['reason'] = '验证失败，需要调整策略'
            decision['recommendations'] = self._get_retry_recommendations(
                verification_result
            )
            return decision

        if verification_status == VerificationStatus.PARTIAL:
            # 检查部分数据是否足够
            valid_ratio = verification_result['valid_items'] / verification_result['total_items']
            min_quality = self.completion_criteria.get('quality_threshold', 0.9)

            if valid_ratio >= min_quality:
                decision['action'] = 'continue'
                decision['reason'] = f'数据质量可接受 ({valid_ratio:.1%})'
            else:
                decision['action'] = 'retry'
                decision['reason'] = f'数据质量不足 ({valid_ratio:.1%})'
                decision['recommendations'] = self._get_retry_recommendations(
                    verification_result
                )
            return decision

        # 检查是否完成
        is_complete = self._check_completion(verification_result, state)

        if is_complete:
            decision['action'] = 'terminate'
            decision['reason'] = '任务完成'
        else:
            decision['action'] = 'continue'
            decision['reason'] = '继续提取数据'

        return decision

    def _check_completion(self, verification_result: Dict[str, Any],
                          state: Any) -> bool:
        """检查任务是否完成"""
        # 检查数据量
        min_items = self.completion_criteria.get('min_items', 10)
        actual_items = verification_result.get('valid_items', 0)

        if actual_items < min_items:
            return False

        # 检查质量
        quality_threshold = self.completion_criteria.get('quality_threshold', 0.9)
        scores = verification_result.get('scores', {})
        avg_score = sum(scores.values()) / len(scores) if scores else 0

        if avg_score < quality_threshold:
            return False

        # 检查错误率
        max_error_rate = self.completion_criteria.get('max_error_rate', 0.1)
        total_items = verification_result.get('total_items', 1)
        error_rate = verification_result.get('invalid_items', 0) / total_items

        if error_rate > max_error_rate:
            return False

        return True

    def _get_retry_recommendations(self, verification_result: Dict[str, Any]) -> List[str]:
        """获取重试建议"""
        recommendations = []

        # 分析问题类型
        issue_types = {}
        for issue in verification_result.get('issues', []):
            issue_type = issue.get('type', 'unknown')
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

        # 根据问题类型给出建议
        if 'missing_required_field' in issue_types:
            recommendations.append(
                '有必填字段缺失，请检查选择器是否正确'
            )

        if 'invalid_format' in issue_types:
            recommendations.append(
                '数据格式不正确，可能需要调整选择器或添加数据清洗'
            )

        # 通用建议
        recommendations.append('建议先感知页面结构，确认选择器')
        recommendations.append('可以尝试使用更宽松的选择器')

        return recommendations
