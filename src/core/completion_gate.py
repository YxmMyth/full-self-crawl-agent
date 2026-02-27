"""
完成门禁检查器 - 战略层
基于证据验证任务完成情况

根据 IMPLEMENTATION.md 第2.1.3节的完整设计
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


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

    def __init__(self):
        self.failed_gates = []
        self.passed_gates = []

    def check(self, state: Dict[str, Any], spec: Dict[str, Any]) -> bool:
        """
        检查是否满足完成门禁

        Args:
            state: 当前状态
            spec: Spec契约

        Returns:
            bool: 是否通过所有门禁
        """
        self.failed_gates = []
        self.passed_gates = []

        # 优先使用 completion_gate，否则从 completion_criteria 生成
        gate_conditions = spec.get('completion_gate', [])
        if not gate_conditions:
            gate_conditions = self._build_gates_from_criteria(spec.get('completion_criteria', {}))

        for gate_condition in gate_conditions:
            if self._evaluate(gate_condition, state):
                self.passed_gates.append(gate_condition)
            else:
                self.failed_gates.append(gate_condition)

        if self.failed_gates:
            state['gate_failed'] = True
            state['failed_gates'] = self.failed_gates
            state['passed_gates'] = self.passed_gates
            return False

        state['gate_passed'] = True
        state['passed_gates'] = self.passed_gates
        return True

    def _build_gates_from_criteria(self, criteria: Dict[str, Any]) -> List[str]:
        """从 completion_criteria 构建门禁条件"""
        gates = []

        # 最小数据量
        min_items = criteria.get('min_items', 1)
        gates.append(f'sample_count >= {min_items}')

        # 质量阈值
        quality_threshold = criteria.get('quality_threshold', 0.5)
        gates.append(f'quality_score >= {quality_threshold}')

        return gates

    def _evaluate(self, condition: str, state: Dict[str, Any]) -> bool:
        """
        评估门禁条件

        支持的门禁条件：
        - html_snapshot_exists: HTML快照存在
        - sense_analysis_valid: 感知分析有效
        - code_syntax_valid: 代码语法正确
        - execution_success: 执行成功
        - quality_score >= X: 质量分数 >= 阈值
        - sample_count >= X: 样本数 >= 阈值
        """
        if condition == 'html_snapshot_exists':
            return state.get('html_snapshot') is not None

        elif condition == 'sense_analysis_valid':
            analysis = state.get('sense_analysis')
            return analysis is not None and len(analysis) > 0

        elif condition == 'code_syntax_valid':
            return state.get('syntax_valid', False)

        elif condition == 'execution_success':
            result = state.get('execution_result', {})
            return result.get('success', False) and result.get('data') is not None

        elif condition.startswith('quality_score >='):
            try:
                threshold = float(condition.split('>=')[1].strip())
                quality = state.get('quality_score', 0)
                return quality >= threshold
            except (ValueError, IndexError):
                return False

        elif condition.startswith('sample_count >='):
            try:
                threshold = int(condition.split('>=')[1].strip())
                data = state.get('sample_data', [])
                return len(data) >= threshold
            except (ValueError, IndexError):
                return False

        raise ValueError(f"Unknown gate condition: {condition}")

    def get_failed_gates(self) -> List[str]:
        """获取失败的门禁"""
        return self.failed_gates

    def get_passed_gates(self) -> List[str]:
        """获取通过的门禁"""
        return self.passed_gates


class GateDecision:
    """
    门禁决策器 - 基于门禁检查结果做决策

    根据 IMPLEMENTATION.md 第1.3.3节实现
    """

    def __init__(self):
        self.completion_gate = CompletionGate()

    def decide(self, state: Dict[str, Any], spec: Dict[str, Any]) -> str:
        """
        门禁决策

        返回决策类型：
        - "complete": 任务完成
        - "soal_repair": 执行失败，进入SOAL修复流程
        - "reflect_and_retry": 质量不达标但可修复
        - "terminate": 无法修复，终止任务
        - "retry_with_delay": 网络问题，延迟后重试
        """
        gate_passed = self.completion_gate.check(state, spec)

        if not gate_passed:
            failed_gates = state.get("failed_gates", [])

            # 执行失败
            if "execution_success" in failed_gates:
                return "soal_repair"

            # 质量分数不达标
            elif any("quality_score" in str(gate) for gate in failed_gates):
                quality = state.get("quality_score", 0)
                if quality >= 0.4:
                    return "reflect_and_retry"
                else:
                    return "terminate"

            # HTML快照不存在
            elif "html_snapshot_exists" in failed_gates:
                return "retry_with_delay"

            # 其他情况终止
            else:
                return "terminate"

        return "complete"

    def get_decision_reason(self, decision: str, state: Dict[str, Any]) -> str:
        """获取决策原因"""
        reasons = {
            "complete": "所有门禁条件满足",
            "soal_repair": "执行失败，进入SOAL修复流程",
            "reflect_and_retry": "质量不达标但可修复，反思后重试",
            "terminate": "无法修复，终止任务",
            "retry_with_delay": "网络问题，延迟后重试"
        }
        return reasons.get(decision, "未知决策")


class Verifier:
    """
    结果验证器 - 独立验证执行结果的正确性

    根据 IMPLEMENTATION.md 第1.3.1节实现
    """

    def verify_sense(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """验证感知分析"""
        issues = []

        if not analysis.get('page_type'):
            issues.append("缺少页面类型")

        if not analysis.get('selectors'):
            issues.append("缺少选择器")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'score': max(0.0, 1.0 - len(issues) * 0.1)
        }

    def verify_code(self, code: str) -> Dict[str, Any]:
        """验证生成的代码"""
        import ast

        try:
            ast.parse(code)
            syntax_valid = True
            issues = []
        except SyntaxError as e:
            syntax_valid = False
            issues = [str(e)]

        return {
            'valid': syntax_valid,
            'syntax_valid': syntax_valid,
            'issues': issues
        }

    def verify_execution(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证执行结果"""
        data = result.get('data', [])

        return {
            'valid': len(data) > 0,
            'has_data': len(data) > 0,
            'data_count': len(data),
            'issues': [] if len(data) > 0 else ["没有返回数据"]
        }

    def verify_quality(self, data: List[Any], expected_fields: List[str]) -> Dict[str, Any]:
        """验证数据质量"""
        if not data:
            return {'valid': False, 'score': 0.0, 'issues': ['无数据']}

        # 检查字段完整性
        missing_fields = []
        for item in data[:5]:  # 检查前5个样本
            for field in expected_fields:
                if field not in item:
                    missing_fields.append(field)

        completeness = 1.0 - (len(set(missing_fields)) / max(len(expected_fields), 1))

        # 检查数据一致性
        consistency = self._check_consistency(data)

        score = (completeness + consistency) / 2

        return {
            'valid': score >= 0.6,
            'score': score,
            'completeness': completeness,
            'consistency': consistency,
            'issues': missing_fields if missing_fields else []
        }

    def _check_consistency(self, data: List[Any]) -> float:
        """检查数据一致性"""
        if len(data) < 2:
            return 1.0

        # 检查字段结构是否一致
        first_keys = set(data[0].keys())
        inconsistent_count = 0

        for item in data[1:5]:
            if set(item.keys()) != first_keys:
                inconsistent_count += 1

        return 1.0 - (inconsistent_count / min(len(data) - 1, 4))


class EvidenceCollector:
    """
    证据收集器 - 收集各阶段的结构化证据

    根据 IMPLEMENTATION.md 第1.3.2节实现
    """

    def __init__(self, output_dir: str):
        from pathlib import Path
        import os

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_index = {}

    def collect_sense(self, html: str, analysis: Dict[str, Any]):
        """收集感知阶段证据"""
        sense_dir = self.output_dir / 'sense'
        sense_dir.mkdir(exist_ok=True)

        with open(sense_dir / 'html_snapshot.html', 'w', encoding='utf-8') as f:
            f.write(html[:100000])

        with open(sense_dir / 'analysis.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        self.evidence_index['sense'] = str(sense_dir)

    def collect_plan(self, code: str, reasoning: str):
        """收集规划阶段证据"""
        plan_dir = self.output_dir / 'plan'
        plan_dir.mkdir(exist_ok=True)

        with open(plan_dir / 'generated_code.py', 'w', encoding='utf-8') as f:
            f.write(code)

        with open(plan_dir / 'reasoning.md', 'w', encoding='utf-8') as f:
            f.write(reasoning)

        self.evidence_index['plan'] = str(plan_dir)

    def collect_act(self, result: Dict[str, Any], logs: List[Any], screenshots: List[str]):
        """收集执行阶段证据"""
        act_dir = self.output_dir / 'act'
        act_dir.mkdir(exist_ok=True)

        with open(act_dir / 'result.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(result, f, ensure_ascii=False, indent=2)

        with open(act_dir / 'execution_log.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(logs, f, ensure_ascii=False, indent=2)

        screenshot_dir = act_dir / 'screenshots'
        screenshot_dir.mkdir(exist_ok=True)
        for i, screenshot in enumerate(screenshots):
            import base64
            img_data = base64.b64decode(screenshot)
            with open(screenshot_dir / f'screenshot_{i}.png', 'wb') as f:
                f.write(img_data)

        self.evidence_index['act'] = str(act_dir)

    def collect_verify(self, sample_data: List[Any], quality_report: Dict[str, Any], issues: List[str]):
        """收集验证阶段证据"""
        verify_dir = self.output_dir / 'verify'
        verify_dir.mkdir(exist_ok=True)

        with open(verify_dir / 'sample_data.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(sample_data, f, ensure_ascii=False, indent=2)

        with open(verify_dir / 'quality_report.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(quality_report, f, ensure_ascii=False, indent=2)

        with open(verify_dir / 'issues.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(issues))

        self.evidence_index['verify'] = str(verify_dir)

    def save_index(self):
        """保存证据索引"""
        with open(self.output_dir / 'evidence_index.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(self.evidence_index, f, ensure_ascii=False, indent=2)

    def get_evidence_path(self) -> str:
        """获取证据包路径"""
        return str(self.output_dir)
