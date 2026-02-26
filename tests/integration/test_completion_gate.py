"""
完成门禁单元测试
"""

import pytest
import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_completion_gate_basic():
    """测试完成门禁基础功能"""
    from src.core.completion_gate import CompletionGate
    
    gate = CompletionGate()
    
    # 测试门禁条件
    state = {'html_snapshot': '<html></html>'}
    spec = {'completion_gate': ['html_snapshot_exists']}
    
    result = gate.check(state, spec)
    assert result is True
    assert len(gate.get_passed_gates()) == 1


def test_completion_gate_failed():
    """测试门禁失败情况"""
    from src.core.completion_gate import CompletionGate
    
    gate = CompletionGate()
    
    state = {}  # 没有html_snapshot
    spec = {'completion_gate': ['html_snapshot_exists']}
    
    result = gate.check(state, spec)
    assert result is False
    assert len(gate.get_failed_gates()) == 1


def test_quality_score_gate():
    """测试质量分数门禁"""
    from src.core.completion_gate import CompletionGate
    
    gate = CompletionGate()
    
    state = {'quality_score': 0.7}
    spec = {'completion_gate': ['quality_score >= 0.6']}
    
    result = gate.check(state, spec)
    assert result is True


def test_quality_score_gate_failed():
    """测试质量分数门禁失败"""
    from src.core.completion_gate import CompletionGate
    
    gate = CompletionGate()
    
    state = {'quality_score': 0.5}
    spec = {'completion_gate': ['quality_score >= 0.6']}
    
    result = gate.check(state, spec)
    assert result is False


def test_sample_count_gate():
    """测试样本数量门禁"""
    from src.core.completion_gate import CompletionGate
    
    gate = CompletionGate()
    
    state = {'sample_data': [1, 2, 3, 4, 5]}
    spec = {'completion_gate': ['sample_count >= 5']}
    
    result = gate.check(state, spec)
    assert result is True


def test_gate_decision():
    """测试门禁决策"""
    from src.core.completion_gate import GateDecision
    
    decision_maker = GateDecision()
    
    # 测试通过情况
    state = {'html_snapshot': 'test', 'gate_passed': True}
    spec = {'completion_gate': ['html_snapshot_exists']}
    
    decision = decision_maker.decide(state, spec)
    assert decision == 'complete'


def test_gate_decision_failed():
    """测试门禁决策失败情况"""
    from src.core.completion_gate import GateDecision
    
    decision_maker = GateDecision()
    
    # 执行失败
    state = {
        'failed_gates': ['execution_success'],
        'gate_failed': True
    }
    spec = {'completion_gate': ['execution_success']}
    
    decision = decision_maker.decide(state, spec)
    assert decision == 'soal_repair'


def test_gate_decision_quality():
    """测试门禁决策 - 质量不达标但可修复"""
    from src.core.completion_gate import GateDecision
    
    decision_maker = GateDecision()
    
    state = {
        'quality_score': 0.5,
        'failed_gates': ['quality_score >= 0.6'],
        'gate_failed': True
    }
    spec = {'completion_gate': ['quality_score >= 0.6']}
    
    decision = decision_maker.decide(state, spec)
    assert decision == 'reflect_and_retry'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
